"""Numerical regression tests; run with python -m unittest discover -s tests."""

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from vehicle_reconstruction_gcpr16.prior import ShapePrior, load_eigen_matrix
from vehicle_reconstruction_gcpr16.optimization import (
    fit_vehicle, objective, pose_residuals, rotation_y, shape_residuals,
)


def finite_difference(function, value, epsilon=1.e-6):
    value = np.asarray(value, dtype=float)
    columns = []
    for axis in range(value.size):
        offset = np.zeros_like(value)
        offset[axis] = epsilon
        columns.append((function(value + offset) - function(value - offset)) / (2 * epsilon))
    return np.stack(columns, axis=-1)


def polynomial_prior():
    """A globally trilinear field, so exact interpolation is independently known."""
    prior = ShapePrior.__new__(ShapePrior)
    prior.resolution = (60, 40, 60)
    prior.origin = np.array([-3., -3., -3.])
    prior.unit = np.array([.1, .1, .1])
    prior.k = 2
    prior.S = np.array([2., 3.])
    x, y, z = np.meshgrid(*(prior.origin[i] + .1 * np.arange(d)
                           for i, d in enumerate(prior.resolution)), indexing='ij')
    prior.mean = (x + .4*y - .3*z + .2*x*y - .15*y*z + .05*x*z + .02*x*y*z).ravel()
    prior.V = np.stack(((x-y+.2*z).ravel(), (1.+.03*x*y*z).ravel()), axis=1)
    return prior


class EigenMatrixTests(unittest.TestCase):
    def test_both_header_widths_and_column_major(self):
        matrix = np.array([[1., 2., 3.], [4., 5., 6.]])
        with tempfile.TemporaryDirectory() as directory:
            for fmt in ('<ii', '<qq'):
                with self.subTest(format=fmt):
                    path = Path(directory) / 'test.bin'
                    path.write_bytes(struct.pack(fmt, 2, 3) + matrix.astype('<f8').tobytes(order='F'))
                    np.testing.assert_array_equal(load_eigen_matrix(path), matrix)

    def test_invalid_headers_sizes_and_values(self):
        bad = [b'', struct.pack('<qq', 0, 1), struct.pack('<ii', -1, 1),
               struct.pack('<qq', 2, 2) + bytes(24),
               struct.pack('<ii', 1, 1) + bytes(9),
               struct.pack('<qq', 2**50, 2**50),
               struct.pack('<ii', 1, 1) + struct.pack('<d', np.nan),
               struct.pack('<qq', 1, 1) + struct.pack('<d', np.inf)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.bin'
            for payload in bad:
                with self.subTest(payload_size=len(payload)):
                    path.write_bytes(payload)
                    with self.assertRaises(ValueError):
                        load_eigen_matrix(path)

    def test_prior_checks_full_matrix_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('S.bin', 'V.bin', 'mean_shape.bin'):
                (Path(directory) / name).write_bytes(struct.pack('<qqd', 1, 1, 1.))
            with self.assertRaisesRegex(ValueError, 'V.bin'):
                ShapePrior(directory, k=1)


class PriorSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = polynomial_prior()
        cls.points = np.array([[-1.173, -1.123, .537], [.337, -.827, -.543],
                               [1.541, .357, -1.273]])
        cls.z = np.array([.13, -.27])

    def test_trilinear_against_polynomial(self):
        x, y, z = self.points.T
        expected = (x+.4*y-.3*z+.2*x*y-.15*y*z+.05*x*z+.02*x*y*z +
                    self.z[0]*(x-y+.2*z) + self.z[1]*(1.+.03*x*y*z))
        expected_gradient = np.stack((
            1.+.2*y+.05*z+.02*y*z+self.z[0]+self.z[1]*.03*y*z,
            .4+.2*x-.15*z+.02*x*z-self.z[0]+self.z[1]*.03*x*z,
            -.3-.15*y+.05*x+.02*x*y+.2*self.z[0]+self.z[1]*.03*x*y), axis=1)
        values, gradient = self.prior.sample(self.points, self.z, mode='pose')
        np.testing.assert_allclose(values, expected, atol=1.e-13)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1.e-13)
        no_jac_values, no_jac = self.prior.sample(self.points, self.z, with_jacobian=False)
        np.testing.assert_allclose(no_jac_values, expected, atol=1.e-13)
        self.assertIsNone(no_jac)

    def test_shape_jacobian_finite_difference(self):
        _, jacobian = self.prior.sample(self.points, self.z)
        numeric = finite_difference(lambda z: self.prior.sample(self.points, z)[0], self.z)
        np.testing.assert_allclose(jacobian, numeric, atol=1.e-9, rtol=1.e-8)

    def test_spatial_jacobian_finite_difference(self):
        _, gradient = self.prior.sample(self.points, self.z, mode='pose')
        for point, analytic in zip(self.points, gradient):
            numeric = finite_difference(
                lambda q: self.prior.sample(q[None], self.z, mode='pose')[0], point)[0]
            np.testing.assert_allclose(analytic, numeric, atol=1.e-9, rtol=1.e-8)

    def test_distinct_boundaries_and_outside_derivatives(self):
        # Lower-most voxel belongs only to shape mode. Highest voxel lower
        # index dim-2 belongs to both; dim-1 belongs to neither.
        points = np.array([[-2.95, -1., 0.], [0., -2.95, 0.], [0., -1., -2.95],
                           [2.85, .85, 2.85], [2.95, 0., 0.], [0., .95, 0.],
                           [0., 0., 2.95], [-3.01, 0., 0.]])
        shape, shape_j = self.prior.sample(points, self.z, mode='shape')
        pose, pose_j = self.prior.sample(points, self.z, mode='pose')
        np.testing.assert_array_equal(shape[4:], np.full(4, 3.))
        np.testing.assert_array_equal(shape_j[4:], np.zeros((4, 2)))
        outside = np.array([0, 1, 2, 4, 5, 6, 7])
        norms = np.linalg.norm(points[outside], axis=1)
        np.testing.assert_allclose(pose[outside], norms+3.)
        np.testing.assert_allclose(pose_j[outside], points[outside]/norms[:, None])
        self.assertAlmostEqual(shape[3], pose[3])
        self.assertGreater(np.linalg.norm(shape_j[:3]), 0.)
        numeric = finite_difference(lambda q: self.prior.sample(q[None], self.z, 'pose')[0], points[0])[0]
        np.testing.assert_allclose(pose_j[0], numeric, atol=1.e-9)
        lower = self.prior.origin[None]
        upper = self.prior.origin + (np.asarray(self.prior.resolution)-1)*self.prior.unit
        _, j_lower = self.prior.sample(lower, self.z)
        self.assertGreater(np.linalg.norm(j_lower), 0.)
        v_upper, j_upper = self.prior.sample(upper[None], self.z)
        np.testing.assert_array_equal(v_upper, [3.])
        np.testing.assert_array_equal(j_upper, np.zeros((1, 2)))

    def test_transformed_pose_and_shape_jacobians(self):
        pose = np.array([1.3, .7, 12., .63])
        points = self.points @ rotation_y(pose[3]).T + pose[:3]
        # Include a stable out-of-grid point in both Jacobian checks.
        points = np.vstack((points, np.array([4., .2, .1]) @ rotation_y(pose[3]).T + pose[:3]))
        _, jp = pose_residuals(self.prior, points, pose, self.z)
        np.testing.assert_allclose(jp, finite_difference(
            lambda p: pose_residuals(self.prior, points, p, self.z)[0], pose), atol=2.e-8, rtol=2.e-7)
        _, js = shape_residuals(self.prior, points, pose, self.z)
        np.testing.assert_allclose(js, finite_difference(
            lambda z: shape_residuals(self.prior, points, pose, z)[0], self.z), atol=2.e-9)

    def test_objective_scaling(self):
        pose = np.array([0., .2, 0., 0.])
        points = self.points + pose[:3]
        for mode in ('pose', 'shape'):
            residual = self.prior.sample(self.points, self.z, mode)[0]
            absolute = np.abs(residual)
            expected = sum(.5*r*r if abs(r) <= .1 else .1*(abs(r)-.05) for r in residual)/(.03*len(points))
            expected += (.5*np.sum((self.z/self.prior.S)**2) if mode == 'shape' else (.2-.7)**2/6.)
            self.assertAlmostEqual(objective(self.prior, points, pose, self.z, .7, mode), expected, places=12)

    def test_reject_bad_inputs(self):
        for points in (np.empty((0, 3)), [[np.nan, 0., 0.]], [[np.inf, 0., 0.]], np.ones((3, 2))):
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    self.prior.sample(points, self.z)
                with self.assertRaises(ValueError):
                    fit_vehicle(self.prior, points, np.zeros(4))
        for pose in (np.zeros(3), np.array([0., 0., 0., np.inf])):
            with self.assertRaises(ValueError):
                pose_residuals(self.prior, self.points, pose, self.z)
        with self.assertRaises(ValueError):
            self.prior.sample(self.points, self.z, mode='bad')
        with self.assertRaises(ValueError):
            fit_vehicle(self.prior, self.points, np.zeros(4), max_outer_iterations=0)
        with self.assertRaises(ValueError):
            fit_vehicle(self.prior, self.points, np.zeros(4), cost_tolerance=np.nan)

    def test_stationary_singular_problem_and_iteration_limits(self):
        prior = polynomial_prior()
        prior.mean[:] = 1.
        prior.V[:] = 0.
        result = fit_vehicle(prior, [[0., -1., 0.]], np.zeros(4),
                             max_outer_iterations=3, max_solver_iterations=2)
        self.assertTrue(result['success'])
        self.assertEqual(len(result['history']), 2)
        self.assertEqual(result['termination'], 'outer_cost_tolerance')
        for entry in result['history']:
            self.assertEqual(entry['pose_solver']['status'], 'gradient_tolerance')
            self.assertEqual(entry['shape_solver']['status'], 'gradient_tolerance')
            self.assertLessEqual(entry['pose_iterations'], 2)


class RealPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = ShapePrior()

    def test_real_weight_dimensions_and_roundtrip(self):
        p = self.prior
        self.assertEqual(p.resolution, (60, 40, 60))
        self.assertEqual(p.S.shape, (5,))
        self.assertEqual(p.V.shape, (144000, 5))
        self.assertEqual(p.mean.shape, (144000,))
        np.testing.assert_array_equal(p.origin, [-3., -3., -3.])
        np.testing.assert_array_equal(p.unit, [.1, .1, .1])
        z = np.array([[1., -2., .3, -.4, .7], [0., 0., 0., 0., 0.]])
        decoded = p.decode(z)
        self.assertEqual(decoded.shape, (2, 60, 40, 60))
        self.assertEqual(p.decode(z[0]).shape, (1, 60, 40, 60))
        np.testing.assert_allclose(p.encode(decoded.reshape(2, -1)), z, atol=2.e-12)
        np.testing.assert_allclose(p.encode(decoded[0].ravel()), z[0], atol=2.e-12)
        with self.assertRaises(ValueError):
            p.decode(np.empty((0, 5)))
        with self.assertRaises(ValueError):
            p.encode(np.ones(2))
        with self.assertRaises(ValueError):
            ShapePrior(k=100)

    def test_real_weight_jacobians(self):
        rng = np.random.default_rng(71)
        local = rng.uniform([-1.8, -1.5, -1.8], [1.8, .3, 1.8], (20, 3))
        pose = np.array([2., 1.3, 15., .72])
        z = np.array([3., -2., .3, .5, -.2])
        points = local @ rotation_y(pose[3]).T + pose[:3]
        _, jp = pose_residuals(self.prior, points, pose, z)
        np.testing.assert_allclose(jp, finite_difference(
            lambda value: pose_residuals(self.prior, points, value, z)[0], pose), atol=2.e-8, rtol=2.e-6)
        _, js = shape_residuals(self.prior, points, pose, z)
        np.testing.assert_allclose(js, finite_difference(
            lambda value: shape_residuals(self.prior, points, pose, value)[0], z), atol=2.e-9)

    def test_true_prior_surface_fit_improves(self):
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            self.skipTest('optional skimage is unavailable')
        p = self.prior
        vertices, _, _, _ = marching_cubes(p.decode(np.zeros(p.k))[0], level=0., spacing=tuple(p.unit))
        rng = np.random.default_rng(9)
        local = vertices[rng.choice(len(vertices), 350, replace=False)] + p.origin
        truth = np.array([1., 1.2, 10., .4])
        points = local @ rotation_y(truth[3]).T + truth[:3]
        initial = truth + [.18, -.1, .13, .12]
        initial_copy, points_copy = initial.copy(), points.copy()
        before = objective(p, points, initial, np.zeros(p.k))
        result = fit_vehicle(p, points, initial, max_outer_iterations=4, max_solver_iterations=60)
        after = objective(p, points, result['pose'], result['z'])
        self.assertTrue(result['success'])
        self.assertLess(after, before * .2)
        self.assertLess(np.linalg.norm(result['pose'][:3]-truth[:3]), .06)
        self.assertLess(abs(result['pose'][3]-truth[3]), .04)
        self.assertGreaterEqual(len(result['history']), 1)
        self.assertLessEqual(len(result['history']), 4)
        for entry in result['history']:
            self.assertAlmostEqual(entry['total_cost'], entry['pose_cost']+entry['shape_cost'])
            for mode in ('pose', 'shape'):
                solver = entry[mode+'_solver']
                self.assertLessEqual(solver['final_cost'], solver['initial_cost'])
                self.assertLessEqual(solver['iterations'], 60)
        self.assertAlmostEqual(result['final_costs']['shape_cost'], after)
        np.testing.assert_array_equal(initial, initial_copy)
        np.testing.assert_array_equal(points, points_copy)


if __name__ == '__main__':
    unittest.main()
