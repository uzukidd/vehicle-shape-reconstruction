"""Optional tests executed when a PyTorch installation is available."""
import importlib.util
import unittest

import numpy as np

from vehicle_reconstruction_gcpr16 import point_cloud_scene, vehicle_object, vehicle_reconstructor
from vehicle_reconstruction_gcpr16.structures import PointcloudBatch, as_numpy


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is optional and not installed')
class TensorCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.model = vehicle_reconstructor()

    def test_differentiable_decode_encode_and_empty_batch(self):
        torch = self.torch
        z = torch.zeros((2, 5), dtype=torch.float64, requires_grad=True)
        grid = self.model.decode(z)
        self.assertEqual(grid.device, z.device)
        self.assertEqual(grid.dtype, z.dtype)
        encoded = self.model.encode_aux(grid)
        torch.testing.assert_close(encoded, z, atol=1.e-10, rtol=1.e-10)
        encoded.sum().backward()
        torch.testing.assert_close(z.grad, torch.ones_like(z), atol=1.e-10, rtol=1.e-10)
        empty = torch.empty((0, 5), dtype=torch.float64)
        self.assertEqual(self.model.decode(empty).shape, (0, 60, 40, 60))
        with self.assertRaises(ValueError):
            self.model.decode(torch.zeros((2, 10)))

    def test_integer_boxes_preserve_fractional_state(self):
        torch = self.torch
        boxes = torch.tensor([[10, 0, 1, 4, 2, 2, 0]])
        clouds = PointcloudBatch([torch.tensor([[10., 0., 1.], [10.1, .1, 1.2]])])
        vehicles = vehicle_object(clouds, boxes)
        self.assertTrue(vehicles.vehicle_latent.is_floating_point())
        self.assertTrue(vehicles.translation.is_floating_point())
        pose = vehicles.poses
        pose[0, 0] += .125
        pose[0, 3] += .075
        vehicles._set_poses(pose)
        self.assertAlmostEqual(float(vehicles.translation[0, 1]), -.125)
        self.assertAlmostEqual(float(vehicles.translation[0, 3]), -.075)
        scene = point_cloud_scene(torch.empty((0, 4)), torch.empty((0, 7)))
        self.assertEqual(scene.get_vehicles().get_tsdf_grid(self.model).shape, (0, 60, 40, 60))

    def test_tensor_scene_fitting_and_mixed_input_reference(self):
        from vehicle_reconstruction_gcpr16.optimization import rotation_y
        from vehicle_reconstruction_gcpr16.scene import boxes_to_poses, camera_to_scene
        torch = self.torch
        local = as_numpy(self.model.decode(np.zeros((1, 5)), to_meshes=True).verts_list()[0])[::12]
        true_box = np.array([[7., 1., .8, 5., 2.2, 1.8, .25]])
        pose = boxes_to_poses(true_box)[0]
        observed = camera_to_scene(local @ rotation_y(pose[3]).T + pose[:3], 'lidar')
        initial = true_box.copy()
        initial[0, :3] += [.08, -.06, .04]
        initial[0, 6] += .03
        points = torch.tensor(observed, dtype=torch.float32)
        boxes = torch.tensor(initial, dtype=torch.float32)
        scene = point_cloud_scene(points, boxes)
        results = scene.pose_estimate(self.model, iter=4, point_threshold=10, max_distance=None)
        self.assertTrue(results[0]['success'])
        vehicles = scene.get_vehicles()
        self.assertEqual(vehicles.vehicle_latent.dtype, torch.float32)
        self.assertLess(np.linalg.norm(as_numpy(vehicles.translation)[0, :3] - true_box[0, :3]), .03)
        meshes, roof = vehicles.reconstruct_at_scene(self.model, show_rooftop=True)
        self.assertEqual(meshes.verts_packed().dtype, torch.float32)
        self.assertEqual(roof.dtype, torch.bool)
        mixed = point_cloud_scene(points, initial).get_vehicles()
        self.assertIsInstance(mixed.vehicle_latent, torch.Tensor)

    def test_tensor_mesh_outputs_and_roof_mask(self):
        torch = self.torch
        boxes = torch.tensor([[5., 0., 1., 5., 2., 2., 0.]], dtype=torch.float64)
        vehicles = vehicle_object(PointcloudBatch([torch.zeros((11, 3), dtype=torch.float64)]), boxes)
        mesh, roof = vehicles.reconstruct_at_scene(self.model, threshold=10, show_rooftop=True)
        self.assertIsInstance(mesh.verts_packed(), torch.Tensor)
        self.assertEqual(mesh.faces_packed().dtype, torch.int64)
        self.assertEqual(roof.dtype, torch.bool)
        self.assertEqual(len(roof), len(mesh.verts_packed()))
        self.assertTrue(np.isfinite(as_numpy(mesh.verts_packed())).all())


if __name__ == '__main__':
    unittest.main()
