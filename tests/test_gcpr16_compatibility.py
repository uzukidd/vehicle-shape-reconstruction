"""Integration checks for the replacement API and its coordinate adapter."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vehicle_reconstruction_gcpr16 import point_cloud_scene, vehicle_object, vehicle_reconstructor
from vehicle_reconstruction_gcpr16.optimization import objective, rotation_y
from vehicle_reconstruction_gcpr16.scene import (boxes_to_poses, camera_to_scene,
                                                poses_to_boxes, scene_to_camera)
from vehicle_reconstruction_gcpr16.structures import MeshBatch, PointcloudBatch, as_numpy


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reconstructor = vehicle_reconstructor()

    def test_native_grid_and_compatibility_methods(self):
        model = self.reconstructor
        self.assertEqual(model.TSDF.resolution, (60, 40, 60))
        z = np.array([[1., 2., 3., 4., 5.], [-2., 1., 0., 3., -1.]])
        grid = model.decode(z)
        np.testing.assert_allclose(model.encode(grid), z, atol=1.e-10)
        np.testing.assert_allclose(model.reconsturct(grid), grid, atol=1.e-10)
        with self.assertRaisesRegex(ValueError, 'GCPR16 weights require'):
            vehicle_reconstructor(resolution=[20, 30, 40])
        for malformed in (np.zeros((2, 10)), np.zeros((5, 2)), np.zeros((1, 1, 5))):
            with self.assertRaises(ValueError):
                model.decode(malformed)
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / 'prior.npz'
            model.save_parameters(snapshot)
            loaded = vehicle_reconstructor().load_parameters(snapshot)
            np.testing.assert_array_equal(loaded.decode(z), grid)

    def test_pose_coordinate_roundtrip_and_front_direction(self):
        boxes = np.array([[3., 8., 1., 4.6, 2., 1.7, 0.],
                          [-2., 5., -.2, 4., 1.8, 1.5, 1.2]])
        poses = boxes_to_poses(boxes)
        np.testing.assert_allclose(poses_to_boxes(poses, boxes), boxes, atol=1.e-14)
        for box, pose in zip(boxes, poses):
            front_camera = np.array([[0., 0., -1.]]) @ rotation_y(pose[3]).T
            front_lidar = camera_to_scene(front_camera, 'lidar')[0]
            np.testing.assert_allclose(front_lidar, [np.cos(box[6]), np.sin(box[6]), 0.], atol=1.e-14)
        points = np.array([[1., 2., 3.], [5., -2., 7.]])
        np.testing.assert_array_equal(camera_to_scene(scene_to_camera(points, 'lidar'), 'lidar'), points)

    def test_rotated_roi_and_ownership_without_padding(self):
        box = np.array([[10., 0., 1., 4., 2., 2., np.pi/2]])
        points = np.array([[10., 1.5, 1., 0.], [10.8, 0., 1., 0.],
                           [11.2, 0., 1., 0.], [10., 0., 2.2, 0.]])
        scene = point_cloud_scene(points, np.repeat(box, 2, axis=0))
        vehicles = scene.get_vehicles()
        np.testing.assert_array_equal(vehicles.pts.num_points_per_cloud(), [2, 0])
        np.testing.assert_array_equal(scene.pts_assign, [0, 0, -1, -1])
        training = vehicles.get_training_point_clouds()
        self.assertEqual(training.num_points_per_cloud()[1], 0)
        local = vehicles.get_local_point_clouds().points_list()[0]
        pose = vehicles.poses[0]
        recovered = camera_to_scene(local @ rotation_y(pose[3]).T + pose[:3], 'lidar')
        np.testing.assert_allclose(recovered, points[:2, :3], atol=1.e-12)
        np.testing.assert_allclose(training.points_list()[0], local + 3.)

    def test_scene_fit_updates_pose_and_mesh_in_same_frame(self):
        model = self.reconstructor
        native = model.decode(np.zeros((1, 5)), to_meshes=True)
        local = as_numpy(native.verts_list()[0])[::7]
        true_box = np.array([[8., 2., .85, 5., 2.2, 1.8, .4]])
        pose = boxes_to_poses(true_box)[0]
        observed = camera_to_scene(local @ rotation_y(pose[3]).T + pose[:3], 'lidar')
        initial_box = true_box.copy()
        initial_box[0, :3] += [.12, -.1, .07]
        initial_box[0, 6] += .05
        vehicles = vehicle_object(PointcloudBatch([observed]), initial_box)
        scene = point_cloud_scene(observed, initial_box)
        scene.vehicles = vehicles  # Use identical observations to isolate fitting from ROI clipping.
        initial_pose = vehicles.poses[0]
        camera_points = scene_to_camera(observed, 'lidar')
        before = objective(model.prior, camera_points, initial_pose, np.zeros(5))
        results = scene.pose_estimate(model, iter=5, max_solver_iterations=60, point_threshold=10, max_distance=None)
        after = objective(model.prior, camera_points, vehicles.poses[0], as_numpy(vehicles.vehicle_latent)[0])
        self.assertTrue(results[0]['success'])
        self.assertGreaterEqual(results[0]['elapsed_seconds'], 0.)
        self.assertEqual(scene.fit_timing['vehicle_count'], 1)
        self.assertEqual(scene.fit_timing['solver_seconds'], results[0]['elapsed_seconds'])
        self.assertEqual(scene.fit_timing['mean_vehicle_seconds'], results[0]['elapsed_seconds'])
        self.assertGreaterEqual(scene.fit_timing['total_seconds'], scene.fit_timing['solver_seconds'])
        self.assertLess(after, before * .05)
        self.assertLess(np.linalg.norm(vehicles.translation[0, :3] - true_box[0, :3]), .03)
        self.assertLess(abs(vehicles.translation[0, 3] - true_box[0, 6]), .01)
        meshes, roof = vehicles.reconstruct_at_scene(model, show_rooftop=True)
        self.assertEqual(len(meshes), 1)
        self.assertEqual(len(roof), len(meshes.verts_packed()))
        self.assertTrue(roof.any())
        minimum = meshes.verts_packed().min(0)
        maximum = meshes.verts_packed().max(0)
        self.assertLess(np.linalg.norm((minimum + maximum)/2 - true_box[0, :3]), .25)
        local_updated = vehicles.get_local_point_clouds().points_list()[0]
        np.testing.assert_allclose(local_updated, local, atol=.03)

    def test_empty_and_skipped_scenes(self):
        scene = point_cloud_scene(np.empty((0, 4)), np.empty((0, 7)))
        self.assertEqual(scene.pose_estimate(self.reconstructor), [])
        self.assertEqual(scene.fit_timing['vehicle_count'], 0)
        self.assertEqual(scene.fit_timing['solver_seconds'], 0.)
        self.assertEqual(scene.fit_timing['mean_vehicle_seconds'], 0.)
        self.assertEqual(scene.get_vehicles().get_tsdf_grid(self.reconstructor).shape, (0, 60, 40, 60))
        self.assertEqual(self.reconstructor.encode_aux(np.empty((0, 144000))).shape, (0, 5))
        meshes, roof = scene.get_vehicles().reconstruct_at_scene(self.reconstructor, show_rooftop=True)
        self.assertEqual(len(meshes), 0)
        self.assertEqual(roof.size, 0)
        scene = point_cloud_scene(np.array([[40., 0., 0.]]), np.array([[40., 0., 0., 4., 2., 2., 0.]]))
        results = scene.pose_estimate(self.reconstructor, point_threshold=0)
        self.assertEqual(results[0]['skipped'], 'forward_distance')
        meshes, _ = scene.get_vehicles().reconstruct_at_scene(self.reconstructor, threshold=0)
        self.assertEqual(len(meshes), 0)

    def test_camera_adapter(self):
        box = np.array([[0., 1., 10., 4.5, 2., 1.7, .3]])
        pose = boxes_to_poses(box, 'camera')[0]
        local = np.array([[.4, -.5, .7], [-.2, -1.2, -1.]])
        points = local @ rotation_y(pose[3]).T + pose[:3]
        scene = point_cloud_scene(points, box, coordinate_frame='camera')
        vehicles = scene.get_vehicles()
        np.testing.assert_allclose(vehicles.get_local_point_clouds().points_list()[0], local, atol=1.e-12)

    def test_mesh_packed_faces_and_point_lengths(self):
        mesh = MeshBatch([np.zeros((3, 3)), np.ones((4, 3))],
                         [np.array([[0, 1, 2]]), np.array([[1, 2, 3]])])
        np.testing.assert_array_equal(mesh.faces_packed(), [[0, 1, 2], [4, 5, 6]])
        np.testing.assert_array_equal(mesh.verts_padded_to_packed_idx(), [0, 1, 2, 4, 5, 6, 7])
        points = PointcloudBatch([np.zeros((2, 3)), np.zeros((0, 3))])
        self.assertEqual(points[1].points_padded().shape, (1, 0, 3))
        self.assertEqual(points[np.array([False, False])].points_padded().shape, (0, 0, 3))


if __name__ == '__main__':
    unittest.main()
