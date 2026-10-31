"""Scene adapters: LiDAR box centers to the original camera/bottom-center pose."""
from __future__ import annotations

from time import perf_counter

import numpy as np

from .optimization import fit_vehicle, rotation_y
from .structures import (PointcloudBatch, as_numpy, is_tensor, like, make_meshes,
                         smooth_vertices)

# Right-handed LiDAR (front,left,up) -> camera (right,down,forward).
_LIDAR_TO_CAMERA = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def scene_to_camera(points, coordinate_frame):
    points = np.asarray(points, dtype=np.float64)
    return points @ _LIDAR_TO_CAMERA.T if coordinate_frame == 'lidar' else points.copy()


def camera_to_scene(points, coordinate_frame):
    points = np.asarray(points, dtype=np.float64)
    return points @ _LIDAR_TO_CAMERA if coordinate_frame == 'lidar' else points.copy()


def boxes_to_poses(boxes, coordinate_frame='lidar'):
    """LiDAR boxes use center xyz,l,w,h,yaw; camera boxes use bottom xyz,l,w,h,ry.

    The pretrained canonical vehicle faces -z. Therefore a LiDAR front +x
    with yaw=0 has native rotation_y=pi. Camera angles are already native;
    raw KITTI label rotation_y must first subtract pi/2, as in the C++ reader.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if coordinate_frame == 'lidar':
        bottom = boxes[:, :3].copy()
        bottom[:, 2] -= boxes[:, 5] / 2
        return np.column_stack((scene_to_camera(bottom, coordinate_frame),
                                _wrap(np.pi - boxes[:, 6])))
    return np.column_stack((boxes[:, :3], boxes[:, 6]))


def poses_to_boxes(poses, boxes, coordinate_frame='lidar'):
    result = np.asarray(boxes, dtype=np.float64).copy()
    if coordinate_frame == 'lidar':
        result[:, :3] = camera_to_scene(poses[:, :3], coordinate_frame)
        result[:, 2] += result[:, 5] / 2
        result[:, 6] = _wrap(np.pi - poses[:, 3])
    else:
        result[:, :3] = poses[:, :3]
        result[:, 6] = _wrap(poses[:, 3])
    return result


class vehicle_object:
    def __init__(self, pts, bbox, training_feature=None, k=5, *, coordinate_frame='lidar'):
        if coordinate_frame not in ('lidar', 'camera'):
            raise ValueError("coordinate_frame must be 'lidar' or 'camera'.")
        self.reference = bbox if is_tensor(bbox) else getattr(pts, 'reference', None)
        if self.reference is None and is_tensor(pts):
            self.reference = pts
        self.pts = PointcloudBatch(pts, self.reference)
        boxes = as_numpy(bbox).astype(np.float64)
        if boxes.shape != (len(self.pts), 7) or not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
            raise ValueError('bbox must be finite [B,7] with positive l,w,h, matching point clouds.')
        if any(not np.isfinite(as_numpy(p)).all() for p in self.pts.points_list()):
            raise ValueError('Point clouds must be finite.')
        self.coordinate_frame = coordinate_frame
        self.bbox = like(boxes.copy(), self.reference)
        self.batch_size = len(self.pts)
        self.training_feature = training_feature
        self.vehicle_latent = like(np.zeros((len(self.pts), k)), self.reference)
        self.translation = like(boxes[:, [0, 1, 2, 6]].copy(), self.reference)
        self.fit_results = [None] * len(self.pts)
        self._fit_mask = None
        self.reconstructed_indices = np.empty(0, dtype=int)

    def __len__(self):
        return len(self.pts)

    @property
    def poses(self):
        boxes = as_numpy(self.bbox).copy()
        boxes[:, [0, 1, 2, 6]] = as_numpy(self.translation)
        return boxes_to_poses(boxes, self.coordinate_frame)

    def _set_poses(self, poses):
        boxes = poses_to_boxes(poses, as_numpy(self.bbox), self.coordinate_frame)
        self.translation = like(boxes[:, [0, 1, 2, 6]], self.reference)

    def get_local_point_clouds(self, pts=None):
        clouds = self.pts if pts is None else PointcloudBatch(pts, self.reference)
        if len(clouds) != len(self):
            raise ValueError('Point cloud batch must match the number of vehicles.')
        result = []
        for points, pose in zip(clouds.points_list(), self.poses):
            camera = scene_to_camera(as_numpy(points), self.coordinate_frame)
            result.append((camera - pose[:3]) @ rotation_y(pose[3]))
        return PointcloudBatch(result, self.reference)

    def get_training_point_clouds(self):
        # Preserve all actual points. Padded zeros and ROI duplicates never
        # become observations in either optimization subproblem.
        local = self.get_local_point_clouds()
        return PointcloudBatch([as_numpy(p) + 3. for p in local.points_list()], self.reference)

    def get_tsdf_grid(self, vehi_reconstructor):
        return vehi_reconstructor.decode(self.vehicle_latent)

    def reconstruct_at_scene(self, vehi_reconstructor, threshold=10, smoothing=False,
                             show_rooftop=False):
        if not isinstance(threshold, (int, np.integer)) or threshold < 0:
            raise ValueError('threshold must be a nonnegative integer.')
        counts = as_numpy(self.pts.num_points_per_cloud())
        selected = counts > threshold
        if self._fit_mask is not None:
            selected &= self._fit_mask
        self.reconstructed_indices = np.flatnonzero(selected)
        verts, faces, masks = [], [], []
        poses = self.poses
        z = as_numpy(self.vehicle_latent)
        for index in self.reconstructed_indices:
            native = vehi_reconstructor.to_meshes(vehi_reconstructor.prior.decode(z[index]))
            vertices = as_numpy(native.verts_list()[0])
            triangles = as_numpy(native.faces_list()[0])
            if smoothing:
                vertices = smooth_vertices(vertices, triangles)
            pose = poses[index]
            camera = vertices @ rotation_y(pose[3]).T + pose[:3]
            world = camera_to_scene(camera, self.coordinate_frame)
            verts.append(world)
            faces.append(triangles)
            if show_rooftop:
                height = world[:, 2] if self.coordinate_frame == 'lidar' else -world[:, 1]
                masks.append(height.max() - height < .2 if len(height) else np.empty(0, dtype=bool))
        meshes = make_meshes(verts, faces, self.reference)
        roof = None
        if show_rooftop:
            roof = like(np.concatenate(masks) if masks else np.empty(0, dtype=bool),
                        self.reference, boolean=True)
        return meshes, roof


class point_cloud_scene:
    """Compatible scene API without the old custom CUDA extensions.

    ROI extraction is an input adapter, not the original stereo segmentation.
    All selected points are kept. Original >=200-points and forward-distance
    <=30 m filters are defaults; explicitly relax them for sparse LiDAR data.
    """

    def __init__(self, pts, gt_boxes, *, coordinate_frame='lidar', vehi_reconstructor=None):
        points = as_numpy(pts).astype(np.float64)
        boxes = as_numpy(gt_boxes).astype(np.float64)
        if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points).all():
            raise ValueError('pts must be finite [N,3] or [N,4].')
        if boxes.ndim != 2 or boxes.shape[1] != 7 or not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
            raise ValueError('gt_boxes must be finite [B,7] with positive l,w,h.')
        if coordinate_frame not in ('lidar', 'camera'):
            raise ValueError("coordinate_frame must be 'lidar' or 'camera'.")
        self.pts = like(points.copy(), pts)
        self.gt_boxes = like(boxes.copy(), gt_boxes)
        self.coordinate_frame = coordinate_frame
        self.vehi_reconstructor = vehi_reconstructor
        self.vehicles = None
        self.segmented_pts = None
        self.segmented_gt_boxes = None
        self.fit_results = []
        self.fit_timing = None

    def vehicle_seg(self):
        points = as_numpy(self.pts)[:, :3]
        boxes = as_numpy(self.gt_boxes)
        assignment = np.full(len(points), -1, dtype=int)
        clouds = []
        poses = boxes_to_poses(boxes, self.coordinate_frame)
        camera = scene_to_camera(points, self.coordinate_frame)
        for index, (box, pose) in enumerate(zip(boxes, poses)):
            if self.coordinate_frame == 'lidar':
                offset = points - box[:3]
                c, s = np.cos(box[6]), np.sin(box[6])
                local = np.column_stack((offset[:, 0] * c + offset[:, 1] * s,
                                         -offset[:, 0] * s + offset[:, 1] * c, offset[:, 2]))
                mask = np.all(np.abs(local) <= box[3:6] / 2 + 1.e-9, axis=1)
            else:
                local = (camera - pose[:3]) @ rotation_y(pose[3])
                mask = ((np.abs(local[:, 0]) <= box[4] / 2 + 1.e-9) &
                        (np.abs(local[:, 2]) <= box[3] / 2 + 1.e-9) &
                        (local[:, 1] <= 1.e-9) & (local[:, 1] >= -box[5] - 1.e-9))
            mask &= assignment < 0
            assignment[mask] = index
            clouds.append(points[mask])
        reference = self.gt_boxes if is_tensor(self.gt_boxes) else self.pts
        self.pts_assign = like(assignment, reference, integer=True)
        self.segmented_pts = PointcloudBatch(clouds, reference)
        self.segmented_gt_boxes = self.gt_boxes
        k = self.vehi_reconstructor.k if self.vehi_reconstructor is not None else 5
        self.vehicles = vehicle_object(self.segmented_pts, self.gt_boxes, k=k,
                                       coordinate_frame=self.coordinate_frame)
        return self.vehicles

    def get_vehicles(self):
        if self.vehicles is None:
            self.vehicle_seg()
        return self.vehicles

    def pose_estimate(self, vehi_reconstructor=None, iter=10, point_threshold=199, *,
                      max_solver_iterations=100, cost_tolerance=.1, max_distance=30.):
        reconstructor = vehi_reconstructor or self.vehi_reconstructor
        if reconstructor is None:
            raise ValueError('Pass vehi_reconstructor to pose_estimate() or the scene constructor.')
        if isinstance(iter, (bool, np.bool_)) or not isinstance(iter, (int, np.integer)) or iter < 1:
            raise ValueError('iter must be a positive outer-iteration limit.')
        if not isinstance(point_threshold, (int, np.integer)) or point_threshold < 0:
            raise ValueError('point_threshold must be a nonnegative integer.')
        if max_distance is not None and (not np.isfinite(max_distance) or max_distance <= 0):
            raise ValueError('max_distance must be positive or None.')
        vehicles = self.get_vehicles()
        latent = as_numpy(vehicles.vehicle_latent).copy()
        if latent.shape != (len(vehicles), reconstructor.k):
            if not np.any(latent):
                latent = np.zeros((len(vehicles), reconstructor.k))
            else:
                raise ValueError('Existing coefficients do not match the selected prior dimension.')
        poses = vehicles.poses
        fitted = np.zeros(len(vehicles), dtype=bool)
        results = []
        fit_started = perf_counter()
        for index, points in enumerate(vehicles.pts.points_list()):
            points = as_numpy(points)
            if len(points) <= point_threshold:
                results.append({'success': False, 'skipped': 'insufficient_points', 'point_count': len(points)})
                continue
            if max_distance is not None and poses[index, 2] > max_distance:
                results.append({'success': False, 'skipped': 'forward_distance', 'point_count': len(points)})
                continue
            result = fit_vehicle(reconstructor.prior, scene_to_camera(points, self.coordinate_frame),
                                 poses[index], latent[index], max_outer_iterations=iter,
                                 max_solver_iterations=max_solver_iterations, cost_tolerance=cost_tolerance)
            result['point_count'] = len(points)
            poses[index], latent[index] = result['pose'], result['z']
            fitted[index] = result['success']
            results.append(result)
        vehicles._set_poses(poses)
        vehicles.vehicle_latent = like(latent, vehicles.reference)
        vehicles._fit_mask = fitted
        vehicles.fit_results = results
        self.fit_results = results
        self.vehi_reconstructor = reconstructor
        fit_wall_seconds = perf_counter() - fit_started
        solver_results = [result for result in results if 'skipped' not in result]
        solver_seconds = sum(result['elapsed_seconds'] for result in solver_results)
        # Excludes model loading, ROI extraction, mesh extraction and rendering.
        # 不含权重加载、框内分割、网格提取和可视化。
        self.fit_timing = {
            'total_seconds': fit_wall_seconds,
            'solver_seconds': solver_seconds,
            'mean_vehicle_seconds': solver_seconds / len(solver_results) if solver_results else 0.,
            'vehicle_count': len(solver_results),
        }
        return results


GCPR16Scene = point_cloud_scene
GCPR16Vehicles = vehicle_object
