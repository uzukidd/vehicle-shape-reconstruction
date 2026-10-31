"""Run the pretrained GCPR16 backend on .npy inputs, or execute its tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vehicle_reconstruction_gcpr16 import point_cloud_scene, vehicle_reconstructor
from vehicle_reconstruction_gcpr16.structures import as_numpy


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot encode {type(value).__name__}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true', help='Run core and compatibility regression tests.')
    parser.add_argument('--points', type=Path, default=ROOT / 'data/00043_pts.npy')
    parser.add_argument('--boxes', type=Path, default=ROOT / 'data/00043_gt.npy')
    parser.add_argument('--matrices', type=Path, default=ROOT / 'data/matrices')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/gcpr16')
    parser.add_argument('--outer-iterations', type=int, default=10)
    parser.add_argument('--solver-iterations', type=int, default=100)
    parser.add_argument('--min-points', type=int, default=200, help='Inclusive minimum; original default is 200.')
    parser.add_argument('--max-distance', type=float, default=30., help='Forward camera distance; <=0 disables filter.')
    parser.add_argument('--coordinate-frame', choices=('lidar', 'camera'), default='lidar')
    args = parser.parse_args(argv)
    if args.test:
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_gcpr16*.py')
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.min_points < 1:
        parser.error('--min-points must be at least 1')
    if args.outer_iterations < 1 or args.solver_iterations < 1:
        parser.error('iteration limits must be positive')
    model = vehicle_reconstructor(matrices_dir=args.matrices)
    scene = point_cloud_scene(np.load(args.points, allow_pickle=False),
                              np.load(args.boxes, allow_pickle=False),
                              coordinate_frame=args.coordinate_frame)
    results = scene.pose_estimate(model, iter=args.outer_iterations,
                                  point_threshold=args.min_points - 1,
                                  max_solver_iterations=args.solver_iterations,
                                  max_distance=None if args.max_distance <= 0 else args.max_distance)
    vehicles = scene.get_vehicles()
    meshes, rooftop = vehicles.reconstruct_at_scene(model, threshold=args.min_points - 1,
                                                   smoothing=False, show_rooftop=True)
    args.output.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output / 'vehicles.obj'
    with mesh_path.open('w', encoding='utf-8', newline='\n') as stream:
        for vertex in as_numpy(meshes.verts_packed()):
            stream.write('v ' + ' '.join(f'{value:.9g}' for value in vertex) + '\n')
        for face in as_numpy(meshes.faces_packed()) + 1:
            stream.write('f ' + ' '.join(str(int(value)) for value in face) + '\n')
    np.savez(args.output / 'reconstruction.npz', z=as_numpy(vehicles.vehicle_latent),
             translation=as_numpy(vehicles.translation), poses=vehicles.poses,
             initial_boxes=as_numpy(scene.gt_boxes), fitted=vehicles._fit_mask,
             reconstructed_indices=vehicles.reconstructed_indices,
             rooftop_mask=as_numpy(rooftop),
             vertices=as_numpy(meshes.verts_packed()), faces=as_numpy(meshes.faces_packed()))
    report = {'backend': 'GCPR16 NumPy LM/IRLS', 'coordinate_frame': args.coordinate_frame,
              'matrices': str(args.matrices.resolve()), 'point_count': len(scene.pts),
              'vehicle_count': len(vehicles), 'fitted_count': int(vehicles._fit_mask.sum()),
              'timing': scene.fit_timing, 'results': results}
    with (args.output / 'diagnostics.json').open('w', encoding='utf-8') as stream:
        json.dump(report, stream, default=_json_value, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'Fitted {report["fitted_count"]}/{report["vehicle_count"]} vehicles; '
          f'{len(meshes.verts_packed())} vertices, {len(meshes.faces_packed())} triangles.')
    timing = scene.fit_timing
    print(f'Fitting total / 拟合总耗时: {timing["total_seconds"]:.6f} s; '
          f'solver total / 求解器累计耗时: {timing["solver_seconds"]:.6f} s; '
          f'mean per vehicle / 平均每车: {timing["mean_vehicle_seconds"] * 1000:.3f} ms '
          f'({timing["vehicle_count"]} vehicles / 辆).')
    print(f'Output: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
