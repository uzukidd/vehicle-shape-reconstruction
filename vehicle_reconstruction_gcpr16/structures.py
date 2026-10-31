"""Small array adapters used when PyTorch3D is not installed.

Optional tensors stay on the caller's device; optimization uses NumPy float64.
"""
from __future__ import annotations

import numpy as np


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def is_tensor(value):
    return hasattr(value, "detach") and hasattr(value, "new_tensor")


def like(value, reference=None, *, integer=False, boolean=False):
    array = np.asarray(value)
    if is_tensor(reference):
        import torch
        dtype = torch.bool if boolean else torch.long if integer else (
            reference.dtype if reference.is_floating_point() else torch.float64)
        return torch.as_tensor(array.copy(), device=reference.device, dtype=dtype)
    return array.astype(bool if boolean else np.int64 if integer else array.dtype, copy=False)


def _indices(index, size):
    result = np.arange(size)[as_numpy(index) if is_tensor(index) else index]
    return np.atleast_1d(result).astype(int)


class PointcloudBatch:
    """Subset of the Pointclouds interface, including empty clouds and batches."""

    def __init__(self, points, reference=None):
        if hasattr(points, "points_list"):
            points = points.points_list()
        elif isinstance(points, np.ndarray) or is_tensor(points):
            if points.ndim == 2:
                points = [points]
            elif points.ndim != 3:
                raise ValueError("Points must have shape [N,3] or [B,N,3].")
        points = list(points)
        self.reference = reference
        if self.reference is None and points:
            self.reference = points[0]
        self._points = [like(as_numpy(p).reshape(-1, 3), self.reference) for p in points]

    @property
    def device(self):
        return self.reference.device if is_tensor(self.reference) else "cpu"

    def __len__(self):
        return len(self._points)

    def __getitem__(self, index):
        return PointcloudBatch([self._points[i] for i in _indices(index, len(self))], self.reference)

    def points_list(self):
        return list(self._points)

    def num_points_per_cloud(self):
        return like([len(p) for p in self._points], self.reference, integer=True)

    def points_packed(self):
        array = np.concatenate([as_numpy(p) for p in self._points]) if self._points else np.empty((0, 3))
        return like(array, self.reference)

    def points_padded(self):
        maximum = max((len(p) for p in self._points), default=0)
        array = np.zeros((len(self), maximum, 3), dtype=np.float64)
        for i, points in enumerate(self._points):
            array[i, :len(points)] = as_numpy(points)
        return like(array, self.reference)

    def update_padded(self, points):
        if len(points) != len(self):
            raise ValueError("The updated point batch must preserve batch size.")
        return PointcloudBatch([points[i, :len(p)] for i, p in enumerate(self._points)], self.reference)

    def to(self, device):
        import torch
        return PointcloudBatch([torch.as_tensor(as_numpy(p), device=device) for p in self._points],
                               torch.empty(0, device=device))


class MeshBatch:
    """Fallback for the packed/list/padded Meshes methods used by this project."""

    def __init__(self, verts, faces, reference=None):
        self.reference = reference
        self._verts = [like(np.asarray(v).reshape(-1, 3), reference) for v in verts]
        self._faces = [like(np.asarray(f).reshape(-1, 3), reference, integer=True) for f in faces]

    @property
    def device(self):
        return self.reference.device if is_tensor(self.reference) else "cpu"

    def __len__(self):
        return len(self._verts)

    def __getitem__(self, index):
        ids = _indices(index, len(self))
        return MeshBatch([as_numpy(self._verts[i]) for i in ids],
                         [as_numpy(self._faces[i]) for i in ids], self.reference)

    def verts_list(self):
        return list(self._verts)

    def faces_list(self):
        return list(self._faces)

    def verts_packed(self):
        data = np.concatenate([as_numpy(v) for v in self._verts]) if self._verts else np.empty((0, 3))
        return like(data, self.reference)

    def faces_packed(self):
        result, offset = [], 0
        for verts, faces in zip(self._verts, self._faces):
            result.append(as_numpy(faces) + offset)
            offset += len(verts)
        return like(np.concatenate(result) if result else np.empty((0, 3), dtype=int),
                    self.reference, integer=True)

    def verts_padded(self):
        maximum = max((len(v) for v in self._verts), default=0)
        data = np.zeros((len(self), maximum, 3))
        for i, verts in enumerate(self._verts):
            data[i, :len(verts)] = as_numpy(verts)
        return like(data, self.reference)

    def verts_padded_to_packed_idx(self):
        maximum = max((len(v) for v in self._verts), default=0)
        ids = np.concatenate([i * maximum + np.arange(len(v)) for i, v in enumerate(self._verts)]) if self._verts else np.empty(0, dtype=int)
        return like(ids, self.reference, integer=True)

    def num_verts_per_mesh(self):
        return like([len(v) for v in self._verts], self.reference, integer=True)

    def update_padded(self, verts):
        return MeshBatch([as_numpy(verts[i, :len(v)]) for i, v in enumerate(self._verts)],
                         [as_numpy(f) for f in self._faces], self.reference)

    def offset_verts_(self, offset):
        packed = as_numpy(self.verts_packed()) + as_numpy(offset)
        cursor = 0
        for i, verts in enumerate(self._verts):
            self._verts[i] = like(packed[cursor:cursor + len(verts)], self.reference)
            cursor += len(verts)
        return self

    def offset_verts(self, offset):
        return MeshBatch([as_numpy(v).copy() for v in self._verts],
                         [as_numpy(f).copy() for f in self._faces], self.reference).offset_verts_(offset)


def make_meshes(verts, faces, reference=None):
    if is_tensor(reference):
        try:
            from pytorch3d.structures import Meshes
        except ImportError:
            pass
        else:
            if verts:
                return Meshes(verts=[like(v, reference) for v in verts],
                              faces=[like(f, reference, integer=True) for f in faces])
    return MeshBatch(verts, faces, reference)


def smooth_vertices(verts, faces, iterations=10):
    """Optional Taubin smoothing; zero-surface extraction itself is unsmoothed."""
    verts = np.array(verts, dtype=float, copy=True)
    if not len(faces):
        return verts
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]])
    degrees = np.bincount(edges[:, 0], minlength=len(verts))
    active = degrees > 0
    for _ in range(iterations):
        for factor in (0.5, -0.53):
            neighbors = np.zeros_like(verts)
            np.add.at(neighbors, edges[:, 0], verts[edges[:, 1]])
            delta = np.zeros_like(verts)
            delta[active] = neighbors[active] / degrees[active, None] - verts[active]
            verts += factor * delta
    return verts
