"""Pretrained GCPR16 shape decoder with the existing reconstructor method names."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .prior import ShapePrior
from .structures import as_numpy, is_tensor, like, make_meshes


class vehicle_reconstructor:
    """Load the published Eigen matrices; coefficients are the original z.

    The grid is fixed by the weights. Device/dtype control tensor outputs only;
    scene fitting uses CPU float64. No mesh-to-SDF or PCA training is required.
    """

    def __init__(self, resolution=None, bbox=None, sampling_count=4096,
                 downsampling_count=2048, unit=None, *, matrices_dir=None, k=5,
                 device=None, dtype=None):
        self.prior = ShapePrior(matrices_dir=matrices_dir, k=k)
        self._reference = None
        for value in (unit, bbox, resolution):
            if is_tensor(value):
                self._reference = value
                break
        if device is not None or dtype is not None:
            import torch
            self._reference = torch.empty(0, device=device, dtype=dtype or torch.float64)
        if self._reference is not None:
            import torch
            if not self._reference.is_floating_point():
                self._reference = self._reference.to(dtype=torch.float64)
        self._validate_grid(resolution, bbox, unit)
        self._sync_metadata()

    def _validate_grid(self, resolution, bbox, unit):
        for name, supplied, expected in (
            ("resolution", resolution, self.prior.resolution),
            ("bbox", bbox, np.array(self.prior.resolution) * self.prior.unit),
            ("unit", unit, self.prior.unit),
        ):
            if supplied is None:
                continue
            supplied = as_numpy(supplied)
            if supplied.ndim == 0:
                supplied = np.repeat(supplied, 3)
            if supplied.shape != (3,) or not np.allclose(supplied, expected):
                raise ValueError(f"GCPR16 weights require {name}={list(expected)}; got {supplied}. "
                                 "Instantiate vehicle_reconstructor() without adaptive-grid arguments.")

    def _sync_metadata(self):
        self.k = self.prior.k
        self.unit = like(self.prior.unit, self._reference)
        self.TSDF = SimpleNamespace(
            resolution=tuple(self.prior.resolution),
            bbox=like(np.array(self.prior.resolution) * self.prior.unit, self._reference),
            origin=like(self.prior.origin, self._reference),
        )
        self._U = None
        self._S = like(self.prior.S, self._reference)
        self._V = like(self.prior.V, self._reference)
        self.batch_tsdf_mean = like(self.prior.mean, self._reference)

    @property
    def device(self):
        return self._reference.device if is_tensor(self._reference) else "cpu"

    def to(self, device=None, dtype=None):
        import torch
        self._reference = torch.empty(0, device=device or self.device, dtype=dtype or torch.float64)
        self._sync_metadata()
        return self

    def encode_aux(self, batch_tsdf):
        if not is_tensor(batch_tsdf):
            batch_tsdf = np.asarray(batch_tsdf, dtype=np.float64)
        if batch_tsdf.ndim not in (1, 2, 3, 4):
            raise ValueError('Expected native grids or flat GCPR16 fields.')
        if batch_tsdf.ndim < 3:
            valid = batch_tsdf.shape[-1] == self.prior.mean.size
        else:
            valid = tuple(batch_tsdf.shape[-3:]) == self.prior.resolution
        if not valid:
            raise ValueError('Expected native grids [B,60,40,60] or flat fields [B,144000].')
        if is_tensor(batch_tsdf):
            import torch
            if not batch_tsdf.is_floating_point():
                batch_tsdf = batch_tsdf.to(dtype=torch.float64)
            if not torch.isfinite(batch_tsdf).all():
                raise ValueError('TSDF fields must be finite.')
            values = batch_tsdf.reshape(-1, self.prior.mean.size)
            basis = torch.as_tensor(self.prior.V.copy(), device=values.device, dtype=values.dtype)
            mean = torch.as_tensor(self.prior.mean.copy(), device=values.device, dtype=values.dtype)
            return (values - mean) @ basis
        values = batch_tsdf.reshape(-1, self.prior.mean.size)
        if not np.isfinite(values).all():
            raise ValueError('TSDF fields must be finite.')
        return (values - self.prior.mean) @ self.prior.V

    def decode_aux(self, latent):
        if not is_tensor(latent):
            latent = np.asarray(latent, dtype=np.float64)
        if latent.ndim not in (1, 2) or latent.shape[-1] != self.k:
            raise ValueError(f"Shape coefficients must have shape [{self.k}] or [B,{self.k}].")
        if is_tensor(latent):
            import torch
            if not latent.is_floating_point():
                latent = latent.to(dtype=torch.float64)
            if not torch.isfinite(latent).all():
                raise ValueError('Shape coefficients must be finite.')
            values = latent.reshape(-1, self.k)
            basis = torch.as_tensor(self.prior.V.copy(), device=values.device, dtype=values.dtype)
            mean = torch.as_tensor(self.prior.mean.copy(), device=values.device, dtype=values.dtype)
            return values @ basis.T + mean
        values = np.asarray(latent, dtype=np.float64).reshape(-1, self.k)
        if not np.isfinite(values).all():
            raise ValueError('Shape coefficients must be finite.')
        return values @ self.prior.V.T + self.prior.mean

    def decode(self, latent, to_meshes=False):
        grid = self.decode_aux(latent).reshape(-1, *self.prior.resolution)
        if to_meshes:
            return self.to_meshes(grid)
        return grid

    def encode(self, vehicles):
        return self.encode_aux(self.prepare_tsdf(vehicles))

    def prepare_tsdf(self, vehicles):
        if isinstance(vehicles, np.ndarray) or is_tensor(vehicles):
            values = vehicles
            if values.ndim < 2 or values.shape[-1] != self.prior.mean.size:
                if tuple(values.shape[-3:]) != self.prior.resolution:
                    raise ValueError("Expected native GCPR16 grid [B,60,40,60] or [B,144000].")
            return values.reshape(-1, *self.prior.resolution)
        raise NotImplementedError("This pretrained decoder accepts native SDF grids, not mesh-to-SDF training inputs.")

    def reconstruct(self, vehicles):
        return self.decode(self.encode(vehicles))

    reconsturct = reconstruct  # Preserve the misspelled public method in the existing project.

    def fit_meshes(self, vehicles, k=5):
        raise NotImplementedError("GCPR16 loads data/matrices directly; omit fit_meshes() and dataset preprocessing.")

    def to_meshes(self, grids):
        from skimage.measure import marching_cubes
        verts, faces = [], []
        for grid in as_numpy(grids).reshape(-1, *self.prior.resolution):
            if not (grid.min() < 0 < grid.max()):
                verts.append(np.empty((0, 3)))
                faces.append(np.empty((0, 3), dtype=int))
                continue
            vertices, triangles, _, _ = marching_cubes(grid, level=0, spacing=tuple(self.prior.unit))
            verts.append(vertices + self.prior.origin)
            faces.append(triangles)
        return make_meshes(verts, faces, grids if is_tensor(grids) else self._reference)

    def save_parameters(self, path):
        """Store an optional numeric snapshot, without pickle or retraining."""
        with Path(path).open("wb") as stream:
            np.savez(stream, V=self.prior.V, S=self.prior.S, mean=self.prior.mean,
                     resolution=self.prior.resolution, origin=self.prior.origin, unit=self.prior.unit)

    def load_parameters(self, path=None):
        if path is None or Path(path).is_dir():
            self.prior = ShapePrior(matrices_dir=path, k=self.k)
        else:
            with np.load(path, allow_pickle=False) as snapshot:
                for key, expected in (("resolution", self.prior.resolution),
                                      ("origin", self.prior.origin), ("unit", self.prior.unit)):
                    if key not in snapshot or not np.allclose(snapshot[key], expected):
                        raise ValueError(f"Snapshot has incompatible {key}.")
                V, S, mean = [np.asarray(snapshot[key], dtype=np.float64).copy() for key in ("V", "S", "mean")]
            if (V.shape != self.prior.V.shape or S.shape != self.prior.S.shape or
                    mean.shape != self.prior.mean.shape or not all(np.isfinite(a).all() for a in (V, S, mean))
                    or np.any(S <= 0)):
                raise ValueError("Snapshot contains invalid prior arrays.")
            self.prior.V, self.prior.S, self.prior.mean = V, S, mean
        self._sync_metadata()
        return self


def quantify_bbox(bbox, unit):
    if is_tensor(bbox):
        return unit * (bbox / unit).ceil()
    return np.asarray(unit) * np.ceil(np.asarray(bbox) / np.asarray(unit))


GCPR16Reconstructor = vehicle_reconstructor
