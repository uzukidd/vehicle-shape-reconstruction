"""PCA shape space and trilinear TSDF evaluation for the GCPR16 model."""

from pathlib import Path
import struct

import numpy as np


_RESOLUTION = (60, 40, 60)
_SIZE = 144000
_CORNERS = np.array([(x, y, z) for x in (0, 1) for y in (0, 1)
                     for z in (0, 1)], dtype=np.int64)


def load_eigen_matrix(path):
    """Read little-endian Eigen doubles with a 32- or 64-bit index header.

    Dimensions must be positive and the file must contain exactly the declared
    column-major payload. Nonfinite values and ambiguous headers are rejected.
    """
    path = Path(path)
    payload = path.read_bytes()
    candidates = []
    for header_size, fmt in ((16, '<qq'), (8, '<ii')):
        if len(payload) < header_size:
            continue
        rows, cols = struct.unpack_from(fmt, payload)
        if rows > 0 and cols > 0 and header_size + rows * cols * 8 == len(payload):
            candidates.append((header_size, rows, cols))
    if len(candidates) != 1:
        raise ValueError(f'{path}: invalid Eigen dimensions or file size')
    header_size, rows, cols = candidates[0]
    matrix = np.frombuffer(payload, dtype='<f8', offset=header_size).reshape(
        (rows, cols), order='F').copy()
    if not np.isfinite(matrix).all():
        raise ValueError(f'{path}: matrix contains nonfinite values')
    return matrix


def _finite_array(value, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be a finite numeric array') from exc
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain only finite values')
    return array


class ShapePrior:
    """The original 60 x 40 x 60 PCA TSDF, with z-fastest flattening."""

    def __init__(self, matrices_dir=None, k=5):
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k must be a positive integer')
        directory = (Path(__file__).resolve().parents[1] / 'data' / 'matrices'
                     if matrices_dir is None else Path(matrices_dir))
        singular = load_eigen_matrix(directory / 'S.bin')
        basis = load_eigen_matrix(directory / 'V.bin')
        mean = load_eigen_matrix(directory / 'mean_shape.bin')
        if singular.ndim != 2 or singular.shape[1] != 1:
            raise ValueError('S.bin must be a column vector')
        if basis.shape != (_SIZE, singular.shape[0]):
            raise ValueError('V.bin must have shape (144000, number of singular values)')
        if mean.shape != (_SIZE, 1):
            raise ValueError('mean_shape.bin must have shape (144000, 1)')
        if k > basis.shape[1]:
            raise ValueError(f'k exceeds the available {basis.shape[1]} PCA components')
        if np.any(singular[:k, 0] <= 0):
            raise ValueError('selected singular values must be strictly positive')
        self.resolution = _RESOLUTION
        self.origin = np.array([-3., -3., -3.])
        self.unit = np.array([.1, .1, .1])
        self.k = int(k)
        self.S = singular[:k, 0].copy()
        self.V = basis[:, :k].copy()
        self.mean = mean[:, 0].copy()

    def encode(self, flat):
        """Project [p] or [B,p] TSDF arrays onto the orthonormal PCA basis."""
        flat = _finite_array(flat, 'flat')
        if flat.ndim not in (1, 2) or flat.shape[-1] != _SIZE or flat.size == 0:
            raise ValueError('flat must have shape (144000,) or (B,144000), B > 0')
        return (flat - self.mean) @ self.V

    def decode(self, z):
        """Decode [k] or [B,k] coefficients; always return [B,60,40,60]."""
        z = _finite_array(z, 'z')
        if z.ndim not in (1, 2) or z.shape[-1] != self.k or z.size == 0:
            raise ValueError(f'z must have shape ({self.k},) or (B,{self.k}), B > 0')
        return (np.atleast_2d(z) @ self.V.T + self.mean).reshape(
            (-1,) + self.resolution)

    def sample(self, local_points, z, mode='shape', with_jacobian=True):
        """Evaluate TSDF and coefficient/space derivatives at local [N,3] points.

        Shape mode accepts lower voxel indices 0..dim-2; pose mode accepts
        1..dim-2. Outside, shape returns 3 and zero coefficient derivatives,
        while pose returns ||point||+3 and its spatial derivative. These
        original boundary rules are discontinuous; derivatives apply within
        each region. With derivatives disabled the second return value is None.
        """
        points = _finite_array(local_points, 'local_points')
        z = _finite_array(z, 'z')
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError('local_points must have shape (N,3), N > 0')
        if z.shape != (self.k,):
            raise ValueError(f'z must have shape ({self.k},)')
        if mode not in ('shape', 'pose'):
            raise ValueError("mode must be 'shape' or 'pose'")
        with np.errstate(over='ignore', invalid='ignore'):
            grid = (points - self.origin) / self.unit
            floored = np.floor(grid)
        # Compare before integer conversion so finite extreme points cannot
        # overflow an int64 voxel index and accidentally become interior.
        inside = np.all((floored >= (0 if mode == 'shape' else 1)) &
                        (floored <= np.asarray(self.resolution) - 2), axis=1)
        if mode == 'shape':
            values = np.full(len(points), 3.)
            jacobian = np.zeros((len(points), self.k)) if with_jacobian else None
        else:
            norms = np.hypot(np.hypot(points[:, 0], points[:, 1]), points[:, 2])
            values = norms + 3.
            jacobian = np.divide(points, norms[:, None], out=np.zeros_like(points),
                                 where=norms[:, None] > 0) if with_jacobian else None
        if not np.any(inside):
            return values, jacobian
        lower = floored[inside].astype(np.int64)
        fraction = grid[inside] - lower
        indices = lower[:, None, :] + _CORNERS[None, :, :]
        flat = (indices[..., 0] * self.resolution[1] * self.resolution[2] +
                indices[..., 1] * self.resolution[2] + indices[..., 2])
        corner_basis = self.V[flat]
        corners = self.mean[flat] + corner_basis @ z
        factors = np.where(_CORNERS[None, :, :] == 1,
                           fraction[:, None, :], 1. - fraction[:, None, :])
        weights = np.prod(factors, axis=2)
        values[inside] = np.sum(weights * corners, axis=1)
        if with_jacobian:
            if mode == 'shape':
                jacobian[inside] = np.einsum('ni,nik->nk', weights, corner_basis)
            else:
                for axis in range(3):
                    other = [a for a in range(3) if a != axis]
                    derivative = (np.prod(factors[:, :, other], axis=2) *
                                  (2 * _CORNERS[:, axis] - 1) / self.unit[axis])
                    jacobian[inside, axis] = np.sum(derivative * corners, axis=1)
        return values, jacobian
