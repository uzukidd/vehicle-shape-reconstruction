"""Alternating GCPR16 fitting with analytic derivatives and dense LM/IRLS.

The robust objective matches Ceres' 0.5 * rho(r**2) convention. The NumPy
trust-region implementation is numerically different from Ceres and is not
intended to produce bitwise identical iterates.
"""

from time import perf_counter

import numpy as np

from .prior import _finite_array


_HUBER_DELTA = .1


def rotation_y(angle):
    """Return the canonical camera-coordinate rotation around the y axis."""
    angle = _finite_array(angle, 'angle')
    if angle.ndim != 0:
        raise ValueError('angle must be a finite scalar')
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _inputs(prior, points, pose, z):
    points = _finite_array(points, 'points')
    pose = _finite_array(pose, 'pose')
    z = _finite_array(z, 'z')
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError('points must have shape (N,3), N > 0')
    if pose.shape != (4,):
        raise ValueError('pose must have shape (4,): tx, ty, tz, ry')
    if z.shape != (prior.k,):
        raise ValueError(f'z must have shape ({prior.k},)')
    return points, pose, z


def pose_residuals(prior, points, pose, z):
    """Return raw data residuals [N] and derivatives [N,4] for tx,ty,tz,ry."""
    points, pose, z = _inputs(prior, points, pose, z)
    rotation = rotation_y(pose[3])
    centered = points - pose[:3]
    local = centered @ rotation
    residual, spatial = prior.sample(local, z, mode='pose')
    c, s = np.cos(pose[3]), np.sin(pose[3])
    # dR/dyaw has a ZERO middle diagonal entry. This also avoids the
    # original C++ residual block's erroneous four-column write into a
    # three-column translation Jacobian.
    derivative = np.array([[-s, 0., c], [0., 0., 0.], [-c, 0., -s]])
    jacobian = np.empty((len(points), 4))
    jacobian[:, :3] = -spatial @ rotation.T
    jacobian[:, 3] = np.einsum('ni,ni->n', spatial, centered @ derivative)
    return residual, jacobian


def shape_residuals(prior, points, pose, z):
    """Return raw data residuals [N] and coefficient derivatives [N,k]."""
    points, pose, z = _inputs(prior, points, pose, z)
    local = (points - pose[:3]) @ rotation_y(pose[3])
    return prior.sample(local, z, mode='shape')


def _data_cost(residual):
    absolute = np.abs(residual)
    # Standard Huber = half of Ceres rho(s), s = residual squared.
    losses = np.empty_like(residual)
    small = absolute <= _HUBER_DELTA
    losses[small] = .5 * residual[small] ** 2
    losses[~small] = _HUBER_DELTA * (absolute[~small] - .5 * _HUBER_DELTA)
    return float(np.sum(losses) / (.03 * len(residual)))


def objective(prior, points, pose, z, vertical_reference=None, mode='shape'):
    """Evaluate one original subproblem objective, including its own prior.

    Shape: Huber data/(.03*N) + .5*sum((z/S)**2).
    Pose: Huber data/(.03*N) + (ty-reference)**2/6. If no reference is
    supplied, the current ty is used. A fit freezes the reference to ty at
    the start of EACH pose subproblem, exactly as the original code does.
    """
    points, pose, z = _inputs(prior, points, pose, z)
    if mode not in ('shape', 'pose'):
        raise ValueError("mode must be 'shape' or 'pose'")
    local = (points - pose[:3]) @ rotation_y(pose[3])
    residual, _ = prior.sample(local, z, mode=mode, with_jacobian=False)
    if mode == 'shape':
        penalty = .5 * np.sum((z / prior.S) ** 2)
    else:
        reference = pose[1] if vertical_reference is None else _finite_array(
            vertical_reference, 'vertical_reference')
        if np.ndim(reference) != 0:
            raise ValueError('vertical_reference must be a finite scalar')
        penalty = (pose[1] - reference) ** 2 / 6.
    return float(_data_cost(residual) + penalty)


def _linearized(residual, jacobian, regularizer, regularizer_jacobian):
    # IRLS uses rho'(r^2) = min(1, delta/|r|). Quadratic regularizers
    # stay unweighted; they are not included in the robust loss.
    weights = np.ones_like(residual)
    large = np.abs(residual) > _HUBER_DELTA
    weights[large] = _HUBER_DELTA / np.abs(residual[large])
    root = np.sqrt(weights / (.03 * len(residual)))
    return (np.concatenate((root * residual, regularizer)),
            np.vstack((root[:, None] * jacobian, regularizer_jacobian)))


def _lm(initial, evaluate, exact_cost, max_iterations):
    """Damped dense least squares; accept only exact objective improvements."""
    current = initial.copy()
    cost = exact_cost(current)
    initial_cost = cost
    damping = 1.e-3
    accepted = rejected = iterations = 0
    last_rank = None
    consecutive_rejections = 0
    status = 'maximum_iterations'
    usable = bool(np.isfinite(cost))
    if not usable:
        status = 'nonfinite_initial_cost'
    for iteration in range(max_iterations if usable else 0):
        iterations = iteration + 1
        residual, jacobian, regularizer, reg_jacobian = evaluate(current)
        b, a = _linearized(residual, jacobian, regularizer, reg_jacobian)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            status, usable = 'nonfinite_linearization', False
            break
        gradient = a.T @ b
        if np.linalg.norm(gradient, ord=np.inf) <= 1.e-9:
            status = 'gradient_tolerance'
            break
        # Scaling keeps poorly observable variables regularized without
        # forming normal equations. lstsq solves the augmented dense QR/SVD
        # problem and remains usable for rank deficient data Jacobians.
        scale = np.maximum(np.linalg.norm(a, axis=0), 1.e-4)
        augmented = np.vstack((a, np.diag(np.sqrt(damping) * scale)))
        rhs = np.concatenate((-b, np.zeros(len(current))))
        try:
            step, _, last_rank, _ = np.linalg.lstsq(augmented, rhs, rcond=None)
        except np.linalg.LinAlgError:
            status, usable = 'linear_solver_failure', False
            break
        if not np.isfinite(step).all():
            status, usable = 'nonfinite_step', False
            break
        if np.linalg.norm(step) <= 1.e-9 * (np.linalg.norm(current) + 1.e-9):
            status = 'step_tolerance'
            break
        candidate = current + step
        try:
            candidate_cost = exact_cost(candidate)
        except (ValueError, FloatingPointError, OverflowError):
            candidate_cost = np.inf
        actual = cost - candidate_cost
        predicted = -float(gradient @ step) - .5 * float(np.sum((a @ step) ** 2))
        if np.isfinite(candidate_cost) and actual > 0 and predicted > 0:
            previous = cost
            current, cost = candidate, candidate_cost
            accepted += 1
            consecutive_rejections = 0
            ratio = actual / predicted
            if ratio > .75:
                damping = max(damping / 3., 1.e-15)
            elif ratio < .25:
                damping = min(damping * 2., 1.e16)
            if actual <= 1.e-10 * (1. + abs(previous)):
                status = 'cost_tolerance'
                break
        else:
            rejected += 1
            consecutive_rejections += 1
            damping = min(damping * 10., 1.e16)
            if consecutive_rejections >= 16 or damping >= 1.e16:
                status = 'no_improvement'
                break
    return current, {
        'initial_cost': float(initial_cost), 'final_cost': float(cost),
        'iterations': iterations, 'accepted_steps': accepted,
        'rejected_steps': rejected, 'status': status, 'success': usable,
        'damping': float(damping),
        'linear_rank': None if last_rank is None else int(last_rank),
    }


def fit_vehicle(prior, points, initial_pose, initial_z=None,
                max_outer_iterations=10, max_solver_iterations=100,
                cost_tolerance=.1):
    """Alternate pose and shape LM solves with original objective scaling.

    History total_cost is the sum of each subproblem's final cost, evaluated
    when that subproblem completed, matching the original outer stop rule.
    final_costs reevaluates both subproblems at the returned common state;
    the history sum and this final diagnostic need not coincide. success
    means a finite usable solution, including a stationary/no-improvement
    state; termination and per-solver statuses explain convergence details.
    """
    started = perf_counter()
    z = np.zeros(prior.k) if initial_z is None else initial_z
    points, pose, z = _inputs(prior, points, initial_pose, z)
    pose, z = pose.copy(), z.copy()
    for name, value in (('max_outer_iterations', max_outer_iterations),
                        ('max_solver_iterations', max_solver_iterations)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    tolerance = _finite_array(cost_tolerance, 'cost_tolerance')
    if tolerance.ndim != 0 or tolerance < 0:
        raise ValueError('cost_tolerance must be a nonnegative finite scalar')
    history = []
    previous_total = np.inf
    success = True
    termination = 'maximum_outer_iterations'
    reference = float(pose[1])
    for outer in range(max_outer_iterations):
        reference = float(pose[1])

        def pose_evaluate(value):
            r, j = pose_residuals(prior, points, value, z)
            reg_j = np.zeros((1, 4))
            reg_j[0, 1] = 1. / np.sqrt(3.)
            return r, j, np.array([(value[1] - reference) / np.sqrt(3.)]), reg_j

        pose, pose_info = _lm(
            pose, pose_evaluate,
            lambda value: objective(prior, points, value, z, reference, 'pose'),
            max_solver_iterations)

        def shape_evaluate(value):
            r, j = shape_residuals(prior, points, pose, value)
            return r, j, value / prior.S, np.diag(1. / prior.S)

        z, shape_info = _lm(
            z, shape_evaluate,
            lambda value: objective(prior, points, pose, value, mode='shape'),
            max_solver_iterations)
        total = pose_info['final_cost'] + shape_info['final_cost']
        history.append({
            'outer_iteration': outer + 1,
            'pose_cost': pose_info['final_cost'],
            'shape_cost': shape_info['final_cost'], 'total_cost': float(total),
            'pose_iterations': pose_info['iterations'],
            'shape_iterations': shape_info['iterations'],
            'vertical_reference': reference,
            'pose_solver': pose_info, 'shape_solver': shape_info,
        })
        success = success and pose_info['success'] and shape_info['success']
        if not success:
            termination = 'solver_failure'
            break
        if previous_total - total <= tolerance:
            termination = 'outer_cost_tolerance'
            break
        previous_total = total
    final_pose_cost = objective(prior, points, pose, z, reference, 'pose')
    final_shape_cost = objective(prior, points, pose, z, mode='shape')
    return {
        'pose': pose, 'z': z, 'history': history, 'success': bool(success),
        'termination': termination,
        'elapsed_seconds': perf_counter() - started,
        'final_costs': {'pose_cost': final_pose_cost, 'shape_cost': final_shape_cost,
                        'total_cost': final_pose_cost + final_shape_cost,
                        'vertical_reference': reference},
    }
