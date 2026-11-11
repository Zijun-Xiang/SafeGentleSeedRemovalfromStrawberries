"""Small iLQR (Gauss-Newton) for the reduced task-space model.

State  x = [tip position (3), jaw closure (1)]
Input  u = [tip velocity (3), closure rate (1)]
Model  x_{k+1} = x_k + dt * u_k   (the arm and hand track velocity commands)

The cost is a sum of squared residuals, so only first derivatives are needed;
they are taken by finite differences over the whole horizon at once.
"""
import numpy as np


def _rollout(x0, U, dt):
    X = np.empty((len(U) + 1, len(x0)))
    X[0] = x0
    X[1:] = x0 + dt * np.cumsum(U, axis=0)
    return X


def _cost(X, U, res, res_T):
    r = res(X[:-1], U)
    rT = res_T(X[-1])
    return 0.5 * float(np.sum(r * r) + rT @ rT)


def ilqr(x0, U, res, res_T, dt, iters=2, reg=1e-4, eps=1e-6):
    """Returns (U_opt, X_opt, cost). `res(X, U)` maps (N,n),(N,m) -> (N,r); `res_T(x)` -> (rT,)."""
    N, m = U.shape
    n = len(x0)
    X = _rollout(x0, U, dt)
    J = _cost(X, U, res, res_T)
    for _ in range(iters):
        r0 = res(X[:-1], U)
        Rx = np.empty((N, r0.shape[1], n)); Ru = np.empty((N, r0.shape[1], m))
        for i in range(n):
            Xp = X[:-1].copy(); Xp[:, i] += eps
            Rx[:, :, i] = (res(Xp, U) - r0) / eps
        for i in range(m):
            Up = U.copy(); Up[:, i] += eps
            Ru[:, :, i] = (res(X[:-1], Up) - r0) / eps
        rT = res_T(X[-1])
        RT = np.empty((len(rT), n))
        for i in range(n):
            xp = X[-1].copy(); xp[i] += eps
            RT[:, i] = (res_T(xp) - rT) / eps
        Vx, Vxx = RT.T @ rT, RT.T @ RT
        k = np.empty((N, m)); K = np.empty((N, m, n))
        for t in range(N - 1, -1, -1):
            lx, lu = Rx[t].T @ r0[t], Ru[t].T @ r0[t]
            lxx, luu, lux = Rx[t].T @ Rx[t], Ru[t].T @ Ru[t], Ru[t].T @ Rx[t]
            Qx = lx + Vx
            Qu = lu + dt * Vx
            Qxx = lxx + Vxx
            Quu = luu + dt * dt * Vxx + reg * np.eye(m)
            Qux = lux + dt * Vxx
            Quu_inv = np.linalg.inv(Quu)
            k[t] = -Quu_inv @ Qu
            K[t] = -Quu_inv @ Qux
            Vx = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
            Vxx = 0.5 * (Vxx + Vxx.T)
        improved = False
        for a in (1.0, 0.5, 0.25, 0.1):
            Xn = np.empty_like(X); Un = np.empty_like(U)
            Xn[0] = x0
            for t in range(N):
                Un[t] = U[t] + a * k[t] + K[t] @ (Xn[t] - X[t])
                Xn[t + 1] = Xn[t] + dt * Un[t]
            Jn = _cost(Xn, Un, res, res_T)
            if Jn < J:
                X, U, J, improved = Xn, Un, Jn, True
                break
        if not improved:
            break
    return U, X, J
