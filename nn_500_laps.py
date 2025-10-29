# ----------------------------------------------------------------------------------------------------------------------
# Baseline (your original flow) vs Robust NN-Residual Benchmark — Metrics Only
# - Baseline: PID -> Regression (ridge) -> MPC -> TV-MPC -> LMPC  (unchanged)
# - Robust NN-Residual: adds standardized, weighted MLP ensemble residual + uncertainty-aware blending of (A,B)
# - Benchmarks up to 100 LMPC laps; plots only metrics (no XY track/animations)
# ----------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append('fnc/simulator')
sys.path.append('fnc/controller')
sys.path.append('fnc')

import numpy as np
import matplotlib.pyplot as plt

from initControllerParameters import initMPCParams, initLMPCParams
from PredictiveControllers import MPC, LMPC
from PredictiveModel import PredictiveModel
from Utilities import Regression, PID
from SysModel import Simulator
from Track import Map


# ======================= Small helpers =======================

def build_dataset(x_cl: np.ndarray, u_cl: np.ndarray):
    """Return design Phi=[x_k, u_k], target Y=x_{k+1}."""
    assert x_cl.shape[0] == u_cl.shape[0], "x and u must have same length"
    X = x_cl[:-1, :]
    U = u_cl[:-1, :]
    Y = x_cl[1:, :]
    Phi = np.hstack([X, U])
    return Phi, Y

def split_AB(W: np.ndarray, n: int, d: int):
    """Split stacked W=[A^T; B^T]^T with A in first n rows (before transpose)."""
    A = W[:n, :].T
    B = W[n:n+d, :].T
    return A, B

def stabilize_A(A: np.ndarray, rho_cap: float = 0.985):
    """Spectral radius cap to keep predictions stable."""
    eigvals = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eigvals)))
    if rho > rho_cap:
        A = A * (rho_cap / (rho + 1e-12))
    return A

def control_energy(u: np.ndarray):
    if u is None or u.size == 0: return np.nan
    return float(np.sum(u**2))

def path_length_xy(x_glob: np.ndarray):
    if x_glob is None or x_glob.shape[0] < 2 or x_glob.shape[1] < 2: return np.nan
    dxy = np.diff(x_glob[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(dxy, axis=1)))

def extract_lap_times(lmpc, dt: float):
    laps = []
    if not hasattr(lmpc, 'Qfun'): return laps
    try:
        for i in range(len(lmpc.Qfun)):
            val = lmpc.Qfun[i]
            laps.append(float(val[0]) * dt if isinstance(val, (list, tuple, np.ndarray)) else float(val) * dt)
    except Exception:
        pass
    return laps


# ======================= Standardizer (z-score) =======================

class Standardizer:
    def __init__(self, eps: float = 1e-8):
        self.mu = None
        self.sigma = None
        self.eps = eps
    def fit(self, X: np.ndarray):
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
        self.sigma = np.where(self.sigma < 1e-6, 1.0, self.sigma)  # avoid divide-by-zero
        return self
    def transform(self, X: np.ndarray):
        return (X - self.mu) / (self.sigma + self.eps)
    def inv_jacobian(self):
        """Jacobian d z / d x for z-score is diag(1/sigma). We need this to map Jacobians back to original scale."""
        return np.diag(1.0 / (self.sigma + self.eps))


# ======================= Two-layer MLP (residual) + Adam + weighted loss =======================

class MLP2:
    """
    Residual network r = W3*tanh(W2*tanh(W1*z + b1) + b2) + b3, where z = standardized [x;u].
    """
    def __init__(self, in_dim, out_dim, h1=64, h2=64, seed=0, dropout=0.0):
        rng = np.random.default_rng(seed)
        k1 = np.sqrt(1.0 / in_dim)
        k2 = np.sqrt(1.0 / h1)
        k3 = np.sqrt(1.0 / h2)
        self.W1 = rng.uniform(-k1, k1, size=(h1, in_dim)); self.b1 = np.zeros((h1,))
        self.W2 = rng.uniform(-k2, k2, size=(h2, h1));    self.b2 = np.zeros((h2,))
        self.W3 = rng.uniform(-k3, k3, size=(out_dim, h2)); self.b3 = np.zeros((out_dim,))
        # Adam buffers
        self.m = [np.zeros_like(self.W1), np.zeros_like(self.b1),
                  np.zeros_like(self.W2), np.zeros_like(self.b2),
                  np.zeros_like(self.W3), np.zeros_like(self.b3)]
        self.v = [np.zeros_like(self.W1), np.zeros_like(self.b1),
                  np.zeros_like(self.W2), np.zeros_like(self.b2),
                  np.zeros_like(self.W3), np.zeros_like(self.b3)]
        self.t = 0
        self.dropout = float(dropout)

    @staticmethod
    def _tanh(z): return np.tanh(z)
    @staticmethod
    def _dtanh(z): return 1.0 - np.tanh(z)**2

    def _apply_dropout(self, H, training: bool):
        if not training or self.dropout <= 0.0: return H
        mask = (np.random.rand(*H.shape) >= self.dropout).astype(H.dtype)
        # Inverted dropout: scale to keep expected activation same
        return (H * mask) / (1.0 - self.dropout)

    def forward(self, z, training: bool):
        z1 = self.W1 @ z.T + self.b1[:, None]     # (h1, N)
        h1 = self._tanh(z1)
        h1 = self._apply_dropout(h1, training)
        z2 = self.W2 @ h1 + self.b2[:, None]      # (h2, N)
        h2 = self._tanh(z2)
        h2 = self._apply_dropout(h2, training)
        out = self.W3 @ h2 + self.b3[:, None]     # (n, N)
        return out.T, (z, z1, h1, z2, h2)

    def loss_and_grads(self, z, target, weights=None, weight_decay=0.0):
        """
        Weighted MSE: mean_i w_i * ||err_i||^2 ; weights normalized to mean 1 for stable scale.
        """
        N = z.shape[0]
        out, (z_, z1, h1, z2, h2) = self.forward(z, training=True)
        err = out - target                          # (N,n)

        if weights is None:
            weights = np.ones((N,), dtype=z.dtype)
        w = weights.reshape(-1, 1)
        w = w * (len(w) / (np.sum(w) + 1e-12))     # normalize mean weight ~1
        loss = float(np.mean(np.sum(w * (err**2), axis=1)))

        dOut = (2.0 / N) * (w * err).T             # (n,N)
        dW3 = dOut @ h2.T                          # (n,h2)
        db3 = np.sum(dOut, axis=1)                 # (n,)
        dh2 = self.W3.T @ dOut                     # (h2,N)
        dz2 = dh2 * self._dtanh(z2)                # (h2,N)
        dW2 = dz2 @ h1.T                           # (h2,h1)
        db2 = np.sum(dz2, axis=1)                  # (h2,)
        dh1 = self.W2.T @ dz2                      # (h1,N)
        dz1 = dh1 * self._dtanh(z1)                # (h1,N)
        dW1 = dz1 @ z                              # (h1,in_dim)
        db1 = np.sum(dz1, axis=1)                  # (h1,)

        if weight_decay > 0.0:
            loss += weight_decay * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
            dW3 += 2*weight_decay*self.W3
            dW2 += 2*weight_decay*self.W2
            dW1 += 2*weight_decay*self.W1

        grads = (dW1, db1, dW2, db2, dW3, db3)
        return loss, grads

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = beta1*self.m[i] + (1-beta1)*g
            self.v[i] = beta2*self.v[i] + (1-beta2)*(g**2)
            mhat = self.m[i] / (1 - beta1**self.t)
            vhat = self.v[i] / (1 - beta2**self.t)
            params[i] -= lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, z, target, weights=None, epochs=220, batch=256, lr=1e-3,
            weight_decay=1e-5, cosine=True, early_stop_patience=25, verbose=False):
        N = z.shape[0]; idx = np.arange(N)
        best_loss = np.inf; best_params = None; patience = 0
        for ep in range(1, epochs+1):
            lr_ep = lr*(0.5*(1 + np.cos(np.pi*(ep-1)/max(1,epochs-1)))) if cosine else lr
            np.random.shuffle(idx)
            for s in range(0, N, batch):
                j = idx[s:s+batch]
                wj = None if weights is None else weights[j]
                loss, grads = self.loss_and_grads(z[j], target[j], weights=wj, weight_decay=weight_decay)
                self.adam_step(grads, lr=lr_ep)
            if verbose and (ep % 25 == 0 or ep == 1):
                print(f"[MLP] epoch {ep:3d}  loss={loss:.6f}")
            if loss + 1e-9 < best_loss:
                best_loss = loss; patience = 0
                best_params = (self.W1.copy(), self.b1.copy(),
                               self.W2.copy(), self.b2.copy(),
                               self.W3.copy(), self.b3.copy())
            else:
                patience += 1
                if patience >= early_stop_patience:
                    break
        if best_params is not None:
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = best_params

    def jacobian_at(self, z_anchor):
        """J = d out / d z (standardized input) at a single anchor (n x (n+d))."""
        z1 = self.W1 @ z_anchor + self.b1
        s1 = 1.0 - np.tanh(z1)**2
        z2 = self.W2 @ np.tanh(z1) + self.b2
        s2 = 1.0 - np.tanh(z2)**2
        return ((self.W3 * s2[None, :]) @ self.W2) @ ((np.eye(self.W1.shape[0]) * s1) @ self.W1)


# ======================= Identification helpers =======================

def pick_anchors(Phi, K=10, rng=None):
    """Farthest-point sampling → diverse anchors."""
    if rng is None: rng = np.random.default_rng(0)
    N = Phi.shape[0]
    if N <= K: return Phi
    anchors = []
    idx0 = rng.integers(0, N); anchors.append(Phi[idx0])
    dists = np.linalg.norm(Phi - anchors[0], axis=1)
    for _ in range(1, K):
        idx = int(np.argmax(dists)); anchors.append(Phi[idx])
        dists = np.minimum(dists, np.linalg.norm(Phi - anchors[-1], axis=1))
    return np.stack(anchors, axis=0)

def robust_AB_from_residual_ensemble(
    Phi, Y, n, d,
    base_lambda=3e-7, rho_cap=0.992,
    hidden=(64,64), epochs=200, lr=1e-3, wd=1e-5,
    M=4, K_anchors=12, seed=0, dropout=0.05,
    weight_gamma=0.5,          # weight ↑ with ||u||^2 (emphasize informative regimes)
    alpha_uncert=3.0,          # higher => stronger shrink to baseline when uncertain
    cap_A=0.25, cap_B=0.35     # norm caps for corrections
):
    """
    1) Base ridge (A,B).
    2) Train M MLPs on standardized residuals with weighted loss.
    3) Average Jacobians over multiple anchors; compute ensemble mean & variance.
    4) Uncertainty-aware blending toward baseline + norm caps + spectral cap.
    Returns A_eff, B_eff and diagnostics dict.
    """
    # Base ridge
    n_plus_d = Phi.shape[1]
    W = np.linalg.solve(Phi.T @ Phi + base_lambda*np.eye(n_plus_d), Phi.T @ Y)
    A_base, B_base = split_AB(W, n, d)
    A_base = stabilize_A(A_base, rho_cap=rho_cap)

    # Targets: residuals
    base_pred = Phi[:, :n] @ A_base.T + Phi[:, n:n+d] @ B_base.T
    R = Y - base_pred

    # Standardize inputs; build weights (1 + gamma*||u||^2)
    std = Standardizer().fit(Phi)
    Z = std.transform(Phi)
    U = Phi[:, n:n+d]
    w = 1.0 + weight_gamma * np.sum(U**2, axis=1)

    # Train ensemble
    rng = np.random.default_rng(seed)
    mlps = []
    for m in range(M):
        mlp = MLP2(in_dim=n+d, out_dim=n, h1=hidden[0], h2=hidden[1],
                   seed=seed + 31*m, dropout=dropout)
        # slight jitter for robustness
        Z_t = Z + 0.01 * rng.standard_normal(Z.shape)
        mlp.fit(Z_t, R, weights=w, epochs=epochs, batch=256, lr=lr,
                weight_decay=wd, cosine=True, early_stop_patience=25, verbose=False)
        mlps.append(mlp)

    # Multi-anchor Jacobians (standardized space)
    anchors = pick_anchors(Phi, K=K_anchors, rng=rng)
    Z_anchors = std.transform(anchors)
    J_list = []
    for mlp in mlps:
        J_acc = np.zeros((n, n+d))
        for za in Z_anchors:
            J_acc += mlp.jacobian_at(za)
        J_list.append(J_acc / len(Z_anchors))
    J_stack = np.stack(J_list, axis=0)    # (M, n, n+d)

    # Map Jacobians back to original variable scaling:
    # out = r(z), z = (phi - mu)/sigma  =>  dr/dphi = dr/dz * dz/dphi  with dz/dphi = diag(1/sigma)
    Dz = std.inv_jacobian()               # (n+d, n+d)
    J_stack_orig = J_stack @ Dz           # (M, n, n+d)
    J_mean = np.mean(J_stack_orig, axis=0)
    J_var  = np.mean((J_stack_orig - J_mean[None, :, :])**2, axis=0)
    # scalar uncertainty — average Frobenius variance over A,B blocks
    Jx_mean = J_mean[:, :n]
    Ju_mean = J_mean[:, n:n+d]
    var_A = np.mean(J_var[:, :n])
    var_B = np.mean(J_var[:, n:n+d])
    uncert = 0.5*(var_A + var_B)

    # Uncertainty-aware blending + norm caps
    w_blend = 1.0 / (1.0 + alpha_uncert * uncert)
    # cap correction norms
    def cap_fro(Mcorr, cap):
        fro = np.linalg.norm(Mcorr, ord='fro')
        return Mcorr if fro <= cap or fro < 1e-12 else (Mcorr * (cap / fro))
    Jx_c = cap_fro(Jx_mean, cap_A)
    Ju_c = cap_fro(Ju_mean, cap_B)

    A_eff = stabilize_A(A_base + w_blend * Jx_c, rho_cap=0.985)
    B_eff = B_base + w_blend * Ju_c

    diag = dict(
        w_blend=float(w_blend),
        uncert=float(uncert),
        corr_norm_A=float(np.linalg.norm(Jx_mean, ord='fro')),
        corr_norm_B=float(np.linalg.norm(Ju_mean, ord='fro'))
    )
    return A_eff, B_eff, diag


# ======================= Pipelines =======================

def run_baseline_pipeline(map_obj, xS, vt, n, d, N, dt=0.1, laps_target=100):
    """Your original flow; metrics only."""
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps_def, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map_obj, N)
    Laps = max(Laps_def, laps_target)

    simulator     = Simulator(map_obj)
    LMPCsimulator = Simulator(map_obj, multiLap=False, flagLMPC=True)

    # PID
    PIDController = PID(vt)
    xPID, uPID, xPID_glob, _ = simulator.sim(xS, PIDController)

    # Ridge (uses your utilities)
    lamb = 1e-7
    A, B, _ = Regression(xPID, uPID, lamb)
    A = stabilize_A(A, rho_cap=0.992)

    # LTI-MPC warm-up
    mpcParam.A = A; mpcParam.B = B
    mpc = MPC(mpcParam); _ = simulator.sim(xS, mpc)

    # TV-MPC warm-up
    predictiveModel = PredictiveModel(n, d, map_obj, 1)
    predictiveModel.addTrajectory(xPID, uPID)
    ltvmpcParam.timeVarying = True
    mpc_tv = MPC(ltvmpcParam, predictiveModel); _ = simulator.sim(xS, mpc_tv)

    # LMPC init
    lmpcpredictiveModel = PredictiveModel(n, d, map_obj, 4)
    for _ in range(4): lmpcpredictiveModel.addTrajectory(xPID, uPID)
    lmpcParameters.timeVarying = True
    lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for _ in range(4): lmpc.addTrajectory(xPID, uPID, xPID_glob)

    laps, energy, path = [], [], []
    for it in range(numSS_it, Laps):
        xLMPC, uLMPC, xLMPC_glob, xS = LMPCsimulator.sim(xS, lmpc)
        lmpc.addTrajectory(xLMPC, uLMPC, xLMPC_glob); lmpcpredictiveModel.addTrajectory(xLMPC, uLMPC)

        lt = extract_lap_times(lmpc, dt=dt); laps.append(lt[-1] if lt else np.nan)
        energy.append(control_energy(uLMPC))
        path.append(path_length_xy(xLMPC_glob))

    return {"lap_times": laps, "energy_last": energy, "path_last": path}


def run_nn_residual_robust_pipeline(map_obj, xS, vt, n, d, N, dt=0.1, laps_target=100):
    """
    Robust NN residual identification with uncertainty-aware blending and periodic retraining.
    """
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps_def, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map_obj, N)
    Laps = max(Laps_def, laps_target)

    simulator     = Simulator(map_obj)
    LMPCsimulator = Simulator(map_obj, multiLap=False, flagLMPC=True)

    # PID
    PIDController = PID(vt)
    xPID, uPID, xPID_glob, _ = simulator.sim(xS, PIDController)

    # Initial robust (A_eff,B_eff)
    Phi_PID, Y_PID = build_dataset(xPID, uPID)
    A_eff, B_eff, diag0 = robust_AB_from_residual_ensemble(
        Phi_PID, Y_PID, n, d,
        base_lambda=3e-7, rho_cap=0.992,
        hidden=(64,64), epochs=200, lr=1e-3, wd=1e-5,
        M=4, K_anchors=12, seed=0, dropout=0.05,
        weight_gamma=0.5, alpha_uncert=3.0, cap_A=0.25, cap_B=0.35
    )

    # LTI-MPC warm
    mpcParam.A = A_eff; mpcParam.B = B_eff
    mpc = MPC(mpcParam); xMPC, uMPC, xMPC_glob, _ = simulator.sim(xS, mpc)

    # TV-MPC warm
    predictiveModel = PredictiveModel(n, d, map_obj, 1)
    predictiveModel.addTrajectory(xPID, uPID)
    ltvmpcParam.timeVarying = True
    mpc_tv = MPC(ltvmpcParam, predictiveModel); _ = simulator.sim(xS, mpc_tv)

    # LMPC init
    lmpcpredictiveModel = PredictiveModel(n, d, map_obj, 4)
    for _ in range(4): lmpcpredictiveModel.addTrajectory(xPID, uPID)
    lmpcParameters.timeVarying = True
    lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for _ in range(4): lmpc.addTrajectory(xPID, uPID, xPID_glob)

    # Accumulate data; periodic retrain
    X_hist = [xPID, xMPC]; U_hist = [uPID, uMPC]
    milestones = {1, 10, 25, 50, 75}

    laps, energy, path = [], [], []
    for it in range(numSS_it, Laps):
        if it in milestones:
            x_all = np.vstack(X_hist); u_all = np.vstack(U_hist)
            Phi_all, Y_all = build_dataset(x_all, u_all)
            A_eff, B_eff, diag = robust_AB_from_residual_ensemble(
                Phi_all, Y_all, n, d,
                base_lambda=3e-7, rho_cap=0.992,
                hidden=(64,64),
                epochs=180 if it>1 else 220,
                lr=8e-4, wd=2e-5,
                M=5 if it>=25 else 4,
                K_anchors=16 if it>=25 else 12,
                seed=it, dropout=0.05,
                weight_gamma=0.6 if it>=25 else 0.5,
                alpha_uncert=3.5 if it>=25 else 3.0,
                cap_A=0.25, cap_B=0.35
            )
            mpcParam.A = A_eff; mpcParam.B = B_eff

        xLMPC, uLMPC, xLMPC_glob, xS = LMPCsimulator.sim(xS, lmpc)
        lmpc.addTrajectory(xLMPC, uLMPC, xLMPC_glob); lmpcpredictiveModel.addTrajectory(xLMPC, uLMPC)

        lt = extract_lap_times(lmpc, dt=dt); laps.append(lt[-1] if lt else np.nan)
        energy.append(control_energy(uLMPC))
        path.append(path_length_xy(xLMPC_glob))

        X_hist.append(xLMPC); U_hist.append(uLMPC)

    return {"lap_times": laps, "energy_last": energy, "path_last": path}


# ======================= Metrics & Summary =======================

def plot_benchmark(baseline, nn, title_suffix=""):
    def pad(a, L):
        out = np.full(L, np.nan, dtype=float)
        out[:min(L, len(a))] = a[:min(L, len(a))]
        return out

    L = max(len(baseline["lap_times"]), len(nn["lap_times"]))
    iters = np.arange(1, L+1)

    LA = pad(np.asarray(baseline["lap_times"]), L)
    LB = pad(np.asarray(nn["lap_times"]), L)

    best_A = np.minimum.accumulate(np.nan_to_num(LA, nan=np.inf))
    best_B = np.minimum.accumulate(np.nan_to_num(LB, nan=np.inf))
    improvement = LB - LA  # negative => NN faster

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax1, ax2, ax3 = axes

    ax1.plot(iters, LA, label='Baseline (Ridge)')
    ax1.plot(iters, LB, label='Robust NN-Residual')
    ax1.set_title('Lap Time per Iteration (up to 100)')
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('Lap time (s)')
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(iters, best_A, label='Baseline (best-so-far)')
    ax2.plot(iters, best_B, label='Robust NN (best-so-far)')
    ax2.set_title('Best-So-Far Lap Time')
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Best lap (s)')
    ax2.grid(True, alpha=0.3); ax2.legend()

    ax3.plot(iters, improvement, label='(Robust NN − Baseline)  ↓ better')
    ax3.axhline(0.0, color='k', linewidth=1)
    ax3.set_title('Improvement vs Baseline (negative = faster)')
    ax3.set_xlabel('Iteration'); ax3.set_ylabel('Seconds')
    ax3.grid(True, alpha=0.3); ax3.legend()

    if title_suffix:
        fig.suptitle(f"LMPC Metrics — {title_suffix}", y=0.98)
    fig.tight_layout()
    plt.show()

def print_summary(baseline, nn):
    def best_last(m):
        laps = m["lap_times"]
        best = np.nanmin(laps) if len(laps) else np.nan
        last = laps[-1] if len(laps) else np.nan
        return best, last
    def fmt(x):
        return "-" if (x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))) else f"{x:.2f}"

    bestA, lastA = best_last(baseline)
    bestB, lastB = best_last(nn)
    print(
        f"\n{'LMPC METRICS SUMMARY (Baseline vs Robust NN-Residual)':^84}\n"
        f"{'-'*84}\n"
        f"{'':28} | {'Baseline':>12} | {'Robust NN':>12}\n"
        f"{'-'*84}\n"
        f"{'Best lap (s)':28} | {fmt(bestA):>12} | {fmt(bestB):>12}\n"
        f"{'Last lap (s)':28} | {fmt(lastA):>12} | {fmt(lastB):>12}\n"
        f"{'-'*84}\n"
    )


# ======================= Main =======================

def main():
    # Match your original initialization
    N = 14
    n = 6; d = 2
    dt = 0.1
    x0 = np.array([0.5, 0, 0, 0, 0, 0])
    xS = [x0, x0]
    map_obj = Map(0.4)   # same as your code
    vt = 0.8             # target velocity

    laps_target = 200    # benchmark up to 100 LMPC iterations

    print("\n=== BASELINE (your original flow) ===")
    metrics_baseline = run_baseline_pipeline(map_obj, xS, vt, n, d, N, dt=dt, laps_target=laps_target)

    print("\n=== ROBUST NN-RESIDUAL (std + weighted + ensemble + uncertainty blending) ===")
    metrics_nn = run_nn_residual_robust_pipeline(map_obj, xS, vt, n, d, N, dt=dt, laps_target=laps_target)

    print_summary(metrics_baseline, metrics_nn)
    plot_benchmark(metrics_baseline, metrics_nn,
                   title_suffix="Baseline vs Robust NN-Residual — 100 Laps")

if __name__ == "__main__":
    main()
