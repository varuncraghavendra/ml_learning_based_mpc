# ----------------------------------------------------------------------------------------------------------------------
# LMPC Benchmark with Dual Real-Time Dashboards (Baseline vs Sim-Aware) — “2 cars racing” view
#
# What’s inside
# -------------
# • Pipeline A: Baseline ridge fit with fixed λ = 1e-7 (+ tiny stability cap).
# • Pipeline B: Simulation-aware λ selection via multi-step rollout error on held-out data
#               + stability cap on A + continual refit on PID+MPC data.
# • Two synchronized, real-time dashboards (left = Baseline, right = Sim-Aware) that replay
#   each pipeline’s **last LMPC lap** like two cars racing head-to-head.
# • Per-frame overlays: elapsed time, step index, cumulative control energy, path length.
#
# Drop this file in place of your main.py and run.
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('fnc/simulator')
sys.path.append('fnc/controller')
sys.path.append('fnc')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from initControllerParameters import initMPCParams, initLMPCParams
from PredictiveControllers import MPC, LMPC
from PredictiveModel import PredictiveModel
from Utilities import Regression, PID
from SysModel import Simulator
from Track import Map


# ======================= Helpers: dataset, ridge fit, stability, metrics =======================
def build_dataset(x_cl: np.ndarray, u_cl: np.ndarray):
    assert x_cl.shape[0] == u_cl.shape[0], "x and u must have same length"
    if x_cl.shape[0] < 3:
        raise ValueError("Not enough samples")
    X = x_cl[:-1, :]
    U = u_cl[:-1, :]
    Y = x_cl[1:, :]
    Phi = np.hstack([X, U])
    return Phi, Y

def ridge_AB(Phi: np.ndarray, Y: np.ndarray, n: int, d: int, lamb: float):
    n_plus_d = Phi.shape[1]
    regI = lamb * np.eye(n_plus_d)
    W = np.linalg.solve(Phi.T @ Phi + regI, Phi.T @ Y)  # (n+d) x n
    A = W[:n, :].T
    B = W[n:n+d, :].T
    return A, B, W

def stabilize_A(A: np.ndarray, rho_cap: float = 0.98):
    eigvals = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eigvals)))
    if rho > rho_cap:
        scale = rho_cap / (rho + 1e-12)
        return A * scale, rho, scale
    return A, rho, 1.0

def one_step_train_mse(A: np.ndarray, B: np.ndarray, x_cl: np.ndarray, u_cl: np.ndarray):
    X = x_cl[:-1, :]
    U = u_cl[:-1, :]
    Y = x_cl[1:, :]
    Y_hat = X @ A.T + U @ B.T
    return float(np.mean((Y_hat - Y) ** 2))

def multi_step_rollout_mse(A: np.ndarray, B: np.ndarray, x_val: np.ndarray, u_val: np.ndarray, horizon: int = 10):
    T = x_val.shape[0]
    if T < horizon + 1:
        return float('inf')
    mse_sum, count = 0.0, 0
    for start in range(0, T - horizon - 1):
        xhat = x_val[start, :].copy()
        for t in range(horizon):
            xhat = A @ xhat + B @ u_val[start + t, :]
        err = xhat - x_val[start + horizon, :]
        mse_sum += float(np.mean(err**2))
        count += 1
    return mse_sum / max(count, 1)

def control_energy_prefix(u: np.ndarray):
    # cumulative Σ||u||^2 per step
    e = np.sum(u**2, axis=1)
    return np.cumsum(e)

def path_length_xy_prefix(x_glob: np.ndarray):
    if x_glob is None or x_glob.shape[1] < 2 or x_glob.shape[0] < 2:
        return np.array([np.nan])
    dxy = np.diff(x_glob[:, :2], axis=0)
    seg = np.linalg.norm(dxy, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])

def extract_last_lmpc_trajectory(lmpc):
    """
    Attempts to extract the last stored LMPC lap trajectories (global).
    Fallback: returns None if unavailable.
    """
    try:
        # Many LMPC implementations store the last added trajectory in these containers
        x_last = lmpc.SS[-1] if hasattr(lmpc, "SS") else None
        u_last = lmpc.uSS[-1] if hasattr(lmpc, "uSS") else None
        x_last_glob = lmpc.SS_glob[-1] if hasattr(lmpc, "SS_glob") else None

        # Some variants: store lap-wise lists; try best-effort
        if x_last is None and hasattr(lmpc, "xStored"):
            x_last = lmpc.xStored[-1]
        if u_last is None and hasattr(lmpc, "uStored"):
            u_last = lmpc.uStored[-1]
        if x_last_glob is None and hasattr(lmpc, "xStored_glob"):
            x_last_glob = lmpc.xStored_glob[-1]

        return x_last, u_last, x_last_glob
    except Exception:
        return None, None, None

def extract_lap_times(lmpc, dt: float):
    laps = []
    if not hasattr(lmpc, 'Qfun'):
        return laps
    try:
        for i in range(len(lmpc.Qfun)):
            val = lmpc.Qfun[i]
            if isinstance(val, (list, tuple, np.ndarray)):
                laps.append(float(val[0]) * dt)
            else:
                laps.append(float(val) * dt)
    except Exception:
        pass
    return laps


# ======================= Simulation-aware λ chooser =======================
def choose_lambda_simaware(x_train: np.ndarray, u_train: np.ndarray,
                           x_val: np.ndarray,   u_val: np.ndarray,
                           n: int, d: int,
                           lamb_grid=None, horizon_grid=(5, 10, 15),
                           rho_cap=0.98):
    if lamb_grid is None:
        lamb_grid = np.logspace(-8, -2, 10)

    best = {"mse": float('inf'), "lamb": None, "H": None, "A": None, "B": None}
    Phi_tr, Y_tr = build_dataset(x_train, u_train)

    for H in horizon_grid:
        for lamb in lamb_grid:
            A, B, _ = ridge_AB(Phi_tr, Y_tr, n, d, lamb)
            A, _, _ = stabilize_A(A, rho_cap=rho_cap)
            mse = multi_step_rollout_mse(A, B, x_val, u_val, horizon=H)
            if mse < best["mse"]:
                best.update({"mse": float(mse), "lamb": float(lamb), "H": int(H), "A": A.copy(), "B": B.copy()})
    return best["lamb"], best["H"], best["A"], best["B"]


# ======================= Pipelines =======================
def run_pipeline(map_obj, xS, vt, n, d, N, fit_strategy: str, seed=0):
    """
    Runs: PID -> fit (A,B) -> MPC -> TV-MPC -> LMPC.
    Returns dict with last LMPC lap trajectory + per-step metrics for dashboard playback.
    """
    rng = np.random.default_rng(seed)

    # Fresh params & sims
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map_obj, N)
    simulator     = Simulator(map_obj)
    LMPCsimulator = Simulator(map_obj, multiLap=False, flagLMPC=True)

    # PID (data source #1)
    PIDController = PID(vt)
    xPID, uPID, xPID_glob, _ = simulator.sim(xS, PIDController)

    # ===== Fit (A,B) =====
    if fit_strategy == "baseline":
        lamb_fixed = 1e-7
        A, B, _ = Regression(xPID, uPID, lamb_fixed)
        A, _, _ = stabilize_A(A, rho_cap=0.995)

    elif fit_strategy == "simaware":
        # Warm MPC to get second regime
        lamb_warm = 1e-6
        Phi_pid, Y_pid = build_dataset(xPID, uPID)
        A0, B0, _ = ridge_AB(Phi_pid, Y_pid, n, d, lamb_warm)
        A0, _, _ = stabilize_A(A0, rho_cap=0.995)
        mpcParam.A, mpcParam.B = A0, B0
        mpc_warm = MPC(mpcParam)
        xMPC_warm, uMPC_warm, xMPC_warm_glob, _ = simulator.sim(xS, mpc_warm)

        # Split PID & warm-MPC sequences into train/val and concat
        def split_seq(x, u, frac=0.7):
            T = x.shape[0]
            cut = max(3, int(frac * T))
            return (x[:cut], u[:cut]), (x[cut:], u[cut:])

        (xPID_tr, uPID_tr), (xPID_va, uPID_va) = split_seq(xPID, uPID, frac=0.7)
        (xMPC_tr, uMPC_tr), (xMPC_va, uMPC_va) = split_seq(xMPC_warm, uMPC_warm, frac=0.7)

        x_tr = np.vstack([xPID_tr, xMPC_tr])
        u_tr = np.vstack([uPID_tr, uMPC_tr])
        x_va = np.vstack([xPID_va, xMPC_va])
        u_va = np.vstack([uPID_va, uMPC_va])

        best_lamb, best_H, A, B = choose_lambda_simaware(
            x_tr, u_tr, x_va, u_va, n, d,
            lamb_grid=np.logspace(-8, -2, 10),
            horizon_grid=(5, 10, 15),
            rho_cap=0.98
        )

        # Continual refit on PID + warm MPC with chosen λ
        Phi_all, Y_all = build_dataset(np.vstack([xPID, xMPC_warm]), np.vstack([uPID, uMPC_warm]))
        A, B, _ = ridge_AB(Phi_all, Y_all, n, d, best_lamb)
        A, _, _ = stabilize_A(A, rho_cap=0.98)
    else:
        raise ValueError("fit_strategy must be 'baseline' or 'simaware'")

    # ===== LTI-MPC =====
    mpcParam.A = A
    mpcParam.B = B
    mpc = MPC(mpcParam)
    xMPC, uMPC, xMPC_glob, _ = simulator.sim(xS, mpc)

    # ===== TV-MPC =====
    predictiveModel = PredictiveModel(n, d, map_obj, 1)
    predictiveModel.addTrajectory(xPID, uPID)
    ltvmpcParam.timeVarying = True
    mpc_tv = MPC(ltvmpcParam, predictiveModel)
    xTV, uTV, xTV_glob, _ = simulator.sim(xS, mpc_tv)

    # ===== LMPC =====
    lmpcpredictiveModel = PredictiveModel(n, d, map_obj, 4)
    for _ in range(4):
        lmpcpredictiveModel.addTrajectory(xPID, uPID)
    lmpcParameters.timeVarying = True
    lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for _ in range(4):
        lmpc.addTrajectory(xPID, uPID, xPID_glob)

    last_xLMPC = last_uLMPC = last_xLMPC_glob = None
    for it in range(numSS_it, Laps):
        xLMPC, uLMPC, xLMPC_glob, xS = LMPCsimulator.sim(xS, lmpc)
        lmpc.addTrajectory(xLMPC, uLMPC, xLMPC_glob)
        lmpcpredictiveModel.addTrajectory(xLMPC, uLMPC)
        last_xLMPC, last_uLMPC, last_xLMPC_glob = xLMPC, uLMPC, xLMPC_glob  # keep last lap

    # Build per-step metrics for dashboard playback (use last LMPC lap)
    if last_xLMPC is None or last_xLMPC_glob is None:
        # Fallback to TV-MPC lap to ensure we can animate something
        last_xLMPC, last_uLMPC, last_xLMPC_glob = xTV, uTV, xTV_glob

    metrics = {
        "lap_times": extract_lap_times(lmpc, dt=0.1),
        "energy_prefix": control_energy_prefix(last_uLMPC) if last_uLMPC is not None else np.array([np.nan]),
        "path_prefix": path_length_xy_prefix(last_xLMPC_glob),
        "dt": 0.1,
    }

    return {
        "x_glob": last_xLMPC_glob,   # (T x 2+) global states for animation
        "u": last_uLMPC,             # (T x d) inputs for energy prefix
        "metrics": metrics,
        "label": fit_strategy,
    }


# ======================= Dual Dashboard (two “cars” racing) =======================
def animate_dual_dashboards(map_obj, data_A, data_B, title_A="Baseline", title_B="Sim-Aware"):
    """
    Builds a single figure with 2 synchronized dashboards (left=A, right=B).
    Each shows a live car marker, path trace, and live metrics as text.
    """
    # Prepare trajectories (XY)
    XA = data_A["x_glob"] if data_A["x_glob"] is not None else np.zeros((2, 2))
    XB = data_B["x_glob"] if data_B["x_glob"] is not None else np.zeros((2, 2))
    TA = XA.shape[0]
    TB = XB.shape[0]
    T = min(TA, TB)  # synchronize playback length

    # Metric prefixes
    eA = data_A["metrics"]["energy_prefix"]
    eB = data_B["metrics"]["energy_prefix"]
    sA = data_A["metrics"]["path_prefix"]
    sB = data_B["metrics"]["path_prefix"]
    dt = data_A["metrics"]["dt"]

    # Figure and axes
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], wspace=0.25, hspace=0.25)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    infoA = fig.add_subplot(gs[1, 0])
    infoB = fig.add_subplot(gs[1, 1])

    # Track limits unified for fairness
    all_xy = np.vstack([XA[:, :2], XB[:, :2]])
    xmin, ymin = np.min(all_xy, axis=0) - 1.0
    xmax, ymax = np.max(all_xy, axis=0) + 1.0

    def setup_ax(ax, title):
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

    setup_ax(axA, f"{title_A} — Last LMPC Lap")
    setup_ax(axB, f"{title_B} — Last LMPC Lap")

    # Static full path (light)
    pathA, = axA.plot(XA[:, 0], XA[:, 1], linewidth=1, alpha=0.4)
    pathB, = axB.plot(XB[:, 0], XB[:, 1], linewidth=1, alpha=0.4)

    # Dynamic trace up to k and car markers
    traceA, = axA.plot([], [], linewidth=2)
    carA,   = axA.plot([], [], marker='o', markersize=8)
    traceB, = axB.plot([], [], linewidth=2)
    carB,   = axB.plot([], [], marker='o', markersize=8)

    # Info panels (turn off axes)
    for ax in (infoA, infoB):
        ax.axis('off')

    textA = infoA.text(0.02, 0.5, "", va='center', fontsize=11)
    textB = infoB.text(0.02, 0.5, "", va='center', fontsize=11)

    # Precompute best lap times for footer
    lapsA = data_A["metrics"]["lap_times"]
    lapsB = data_B["metrics"]["lap_times"]
    bestA = np.min(lapsA) if len(lapsA) > 0 else np.nan
    bestB = np.min(lapsB) if len(lapsB) > 0 else np.nan

    # Animation update
    def update(k):
        # Left dashboard
        xkA = XA[:k+1, 0]
        ykA = XA[:k+1, 1]
        traceA.set_data(xkA, ykA)
        carA.set_data([XA[k, 0]], [XA[k, 1]])

        # Right dashboard
        xkB = XB[:k+1, 0]
        ykB = XB[:k+1, 1]
        traceB.set_data(xkB, ykB)
        carB.set_data([XB[k, 0]], [XB[k, 1]])

        # Metrics
        tA = k * dt
        tB = k * dt
        EA = eA[min(k, len(eA)-1)] if eA is not None and len(eA) > 0 else np.nan
        EB = eB[min(k, len(eB)-1)] if eB is not None and len(eB) > 0 else np.nan
        SA = sA[min(k, len(sA)-1)] if sA is not None and len(sA) > 0 else np.nan
        SB = sB[min(k, len(sB)-1)] if sB is not None and len(sB) > 0 else np.nan

        textA.set_text(
            f"Elapsed: {tA:6.2f} s\n"
            f"Step:    {k:6d}\n"
            f"Energy Σ||u||²: {EA:8.2f}\n"
            f"Path length:    {SA:8.2f} m\n"
            f"Best lap:       {bestA:8.2f} s"
        )
        textB.set_text(
            f"Elapsed: {tB:6.2f} s\n"
            f"Step:    {k:6d}\n"
            f"Energy Σ||u||²: {EB:8.2f}\n"
            f"Path length:    {SB:8.2f} m\n"
            f"Best lap:       {bestB:8.2f} s"
        )
        return traceA, carA, traceB, carB, textA, textB

    ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False, repeat=False)
    plt.suptitle("Baseline vs Sim-Aware — Real-Time Dashboards (Two Cars Racing)")
    plt.show()


# ======================= Main =======================
def main():
    # Common setup
    N = 14
    n = 6; d = 2
    dt = 0.1
    x0 = np.array([0.5, 0, 0, 0, 0, 0])
    xS0 = [x0, x0]
    map_obj = Map(0.4)
    vt = 0.8

    # Run both pipelines and capture **last LMPC lap** for playback
    print("\n=== PIPELINE A: BASELINE (fixed λ = 1e-7) ===")
    data_A = run_pipeline(map_obj, xS0, vt, n, d, N, fit_strategy="baseline", seed=0)

    print("\n=== PIPELINE B: SIM-AWARE (multi-step rollout + ρ(A) cap + continual refit) ===")
    data_B = run_pipeline(map_obj, xS0, vt, n, d, N, fit_strategy="simaware", seed=0)

    # Single figure with 2 real-time dashboards (two cars racing)
    animate_dual_dashboards(map_obj, data_A, data_B, title_A="Baseline", title_B="Sim-Aware")


if __name__ == "__main__":
    main()
