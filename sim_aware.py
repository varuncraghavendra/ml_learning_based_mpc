# ----------------------------------------------------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or research purposes provided that you provide clear attribution to UC Berkeley,
# including references to the LMPC papers by Rosolia & Borrelli.
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('fnc/simulator')
sys.path.append('fnc/controller')
sys.path.append('fnc')

import numpy as np
import matplotlib.pyplot as plt

from plot import plotTrajectory, plotClosedLoopLMPC, animation_xy
from initControllerParameters import initMPCParams, initLMPCParams
from PredictiveControllers import MPC, LMPC
from PredictiveModel import PredictiveModel
from Utilities import Regression, PID
from SysModel import Simulator
from Track import Map


# ======================= Dataset & Linear Ridge (A,B) =======================
def build_dataset(x_cl: np.ndarray, u_cl: np.ndarray):
    """Phi = [x_k, u_k], Y = x_{k+1}."""
    assert x_cl.shape[0] == u_cl.shape[0], "x_cl and u_cl must have same length"
    if x_cl.shape[0] < 3:
        raise ValueError("Not enough samples to build a dataset (need >= 3).")
    X = x_cl[:-1, :]
    U = u_cl[:-1, :]
    Y = x_cl[1:, :]
    Phi = np.hstack([X, U])
    return Phi, Y

def ridge_AB(Phi: np.ndarray, Y: np.ndarray, n: int, d: int, lamb: float):
    """W = (Phi^T Phi + λ I)^(-1) Phi^T Y  =>  split to A,B."""
    n_plus_d = Phi.shape[1]
    regI = lamb * np.eye(n_plus_d)
    W = np.linalg.solve(Phi.T @ Phi + regI, Phi.T @ Y)  # (n+d) x n
    A = W[:n, :].T
    B = W[n:n+d, :].T
    return A, B, W

def stabilize_A(A: np.ndarray, rho_cap: float = 0.98):
    """Project A to have spectral radius <= rho_cap (gentle scaling if needed)."""
    eigvals = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eigvals)))
    if rho > rho_cap:
        scale = rho_cap / (rho + 1e-12)
        return A * scale, rho, scale
    return A, rho, 1.0

def multi_step_rollout_mse(A: np.ndarray, B: np.ndarray,
                           x_val: np.ndarray, u_val: np.ndarray,
                           horizon: int = 10):
    """
    Rollout error using *iterated* predictions:
    x̂_{k+1} = A x̂_k + B u_k, starting from the true x_0 in val segment.
    Computes MSE across a sliding window with given horizon.
    """
    # Need at least horizon+1 samples
    T = x_val.shape[0]
    if T < horizon + 1:
        return float('inf')

    mse_sum = 0.0
    count = 0
    for start in range(0, T - horizon - 1):
        xhat = x_val[start, :].copy()
        for t in range(horizon):
            # predict one step
            xhat = A @ xhat + B @ u_val[start + t, :]
        # compare with ground-truth at k+H
        err = xhat - x_val[start + horizon, :]
        mse_sum += float(np.mean(err**2))
        count += 1
    return mse_sum / max(count, 1)

def one_step_train_mse(A: np.ndarray, B: np.ndarray, x_cl: np.ndarray, u_cl: np.ndarray):
    """One-step (train) MSE, still useful as a sanity check."""
    X = x_cl[:-1, :]
    U = u_cl[:-1, :]
    Y = x_cl[1:, :]
    Y_hat = X @ A.T + U @ B.T
    return float(np.mean((Y_hat - Y) ** 2))

def control_energy(u: np.ndarray) -> float:
    return float(np.sum(u**2))

def trajectory_length_xy(x_glob: np.ndarray) -> float:
    if x_glob is None or x_glob.shape[1] < 2 or x_glob.shape[0] < 2:
        return float('nan')
    dxy = np.diff(x_glob[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(dxy, axis=1)))

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

# ======================= Simulation-aware model selection =======================
def choose_lambda_simaware(x_train: np.ndarray, u_train: np.ndarray,
                           x_val: np.ndarray,   u_val: np.ndarray,
                           n: int, d: int,
                           lamb_grid=None, horizon_grid=(5, 10, 15),
                           rho_cap=0.98):
    """
    Pick λ by minimizing multi-step rollout MSE on held-out data.
    Also applies a gentle stability projection A <- αA if ρ(A) > rho_cap.
    Returns (best_lambda, best_horizon, A*, B*, report_dict)
    """
    if lamb_grid is None:
        lamb_grid = np.logspace(-8, -2, 10)

    best = {
        "mse": float('inf'),
        "lamb": None,
        "H": None,
        "A": None,
        "B": None,
        "rho": None,
        "scale": None,
    }
    report = []

    Phi_tr, Y_tr = build_dataset(x_train, u_train)

    for H in horizon_grid:
        for lamb in lamb_grid:
            # Fit on train
            A, B, _ = ridge_AB(Phi_tr, Y_tr, n, d, lamb)
            # Stability projection
            A_stab, rho, scale = stabilize_A(A, rho_cap=rho_cap)
            if scale != 1.0:
                A = A_stab  # apply

            # Evaluate rollout MSE on validation
            mse = multi_step_rollout_mse(A, B, x_val, u_val, horizon=H)
            report.append((float(lamb), int(H), float(mse), float(rho), float(scale)))

            if mse < best["mse"]:
                best.update({
                    "mse": float(mse),
                    "lamb": float(lamb),
                    "H": int(H),
                    "A": A.copy(),
                    "B": B.copy(),
                    "rho": float(rho),
                    "scale": float(scale),
                })

    return best["lamb"], best["H"], best["A"], best["B"], report


# ======================= Core experiment runner =======================
def run_pipeline(map_obj, xS, vt, n, d, N,
                 fit_strategy: str,
                 seed=0,
                 simaware_conf=None):
    """
    Runs: PID -> (fit A,B) -> MPC -> TV-MPC -> LMPC
    fit_strategy: "baseline" or "simaware"
    simaware_conf: dict with keys: val_split, lamb_grid, horizon_grid, rho_cap
    """
    results = {"label": fit_strategy}
    rng = np.random.default_rng(seed)

    # Fresh params & sims for fairness
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map_obj, N)
    simulator     = Simulator(map_obj)
    LMPCsimulator = Simulator(map_obj, multiLap=False, flagLMPC=True)

    # PID (data source #1)
    PIDController = PID(vt)
    xPID, uPID, xPID_glob, _ = simulator.sim(xS, PIDController)
    results["xPID"], results["uPID"], results["xPID_glob"] = xPID, uPID, xPID_glob

    # ===== Fit (A,B) =====
    if fit_strategy == "baseline":
        # Simple fixed-λ ridge on PID only
        lamb_fixed = 1e-7
        A, B, _ = Regression(xPID, uPID, lamb_fixed)
        # small stability cap for safety (no harm if already stable)
        A, rho, scale = stabilize_A(A, rho_cap=0.995)
        results["lambda"] = lamb_fixed
        results["rho_A"] = rho
        results["A_scale"] = scale
        results["train_mse_pid"] = one_step_train_mse(A, B, xPID, uPID)

    elif fit_strategy == "simaware":
        # First, get a quick LTI-MPC using a decent initial model (fixed λ)
        lamb_warm = 1e-6
        Phi_pid, Y_pid = build_dataset(xPID, uPID)
        A0, B0, _ = ridge_AB(Phi_pid, Y_pid, n, d, lamb_warm)
        A0, _, _ = stabilize_A(A0, rho_cap=0.995)
        mpcParam.A, mpcParam.B = A0, B0
        mpc_warm = MPC(mpcParam)
        xMPC, uMPC, xMPC_glob, _ = simulator.sim(xS, mpc_warm)
        results["xMPC_warm"], results["uMPC_warm"], results["xMPC_warm_glob"] = xMPC, uMPC, xMPC_glob

        # Build train/val splits from PID(+ warm MPC) to better match controller regime
        # Use the first 70% as train, last 30% as validation (on each sequence), then concat.
        def split_seq(x, u, frac=0.7):
            T = x.shape[0]
            cut = max(3, int(frac * T))
            return (x[:cut], u[:cut]), (x[cut:], u[cut:])

        (xPID_tr, uPID_tr), (xPID_va, uPID_va) = split_seq(xPID, uPID, frac=0.7)
        (xMPC_tr, uMPC_tr), (xMPC_va, uMPC_va) = split_seq(xMPC, uMPC, frac=0.7)

        x_tr = np.vstack([xPID_tr, xMPC_tr])
        u_tr = np.vstack([uPID_tr, uMPC_tr])
        x_va = np.vstack([xPID_va, xMPC_va])
        u_va = np.vstack([uPID_va, uMPC_va])

        # Simulation-aware λ selection (minimize multi-step rollout error on validation)
        conf = dict(val_split=0.3, lamb_grid=None, horizon_grid=(5, 10, 15), rho_cap=0.98)
        if simaware_conf is not None:
            conf.update(simaware_conf)

        best_lamb, best_H, A, B, report = choose_lambda_simaware(
            x_tr, u_tr, x_va, u_va, n, d,
            lamb_grid=conf["lamb_grid"],
            horizon_grid=conf["horizon_grid"],
            rho_cap=conf["rho_cap"]
        )
        results["lambda"] = best_lamb
        results["H_selected"] = best_H
        results["simaware_report"] = report
        results["train_mse_pid"] = one_step_train_mse(A, B, xPID, uPID)

        # Continual update on all PID+MPC (train FINAL model used downstream)
        x_all = np.vstack([xPID, xMPC])
        u_all = np.vstack([uPID, uMPC])
        Phi_all, Y_all = build_dataset(x_all, u_all)
        A_all, B_all, _ = ridge_AB(Phi_all, Y_all, n, d, best_lamb)
        A_all, rho, scale = stabilize_A(A_all, rho_cap=0.98)
        A, B = A_all, B_all
        results["rho_A"] = rho
        results["A_scale"] = scale
    else:
        raise ValueError("fit_strategy must be 'baseline' or 'simaware'")

    # ===== LTI-MPC =====
    mpcParam.A = A
    mpcParam.B = B
    mpc = MPC(mpcParam)
    xMPC, uMPC, xMPC_glob, _ = simulator.sim(xS, mpc)
    results["xMPC"], results["uMPC"], results["xMPC_glob"] = xMPC, uMPC, xMPC_glob
    results["uMPC_energy"] = control_energy(uMPC)
    results["MPC_path_length"] = trajectory_length_xy(xMPC_glob)

    # ===== TV-MPC =====
    predictiveModel = PredictiveModel(n, d, map_obj, 1)
    predictiveModel.addTrajectory(xPID, uPID)
    ltvmpcParam.timeVarying = True
    mpc_tv = MPC(ltvmpcParam, predictiveModel)
    xTV, uTV, xTV_glob, _ = simulator.sim(xS, mpc_tv)
    results["xTVMPC"], results["uTVMPC"], results["xTVMPC_glob"] = xTV, uTV, xTV_glob
    results["uTVMPC_energy"] = control_energy(uTV)
    results["TVMPC_path_length"] = trajectory_length_xy(xTV_glob)

    # ===== LMPC =====
    lmpcpredictiveModel = PredictiveModel(n, d, map_obj, 4)
    for _ in range(4):
        lmpcpredictiveModel.addTrajectory(xPID, uPID)
    lmpcParameters.timeVarying = True
    lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for _ in range(4):
        lmpc.addTrajectory(xPID, uPID, xPID_glob)

    for it in range(numSS_it, Laps):
        xLMPC, uLMPC, xLMPC_glob, xS = LMPCsimulator.sim(xS, lmpc)
        lmpc.addTrajectory(xLMPC, uLMPC, xLMPC_glob)
        lmpcpredictiveModel.addTrajectory(xLMPC, uLMPC)

    results["lmpc"] = lmpc
    results["lmpc_lap_times"] = extract_lap_times(lmpc, dt=0.1)
    try:
        results["LMPC_last_path_length"] = trajectory_length_xy(lmpc.SS_glob[-1])
    except Exception:
        results["LMPC_last_path_length"] = float('nan')

    return results


def fmt(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return "-"
        return ", ".join(f"{v:.2f}" for v in x[:6]) + (" ..." if len(x) > 6 else "")
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def print_benchmark_table(base, simaware):
    print("\n================ BENCHMARK: BASELINE vs SIM-AWARE =================")
    print(f"{'Metric':35s} | {'Baseline':20s} | {'Sim-aware':20s}")
    print("-"*83)
    print(f"{'Selected λ':35s} | {fmt(base.get('lambda')):20s} | {fmt(simaware.get('lambda')):20s}")
    print(f"{'A spectral radius ρ(A)':35s} | {fmt(base.get('rho_A')):20s} | {fmt(simaware.get('rho_A')):20s}")
    print(f"{'A scaling applied':35s} | {fmt(base.get('A_scale')):20s} | {fmt(simaware.get('A_scale')):20s}")
    print(f"{'One-step MSE (PID train)':35s} | {fmt(base.get('train_mse_pid')):20s} | {fmt(simaware.get('train_mse_pid')):20s}")
    print(f"{'LTI-MPC control energy Σ||u||^2':35s} | {fmt(base.get('uMPC_energy')):20s} | {fmt(simaware.get('uMPC_energy')):20s}")
    print(f"{'TV-MPC control energy Σ||u||^2':35s} | {fmt(base.get('uTVMPC_energy')):20s} | {fmt(simaware.get('uTVMPC_energy')):20s}")
    print(f"{'MPC path length (m)':35s} | {fmt(base.get('MPC_path_length')):20s} | {fmt(simaware.get('MPC_path_length')):20s}")
    print(f"{'TV-MPC path length (m)':35s} | {fmt(base.get('TVMPC_path_length')):20s} | {fmt(simaware.get('TVMPC_path_length')):20s}")
    print(f"{'LMPC last path length (m)':35s} | {fmt(base.get('LMPC_last_path_length')):20s} | {fmt(simaware.get('LMPC_last_path_length')):20s}")
    print(f"{'LMPC lap times (s)':35s} | {fmt(base.get('lmpc_lap_times')):20s} | {fmt(simaware.get('lmpc_lap_times')):20s}")
    if base.get('lmpc_lap_times') and simaware.get('lmpc_lap_times'):
        print(f"{'Best lap (s)':35s} | {fmt(min(base['lmpc_lap_times'])):20s} | {fmt(min(simaware['lmpc_lap_times'])):20s}")
    print("====================================================================\n")


def main():
    # Common setup
    N = 14
    n = 6; d = 2
    dt = 0.1
    x0 = np.array([0.5, 0, 0, 0, 0, 0])
    xS0 = [x0, x0]
    map_obj = Map(0.4)
    vt = 0.8

    # -------- Pipeline 1: Baseline fixed-λ --------
    print("\n=== PIPELINE 1: BASELINE (fixed λ = 1e-7) ===")
    baseline = run_pipeline(map_obj, xS0, vt, n, d, N, fit_strategy="baseline", seed=0)

    # -------- Pipeline 2: Simulation-aware λ (multi-step rollout + stability cap + continual fit) --------
    print("\n=== PIPELINE 2: SIM-AWARE (multi-step rollout + ρ(A) cap + continual update) ===")
    simaware = run_pipeline(
        map_obj, xS0, vt, n, d, N,
        fit_strategy="simaware",
        seed=0,
        simaware_conf=dict(
            lamb_grid=np.logspace(-8, -2, 10),
            horizon_grid=(5, 10, 15),
            rho_cap=0.98
        )
    )

    # -------- Benchmark report --------
    print_benchmark_table(baseline, simaware)

    # -------- Optional visuals --------
    try:
        print("Plotting quick overlays...")
        plotTrajectory(map_obj, baseline["xMPC"], baseline["xMPC_glob"], baseline["uMPC"], 'MPC (baseline)')
        plotTrajectory(map_obj, simaware["xMPC"], simaware["xMPC_glob"], simaware["uMPC"], 'MPC (sim-aware)')
        plotClosedLoopLMPC(baseline["lmpc"], map_obj)
        plotClosedLoopLMPC(simaware["lmpc"], map_obj)
        animation_xy(map_obj, simaware["lmpc"], max(0, len(simaware["lmpc_lap_times"]) - 1))
        plt.show()
    except Exception as e:
        print(f"[Plot] Skipped plotting due to error: {e}")


if __name__ == "__main__":
    main()
