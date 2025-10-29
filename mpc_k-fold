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


# ======================= ML helpers: ridge with K-fold CV for (A,B) =======================
def _build_dataset(x_cl: np.ndarray, u_cl: np.ndarray):
    """Phi = [x_k, u_k], Y = x_{k+1}."""
    assert x_cl.shape[0] == u_cl.shape[0], "x_cl and u_cl must have same length"
    if x_cl.shape[0] < 3:
        raise ValueError("Not enough samples to build a dataset (need >= 3).")
    X = x_cl[:-1, :]
    U = u_cl[:-1, :]
    Y = x_cl[1:, :]
    Phi = np.hstack([X, U])
    return Phi, Y


def _ridge_fit(Phi: np.ndarray, Y: np.ndarray, lamb: float) -> np.ndarray:
    """W = (Phi^T Phi + λ I)^(-1) Phi^T Y"""
    n_plus_d = Phi.shape[1]
    regI = lamb * np.eye(n_plus_d)
    return np.linalg.solve(Phi.T @ Phi + regI, Phi.T @ Y)


def _AB_from_W(W: np.ndarray, n: int, d: int):
    A = W[:n, :].T
    B = W[n:n + d, :].T
    return A, B


def _kfold_indices(Tm1: int, K: int = 5, seed: int = 0):
    K_eff = max(2, min(K, Tm1))
    rng = np.random.default_rng(seed)
    idx = np.arange(Tm1)
    rng.shuffle(idx)
    return np.array_split(idx, K_eff)


def _cv_mse_for_lambda(Phi: np.ndarray, Y: np.ndarray, lamb: float, K: int = 5, seed: int = 0) -> float:
    folds = _kfold_indices(Phi.shape[0], K=K, seed=seed)
    errs = []
    for k in range(len(folds)):
        val_idx = folds[k]
        tr_idx  = np.concatenate([folds[i] for i in range(len(folds)) if i != k]) if len(folds) > 1 else val_idx
        W = _ridge_fit(Phi[tr_idx], Y[tr_idx], lamb)
        Y_hat = Phi[val_idx] @ W
        errs.append(float(np.mean((Y_hat - Y[val_idx])**2)))
    return float(np.mean(errs)) if errs else float('inf')


def select_lambda_cv(x_cl: np.ndarray, u_cl: np.ndarray, lamb_grid=None, K: int = 5, seed: int = 0):
    """Return λ*, A(λ*), B(λ*), and CV table."""
    if lamb_grid is None:
        lamb_grid = np.logspace(-8, 2, 15)
    Phi, Y = _build_dataset(x_cl, u_cl)
    n, d = x_cl.shape[1], u_cl.shape[1]

    best_lambda, best_mse = None, float('inf')
    cv_table = []
    for lamb in lamb_grid:
        mse = _cv_mse_for_lambda(Phi, Y, lamb, K=K, seed=seed)
        cv_table.append((float(lamb), float(mse)))
        if mse < best_mse:
            best_lambda, best_mse = float(lamb), float(mse)

    W = _ridge_fit(Phi, Y, best_lambda)
    A, B = _AB_from_W(W, n, d)
    return best_lambda, A, B, cv_table


# ======================= Metrics helpers =======================
def control_energy(u: np.ndarray) -> float:
    """Sum of squared inputs."""
    return float(np.sum(u**2))

def trajectory_length_xy(x_glob: np.ndarray) -> float:
    """Approx arc-length using columns 0,1 if present."""
    if x_glob is None or x_glob.shape[1] < 2 or x_glob.shape[0] < 2:
        return float('nan')
    dx = np.diff(x_glob[:, 0])
    dy = np.diff(x_glob[:, 1])
    return float(np.sum(np.sqrt(dx*dx + dy*dy)))

def one_step_pred_mse(A: np.ndarray, B: np.ndarray, x_cl: np.ndarray, u_cl: np.ndarray) -> float:
    """Evaluate model on the dataset it came from (train MSE)."""
    X = x_cl[:-1, :]
    U = u_cl[:-1, :]
    Y = x_cl[1:, :]
    Y_hat = X @ A.T + U @ B.T
    return float(np.mean((Y_hat - Y)**2))

def extract_lap_times(lmpc, dt: float):
    """Robust lap-time extraction from lmpc.Qfun."""
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


# ======================= Core experiment runner =======================
def run_pipeline(map_obj, xS, vt, n, d, N, base_AB=None, use_cv=False, label="baseline", seed=0):
    """
    Runs: PID -> (fit A,B) -> MPC -> TV-MPC -> LMPC
    Returns dict of trajectories, controllers, and metrics.
    """
    results = {"label": label}

    # Initialize controller parameters & simulators fresh for fairness
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map_obj, N)
    simulator     = Simulator(map_obj)
    LMPCsimulator = Simulator(map_obj, multiLap=False, flagLMPC=True)

    # PID
    PIDController = PID(vt)
    xPID_cl, uPID_cl, xPID_cl_glob, _ = simulator.sim(xS, PIDController)
    results["xPID"], results["uPID"], results["xPID_glob"] = xPID_cl, uPID_cl, xPID_cl_glob

    # Fit (A,B)
    if use_cv:
        best_lamb, A, B, cv_table = select_lambda_cv(xPID_cl, uPID_cl, K=5, seed=seed)
        results["lambda"] = best_lamb
        results["cv_table"] = cv_table
    else:
        lamb_fixed = 1e-7
        A, B, _ = Regression(xPID_cl, uPID_cl, lamb_fixed)
        results["lambda"] = lamb_fixed
        results["cv_table"] = None

    results["A"], results["B"] = A, B

    # Report model errors on PID data
    results["train_mse_pid"] = one_step_pred_mse(A, B, xPID_cl, uPID_cl)

    # MPC (LTI)
    mpcParam.A = A
    mpcParam.B = B
    mpc = MPC(mpcParam)
    xMPC_cl, uMPC_cl, xMPC_cl_glob, _ = simulator.sim(xS, mpc)
    results["xMPC"], results["uMPC"], results["xMPC_glob"] = xMPC_cl, uMPC_cl, xMPC_cl_glob
    results["uMPC_energy"] = control_energy(uMPC_cl)
    results["MPC_path_length"] = trajectory_length_xy(xMPC_cl_glob)

    # TV-MPC
    predictiveModel = PredictiveModel(n, d, map_obj, 1)
    predictiveModel.addTrajectory(xPID_cl, uPID_cl)
    ltvmpcParam.timeVarying = True
    mpc_tv = MPC(ltvmpcParam, predictiveModel)
    xTVMPC_cl, uTVMPC_cl, xTVMPC_cl_glob, _ = simulator.sim(xS, mpc_tv)
    results["xTVMPC"], results["uTVMPC"], results["xTVMPC_glob"] = xTVMPC_cl, uTVMPC_cl, xTVMPC_cl_glob
    results["uTVMPC_energy"] = control_energy(uTVMPC_cl)
    results["TVMPC_path_length"] = trajectory_length_xy(xTVMPC_cl_glob)

    # LMPC
    lmpcpredictiveModel = PredictiveModel(n, d, map_obj, 4)
    for _ in range(4):
        lmpcpredictiveModel.addTrajectory(xPID_cl, uPID_cl)
    lmpcParameters.timeVarying = True
    lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for _ in range(4):
        lmpc.addTrajectory(xPID_cl, uPID_cl, xPID_cl_glob)

    for it in range(numSS_it, Laps):
        xLMPC, uLMPC, xLMPC_glob, xS = LMPCsimulator.sim(xS, lmpc)
        lmpc.addTrajectory(xLMPC, uLMPC, xLMPC_glob)
        lmpcpredictiveModel.addTrajectory(xLMPC, uLMPC)

    results["lmpc"] = lmpc
    results["lmpc_lap_times"] = extract_lap_times(lmpc, dt=0.1)
    # Also store final trajectory to estimate distance
    try:
        results["LMPC_last_path_length"] = trajectory_length_xy(lmpc.SS_glob[-1])  # if exists
    except Exception:
        results["LMPC_last_path_length"] = float('nan')

    return results


def print_benchmark_table(baseline, cv):
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

    print("\n===================== BENCHMARK: BASELINE vs CV-λ =====================")
    print(f"{'Metric':35s} | {'Baseline':20s} | {'CV-λ':20s}")
    print("-"*83)

    # Prediction metrics on PID data
    print(f"{'One-step MSE (PID data)':35s} | {fmt(baseline['train_mse_pid']):20s} | {fmt(cv['train_mse_pid']):20s}")

    # Regularization strength
    print(f"{'Selected λ':35s} | {fmt(baseline['lambda']):20s} | {fmt(cv['lambda']):20s}")

    # Control energies
    print(f"{'LTI-MPC control energy Σ||u||^2':35s} | {fmt(baseline['uMPC_energy']):20s} | {fmt(cv['uMPC_energy']):20s}")
    print(f"{'TV-MPC control energy Σ||u||^2':35s} | {fmt(baseline['uTVMPC_energy']):20s} | {fmt(cv['uTVMPC_energy']):20s}")

    # Path lengths
    print(f"{'MPC path length (m)':35s} | {fmt(baseline['MPC_path_length']):20s} | {fmt(cv['MPC_path_length']):20s}")
    print(f"{'TV-MPC path length (m)':35s} | {fmt(baseline['TVMPC_path_length']):20s} | {fmt(cv['TVMPC_path_length']):20s}")
    print(f"{'LMPC last path length (m)':35s} | {fmt(baseline['LMPC_last_path_length']):20s} | {fmt(cv['LMPC_last_path_length']):20s}")

    # Lap times
    base_laps = baseline['lmpc_lap_times']
    cv_laps   = cv['lmpc_lap_times']
    print(f"{'LMPC lap times (s)':35s} | {fmt(base_laps):20s} | {fmt(cv_laps):20s}")
    if base_laps and cv_laps:
        # Compare last lap if available
        print(f"{'Best lap (s)':35s} | {fmt(min(base_laps)):20s} | {fmt(min(cv_laps)):20s}")

    print("========================================================================\n")


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
    baseline = run_pipeline(map_obj, xS0, vt, n, d, N, use_cv=False, label="baseline", seed=0)

    # -------- Pipeline 2: ML-CV-selected λ --------
    print("\n=== PIPELINE 2: ML (CV-selected λ) ===")
    cvrun = run_pipeline(map_obj, xS0, vt, n, d, N, use_cv=True, label="cv", seed=0)

    # -------- Benchmark report --------
    print_benchmark_table(baseline, cvrun)

    # -------- Optional: quick visualizations (comment out if running headless) --------
    try:
        print("Plotting a quick overlay for MPC trajectories...")
        plotTrajectory(map_obj, baseline["xMPC"], baseline["xMPC_glob"], baseline["uMPC"], 'MPC (baseline)')
        plotTrajectory(map_obj, cvrun["xMPC"], cvrun["xMPC_glob"], cvrun["uMPC"], 'MPC (CV-λ)')
        plotClosedLoopLMPC(baseline["lmpc"], map_obj)
        plotClosedLoopLMPC(cvrun["lmpc"], map_obj)
        animation_xy(map_obj, cvrun["lmpc"], max(0, len(cvrun["lmpc_lap_times"]) - 1))
        plt.show()
    except Exception as e:
        print(f"[Plot] Skipped plotting due to error: {e}")


if __name__ == "__main__":
    main()
