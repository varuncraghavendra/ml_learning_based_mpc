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
    if u is None:
        return np.nan
    return float(np.sum(u**2))


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


def compute_lmpc_results(lap_times, x_last, u_last, state_dim=6):
    """
    Compute:
      1) best_lap_time (s)
      2) control_effort_last (sum of squared inputs on last LMPC lap)
      3) ey_rms_last (RMS of lateral error on last LMPC lap)
    Assumes ey is the last state component.
    """
    lap_times = np.array(lap_times, dtype=float)
    if lap_times.size == 0 or x_last is None or u_last is None:
        return {
            "best_lap_time": np.nan,
            "control_effort_last": np.nan,
            "ey_rms_last": np.nan
        }

    best_lap = float(np.min(lap_times))

    control_effort_last = control_energy(u_last)

    # ey is assumed to be last state component (index state_dim - 1)
    ey_index = state_dim - 1
    ey = x_last[:, ey_index]
    ey_rms = float(np.sqrt(np.mean(ey**2)))

    return {
        "best_lap_time": best_lap,
        "control_effort_last": control_effort_last,
        "ey_rms_last": ey_rms
    }


# ======================= Core experiment runner =======================
def run_pipeline(map_obj, xS, vt, n, d, N, use_cv=False, label="baseline", seed=0):
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

    # Report model errors on PID data (optional, still useful)
    results["train_mse_pid"] = one_step_pred_mse(A, B, xPID_cl, uPID_cl)

    # MPC (LTI) – kept but not used in final metrics, useful if you still want to inspect
    mpcParam.A = A
    mpcParam.B = B
    mpc = MPC(mpcParam)
    xMPC_cl, uMPC_cl, xMPC_cl_glob, _ = simulator.sim(xS, mpc)
    results["xMPC"], results["uMPC"], results["xMPC_glob"] = xMPC_cl, uMPC_cl, xMPC_cl_glob

    # TV-MPC – also kept but not used in final metrics
    predictiveModel = PredictiveModel(n, d, map_obj, 1)
    predictiveModel.addTrajectory(xPID_cl, uPID_cl)
    ltvmpcParam.timeVarying = True
    mpc_tv = MPC(ltvmpcParam, predictiveModel)
    xTVMPC_cl, uTVMPC_cl, xTVMPC_cl_glob, _ = simulator.sim(xS, mpc_tv)
    results["xTVMPC"], results["uTVMPC"], results["xTVMPC_glob"] = xTVMPC_cl, uTVMPC_cl, xTVMPC_cl_glob

    # LMPC
    lmpcpredictiveModel = PredictiveModel(n, d, map_obj, 4)
    for _ in range(4):
        lmpcpredictiveModel.addTrajectory(xPID_cl, uPID_cl)
    lmpcParameters.timeVarying = True
    lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for _ in range(4):
        lmpc.addTrajectory(xPID_cl, uPID_cl, xPID_cl_glob)

    xLMPC_last, uLMPC_last = None, None
    for it in range(numSS_it, Laps):
        xLMPC, uLMPC, xLMPC_glob, xS = LMPCsimulator.sim(xS, lmpc)
        lmpc.addTrajectory(xLMPC, uLMPC, xLMPC_glob)
        lmpcpredictiveModel.addTrajectory(xLMPC, uLMPC)

        # keep last LMPC lap trajectory
        xLMPC_last = xLMPC
        uLMPC_last = uLMPC

    results["lmpc"] = lmpc
    results["lmpc_lap_times"] = extract_lap_times(lmpc, dt=0.1)
    results["xLMPC_last"] = xLMPC_last
    results["uLMPC_last"] = uLMPC_last

    return results


def main():
    # Common setup
    N = 14
    n = 6
    d = 2
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

    # -------- Final LMPC metrics (what you asked for) --------
    baseline_res = compute_lmpc_results(
        baseline["lmpc_lap_times"],
        baseline["xLMPC_last"],
        baseline["uLMPC_last"],
        state_dim=n
    )

    cv_res = compute_lmpc_results(
        cvrun["lmpc_lap_times"],
        cvrun["xLMPC_last"],
        cvrun["uLMPC_last"],
        state_dim=n
    )

    print("\n===================== LMPC SUMMARY METRICS =====================")
    print("Baseline (fixed λ):")
    print(f"  best_lap_time [s]        = {baseline_res['best_lap_time']:.3f}")
    print(f"  control_effort_last      = {baseline_res['control_effort_last']:.3f}")
    print(f"  ey_rms_last              = {baseline_res['ey_rms_last']:.5f}")

    print("\nCV-selected λ:")
    print(f"  best_lap_time [s]        = {cv_res['best_lap_time']:.3f}")
    print(f"  control_effort_last      = {cv_res['control_effort_last']:.3f}")
    print(f"  ey_rms_last              = {cv_res['ey_rms_last']:.5f}")
    print("================================================================\n")

    # -------- One graph: lap time vs iteration --------
    base_laps = baseline["lmpc_lap_times"]
    cv_laps   = cvrun["lmpc_lap_times"]

    plt.figure(figsize=(8, 5))
    if len(base_laps) > 0:
        plt.plot(range(1, len(base_laps) + 1), base_laps, 'o-', label='Fixed-λ LMPC')
    if len(cv_laps) > 0:
        plt.plot(range(1, len(cv_laps) + 1), cv_laps, 's-', label='CV-λ LMPC')

    plt.xlabel('LMPC Iteration (Lap #)')
    plt.ylabel('Lap Time [s]')
    plt.title('LMPC Lap Time vs Iteration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
