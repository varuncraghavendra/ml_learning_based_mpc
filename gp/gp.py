# ----------------------------------------------------------------------------------------------------------------------
# STANDARD LMPC vs GAUSSIAN PROCESS LMPC
#
# For BOTH Standard and GP LMPC we report:
#   1. best_lap_time (s)
#   2. control_effort_last (sum of squared inputs on last LMPC lap)
#   3. ey_rms_last (RMS lateral error on last LMPC lap; ey = last state component)
#   4. One graph: Lap time vs iteration (Standard vs GP)
# ----------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append('fnc/simulator')
sys.path.append('fnc/controller')
sys.path.append('fnc')

import os
import numpy as np
import matplotlib.pyplot as plt

from initControllerParameters import initMPCParams, initLMPCParams
from PredictiveControllers import MPC, LMPC
from PredictiveModel import PredictiveModel
from Utilities import Regression, PID
from SysModel import Simulator
from Track import Map

from GaussianProcessDynamics import GPDynamicsPredictor


# ============================ Helpers ============================

def control_energy(u: np.ndarray) -> float:
    """Sum of squared inputs for a lap."""
    if u is None:
        return np.nan
    return float(np.sum(u ** 2))


def compute_lmpc_results(lap_times, x_last, u_last, state_dim=6):
    """
    Returns dict with:
      - best_lap_time
      - control_effort_last
      - ey_rms_last (ey = last state component)
    """
    lap_times = np.array(lap_times, dtype=float)

    if lap_times.size == 0 or x_last is None or u_last is None:
        return {
            "best_lap_time": np.nan,
            "control_effort_last": np.nan,
            "ey_rms_last": np.nan,
        }

    best_lap_time = float(np.min(lap_times))
    control_effort_last = control_energy(u_last)

    ey_index = state_dim - 1  # ey is last state [vx, vy, wz, epsi, s, ey]
    ey = x_last[:, ey_index]
    ey_rms_last = float(np.sqrt(np.mean(ey ** 2)))

    return {
        "best_lap_time": best_lap_time,
        "control_effort_last": control_effort_last,
        "ey_rms_last": ey_rms_last,
    }


# ============================ Main ============================

def main():
    print("\n" + "=" * 100)
    print(" " * 28 + "STANDARD LMPC vs GAUSSIAN PROCESS LMPC")
    print("=" * 100 + "\n")

    # Common configuration
    N = 14
    n = 6
    d = 2
    dt = 0.1

    x0 = np.array([0.5, 0, 0, 0, 0, 0])
    xS0 = [x0, x0]
    vt = 0.8
    map_obj = Map(0.4)

    print(f"Track length:       {map_obj.TrackLength:.2f} m")
    print(f"Target velocity:    {vt} m/s")
    print(f"Horizon length N:   {N}")
    print(f"Time step dt:       {dt} s\n")

    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map_obj, N)

    simulator = Simulator(map_obj)
    LMPCsim_std = Simulator(map_obj, multiLap=False, flagLMPC=True)
    LMPCsim_gp = Simulator(map_obj, multiLap=False, flagLMPC=True)

    os.makedirs("models", exist_ok=True)

    # Init GP (fast configuration)
    gp_predictor = GPDynamicsPredictor(
        state_dim=n,
        input_dim=d,
        noise_level=0.05,
        max_train_points=2000,
        n_restarts=1,
        random_seed=0,
    )

    lap_times = {
        "PID": None,
        "MPC": None,
        "TV-MPC": None,
        "LMPC_Standard": [],
        "LMPC_GP": [],
    }

    # ------------------ Phase 1: PID ------------------
    print("=" * 100)
    print("PHASE 1: PID CONTROLLER")
    print("=" * 100)

    PIDController = PID(vt)
    xPID, uPID, xPID_glob, _ = simulator.sim(xS0, PIDController)
    lap_times["PID"] = xPID.shape[0] * dt
    print(f"✓ PID lap time: {lap_times['PID']:.2f} s\n")

    gp_predictor.add_trajectory(xPID, uPID)

    # ------------------ Phase 2: MPC ------------------
    print("=" * 100)
    print("PHASE 2: MPC (LTI model from PID data)")
    print("=" * 100)

    A, B, _ = Regression(xPID, uPID, 1e-7)
    mpcParam.A = A
    mpcParam.B = B

    mpc = MPC(mpcParam)
    xMPC, uMPC, xMPC_glob, _ = simulator.sim(xS0, mpc)
    lap_times["MPC"] = xMPC.shape[0] * dt
    print(f"✓ MPC lap time: {lap_times['MPC']:.2f} s\n")

    gp_predictor.add_trajectory(xMPC, uMPC)

    # ------------------ Phase 3: TV-MPC ------------------
    print("=" * 100)
    print("PHASE 3: TIME-VARYING MPC")
    print("=" * 100)

    predModel_tv = PredictiveModel(n, d, map_obj, 1)
    predModel_tv.addTrajectory(xPID, uPID)
    ltvmpcParam.timeVarying = True

    mpc_tv = MPC(ltvmpcParam, predModel_tv)
    xTV, uTV, xTV_glob, _ = simulator.sim(xS0, mpc_tv)
    lap_times["TV-MPC"] = xTV.shape[0] * dt
    print(f"✓ TV-MPC lap time: {lap_times['TV-MPC']:.2f} s\n")

    gp_predictor.add_trajectory(xTV, uTV)

    # ------------------ Train GP once on initial data ------------------
    print("=" * 100)
    print("TRAINING GAUSSIAN PROCESS ON INITIAL DATA")
    print("=" * 100 + "\n")

    gp_predictor.train(verbose=True)
    gp_predictor.save_model("models/gp_initial.pkl")
    print("")

    # ------------------ Phase 4: Standard LMPC ------------------
    print("=" * 100)
    print("PHASE 4: STANDARD LMPC")
    print("=" * 100 + "\n")

    predModel_std = PredictiveModel(n, d, map_obj, 4)
    for _ in range(4):
        predModel_std.addTrajectory(xPID, uPID)

    lmpcParameters.timeVarying = True
    lmpc_std = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, predModel_std)
    for _ in range(4):
        lmpc_std.addTrajectory(xPID, uPID, xPID_glob)

    xS_std = [x0, x0]
    x_std_last, u_std_last = None, None

    for it in range(numSS_it, Laps):
        xLMPC, uLMPC, xLMPC_glob, xS_std = LMPCsim_std.sim(xS_std, lmpc_std)
        lmpc_std.addTrajectory(xLMPC, uLMPC, xLMPC_glob)
        predModel_std.addTrajectory(xLMPC, uLMPC)

        lap_time = lmpc_std.Qfun[it][0] * dt
        lap_times["LMPC_Standard"].append(lap_time)

        x_std_last = xLMPC
        u_std_last = uLMPC

        print(f"  Standard LMPC lap {it}: {lap_time:.2f} s")

    print("\n✓ Standard LMPC complete.\n")

    # ------------------ Phase 5: GP LMPC ------------------
    print("=" * 100)
    print("PHASE 5: GP LMPC (LMPC run in parallel, GP trained once)")
    print("=" * 100 + "\n")

    predModel_gp = PredictiveModel(n, d, map_obj, 4)
    for _ in range(4):
        predModel_gp.addTrajectory(xPID, uPID)

    lmpc_gp = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, predModel_gp)
    for _ in range(4):
        lmpc_gp.addTrajectory(xPID, uPID, xPID_glob)

    xS_gp = [x0, x0]
    x_gp_last, u_gp_last = None, None

    for it in range(numSS_it, Laps):
        xLMPC_gp, uLMPC_gp, xLMPC_gp_glob, xS_gp = LMPCsim_gp.sim(xS_gp, lmpc_gp)
        lmpc_gp.addTrajectory(xLMPC_gp, uLMPC_gp, xLMPC_gp_glob)
        predModel_gp.addTrajectory(xLMPC_gp, uLMPC_gp)

        # We still add data (for potential later analysis), but do NOT retrain repeatedly
        gp_predictor.add_trajectory(xLMPC_gp, uLMPC_gp)

        lap_time_gp = lmpc_gp.Qfun[it][0] * dt
        lap_times["LMPC_GP"].append(lap_time_gp)

        x_gp_last = xLMPC_gp
        u_gp_last = uLMPC_gp

        print(f"  GP LMPC lap {it}: {lap_time_gp:.2f} s")

    print("\n✓ GP LMPC complete.\n")
    gp_predictor.save_model("models/gp_final.pkl")

    # ------------------ Metrics ------------------
    std_metrics = compute_lmpc_results(
        lap_times["LMPC_Standard"], x_std_last, u_std_last, state_dim=n
    )
    gp_metrics = compute_lmpc_results(
        lap_times["LMPC_GP"], x_gp_last, u_gp_last, state_dim=n
    )

    print("=" * 100)
    print("FINAL METRICS (Standard vs GP LMPC)")
    print("=" * 100 + "\n")

    print("Standard LMPC:")
    print(f"  best_lap_time [s]   = {std_metrics['best_lap_time']:.3f}")
    print(f"  control_effort_last = {std_metrics['control_effort_last']:.3f}")
    print(f"  ey_rms_last         = {std_metrics['ey_rms_last']:.6f}\n")

    print("GP LMPC:")
    print(f"  best_lap_time [s]   = {gp_metrics['best_lap_time']:.3f}")
    print(f"  control_effort_last = {gp_metrics['control_effort_last']:.3f}")
    print(f"  ey_rms_last         = {gp_metrics['ey_rms_last']:.6f}\n")

    # ------------------ Single plot: lap time vs iteration ------------------
    iterations = list(range(numSS_it, Laps))

    plt.figure(figsize=(8, 5))
    if lap_times["LMPC_Standard"]:
        plt.plot(iterations, lap_times["LMPC_Standard"], "o-",
                 label="Standard LMPC")
    if lap_times["LMPC_GP"]:
        plt.plot(iterations, lap_times["LMPC_GP"], "s-",
                 label="GP LMPC")

    plt.axhline(y=lap_times["PID"], color="r", linestyle="--", alpha=0.5,
                label="PID baseline")

    plt.xlabel("LMPC Iteration (Lap #)")
    plt.ylabel("Lap Time [s]")
    plt.title("Lap Time vs Iteration: Standard LMPC vs GP LMPC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("=" * 100)
    print("SIMULATION COMPLETE")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
