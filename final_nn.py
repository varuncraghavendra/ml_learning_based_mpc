# ----------------------------------------------------------------------------------------------------------------------
# ML-ENHANCED LMPC - COMPARISON WITH STANDARD LMPC
# Final outputs:
#   1. best_lap_time (s)
#   2. control_effort_last (last LMPC lap)
#   3. ey_rms_last (last LMPC lap)
#   4. One plot: Lap time vs iteration (Standard vs ML-Enhanced LMPC)
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('fnc/simulator')
sys.path.append('fnc/controller')
sys.path.append('fnc')

import matplotlib.pyplot as plt
from initControllerParameters import initMPCParams, initLMPCParams
from PredictiveControllers import MPC, LMPC
from PredictiveModel import PredictiveModel
from Utilities import Regression, PID
from SysModel import Simulator
from Track import Map  # Using original Track.py
from NeuralNetworkDynamics import NNDynamicsPredictor  # ML component
import numpy as np
import os

# ======================= Helper metrics =======================

def control_energy(u: np.ndarray) -> float:
    """Sum of squared inputs for a lap."""
    if u is None:
        return np.nan
    return float(np.sum(u**2))

def compute_lmpc_results(lap_times, x_last, u_last, state_dim=6):
    """
    Compute:
      1) best_lap_time (s)
      2) control_effort_last (sum of squared inputs on last LMPC lap)
      3) ey_rms_last (RMS of lateral error on last LMPC lap)
    Assumes ey is the last state component: index state_dim - 1.
    """
    lap_times = np.array(lap_times, dtype=float)

    if lap_times.size == 0 or x_last is None or u_last is None:
        return {
            "best_lap_time": np.nan,
            "control_effort_last": np.nan,
            "ey_rms_last": np.nan
        }

    best_lap_time = float(np.min(lap_times))
    control_effort_last = control_energy(u_last)

    ey_index = state_dim - 1
    ey = x_last[:, ey_index]
    ey_rms_last = float(np.sqrt(np.mean(ey**2)))

    return {
        "best_lap_time": best_lap_time,
        "control_effort_last": control_effort_last,
        "ey_rms_last": ey_rms_last
    }

# ======================= Main experiment =======================

def main():
    print("\n" + "="*100)
    print(" "*25 + "ML-ENHANCED LMPC vs STANDARD LMPC")
    print("="*100 + "\n")
    
    # ------------------------------------------------------------------------------------------------------------------
    # Initialize parameters
    # ------------------------------------------------------------------------------------------------------------------
    N = 14
    n = 6
    d = 2
    x0 = np.array([0.5, 0, 0, 0, 0, 0])
    xS = [x0, x0]
    dt = 0.1
    map = Map(0.4)
    vt = 0.8

    print(f"Track Configuration:")
    print(f"  - Track Type: L-shaped (Original)")
    print(f"  - Track Length: {map.TrackLength:.2f} m")
    print(f"  - Track Half Width: {map.halfWidth} m")
    print(f"  - Target Velocity: {vt} m/s")
    print(f"  - Horizon Length: {N}")
    print(f"  - Time Step: {dt} s\n")

    # Controller parameters & simulators
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map, N)

    simulator = Simulator(map)
    LMPCsimulator = Simulator(map, multiLap=False, flagLMPC=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize Neural Network
    # ------------------------------------------------------------------------------------------------------------------
    print("="*100)
    print("INITIALIZING NEURAL NETWORK FOR ML-ENHANCED LMPC")
    print("="*100 + "\n")
    
    os.makedirs("models", exist_ok=True)
    
    nn_predictor = NNDynamicsPredictor(
        state_dim=n, 
        input_dim=d, 
        hidden_dim=128,
        learning_rate=1e-3,
        device='cpu'
    )
    print("✓ Neural Network initialized\n")

    # Storage for comparison
    lap_times = {
        'PID': None,
        'MPC': None,
        'TV-MPC': None,
        'LMPC_Standard': [],
        'LMPC_ML': []
    }

    # ------------------------------------------------------------------------------------------------------------------
    # PHASE 1: PID
    # ------------------------------------------------------------------------------------------------------------------
    print("="*100)
    print("PHASE 1: PID CONTROLLER (Baseline)")
    print("="*100)
    
    PIDController = PID(vt)
    xPID_cl, uPID_cl, xPID_cl_glob, _ = simulator.sim(xS, PIDController)
    lap_times['PID'] = xPID_cl.shape[0] * dt
    
    print(f"✓ PID Lap time: {lap_times['PID']:.2f} s\n")
    
    nn_predictor.add_trajectory(xPID_cl, uPID_cl)

    # ------------------------------------------------------------------------------------------------------------------
    # PHASE 2: MPC (linear model)
    # ------------------------------------------------------------------------------------------------------------------
    print("="*100)
    print("PHASE 2: MPC WITH LINEAR MODEL")
    print("="*100)
    
    lamb = 1e-7
    A, B, Error = Regression(xPID_cl, uPID_cl, lamb)
    mpcParam.A = A
    mpcParam.B = B
    mpc = MPC(mpcParam)
    xMPC_cl, uMPC_cl, xMPC_cl_glob, _ = simulator.sim(xS, mpc)
    lap_times['MPC'] = xMPC_cl.shape[0] * dt
    
    print(f"✓ MPC Lap time: {lap_times['MPC']:.2f} s\n")
    
    nn_predictor.add_trajectory(xMPC_cl, uMPC_cl)

    # ------------------------------------------------------------------------------------------------------------------
    # PHASE 3: TV-MPC
    # ------------------------------------------------------------------------------------------------------------------
    print("="*100)
    print("PHASE 3: TIME-VARYING MPC")
    print("="*100)
    
    predictiveModel = PredictiveModel(n, d, map, 1)
    predictiveModel.addTrajectory(xPID_cl, uPID_cl)
    ltvmpcParam.timeVarying = True 
    mpc_tv = MPC(ltvmpcParam, predictiveModel)
    xTVMPC_cl, uTVMPC_cl, xTVMPC_cl_glob, _ = simulator.sim(xS, mpc_tv)
    lap_times['TV-MPC'] = xTVMPC_cl.shape[0] * dt
    
    print(f"✓ TV-MPC Lap time: {lap_times['TV-MPC']:.2f} s\n")
    
    nn_predictor.add_trajectory(xTVMPC_cl, uTVMPC_cl)

    # ------------------------------------------------------------------------------------------------------------------
    # Train Neural Network
    # ------------------------------------------------------------------------------------------------------------------
    print("="*100)
    print("TRAINING NEURAL NETWORK ON COLLECTED DATA")
    print("="*100 + "\n")
    
    total_samples = len(nn_predictor.all_states)
    print(f"Total training samples: {total_samples}")
    nn_predictor.train(epochs=100, batch_size=64, verbose=True)
    nn_predictor.save_model("models/nn_dynamics_initial.pth")
    print("")

    # ------------------------------------------------------------------------------------------------------------------
    # PHASE 4: STANDARD LMPC
    # ------------------------------------------------------------------------------------------------------------------
    print("="*100)
    print("PHASE 4: STANDARD LMPC (Without Machine Learning)")
    print("="*100 + "\n")
    
    lmpcpredictiveModel = PredictiveModel(n, d, map, 4)
    for _ in range(4):
        lmpcpredictiveModel.addTrajectory(xPID_cl, uPID_cl)

    lmpcParameters.timeVarying = True 
    lmpc_standard = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for _ in range(4):
        lmpc_standard.addTrajectory(xPID_cl, uPID_cl, xPID_cl_glob)
    
    xS_standard = [x0, x0]
    xLMPC_std_last, uLMPC_std_last = None, None
    
    for it in range(numSS_it, Laps):
        xLMPC, uLMPC, xLMPC_glob, xS_standard = LMPCsimulator.sim(xS_standard, lmpc_standard)
        lmpc_standard.addTrajectory(xLMPC, uLMPC, xLMPC_glob)
        lmpcpredictiveModel.addTrajectory(xLMPC, uLMPC)
        
        lap_time = lmpc_standard.Qfun[it][0] * dt
        lap_times['LMPC_Standard'].append(lap_time)

        # keep last LMPC-Standard lap
        xLMPC_std_last = xLMPC
        uLMPC_std_last = uLMPC
        
        print(f"  Standard LMPC Lap {it}: {lap_time:.2f} s")
    
    print("\n✓ Standard LMPC complete.\n")

    # ------------------------------------------------------------------------------------------------------------------
    # PHASE 5: ML-ENHANCED LMPC
    # ------------------------------------------------------------------------------------------------------------------
    print("="*100)
    print("PHASE 5: ML-ENHANCED LMPC (With Neural Network)")
    print("="*100 + "\n")
    
    xS_ml = [x0, x0]
    LMPCsimulator_ML = Simulator(map, multiLap=False, flagLMPC=True)
    
    lmpcpredictiveModel_ML = PredictiveModel(n, d, map, 4)
    for _ in range(4):
        lmpcpredictiveModel_ML.addTrajectory(xPID_cl, uPID_cl)

    lmpc_ml = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel_ML)
    for _ in range(4):
        lmpc_ml.addTrajectory(xPID_cl, uPID_cl, xPID_cl_glob)
    
    xLMPC_ml_last, uLMPC_ml_last = None, None
    
    for it in range(numSS_it, Laps):
        xLMPC_ML, uLMPC_ML, xLMPC_ML_glob, xS_ml = LMPCsimulator_ML.sim(xS_ml, lmpc_ml)
        lmpc_ml.addTrajectory(xLMPC_ML, uLMPC_ML, xLMPC_ML_glob)
        lmpcpredictiveModel_ML.addTrajectory(xLMPC_ML, uLMPC_ML)
        
        # Add new data to neural network
        nn_predictor.add_trajectory(xLMPC_ML, uLMPC_ML)
        
        # Retrain every 3 laps (optional; keep as in your original script)
        if it % 3 == 0 and it > numSS_it:
            print(f"\n  → Retraining neural network after lap {it}...")
            nn_predictor.train(epochs=20, batch_size=64, verbose=False)
            nn_predictor.save_model(f"models/nn_dynamics_lap{it}.pth")
            print(f"  ✓ Retrained and saved\n")
        
        lap_time_ml = lmpc_ml.Qfun[it][0] * dt
        lap_times['LMPC_ML'].append(lap_time_ml)

        # keep last LMPC-ML lap
        xLMPC_ml_last = xLMPC_ML
        uLMPC_ml_last = uLMPC_ML
        
        print(f"  ML-Enhanced LMPC Lap {it}: {lap_time_ml:.2f} s")
    
    print("\n✓ ML-Enhanced LMPC complete.\n")
    
    nn_predictor.save_model("models/nn_dynamics_final.pth")

    # ------------------------------------------------------------------------------------------------------------------
    # FINAL METRICS (what you asked for)
    # ------------------------------------------------------------------------------------------------------------------
    std_metrics = compute_lmpc_results(
        lap_times['LMPC_Standard'],
        xLMPC_std_last,
        uLMPC_std_last,
        state_dim=n
    )

    ml_metrics = compute_lmpc_results(
        lap_times['LMPC_ML'],
        xLMPC_ml_last,
        uLMPC_ml_last,
        state_dim=n
    )

    print("="*100)
    print("FINAL LMPC METRICS")
    print("="*100 + "\n")

    print("Standard LMPC:")
    print(f"  best_lap_time [s]        = {std_metrics['best_lap_time']:.3f}")
    print(f"  control_effort_last      = {std_metrics['control_effort_last']:.3f}")
    print(f"  ey_rms_last              = {std_metrics['ey_rms_last']:.6f}\n")

    print("ML-Enhanced LMPC:")
    print(f"  best_lap_time [s]        = {ml_metrics['best_lap_time']:.3f}")
    print(f"  control_effort_last      = {ml_metrics['control_effort_last']:.3f}")
    print(f"  ey_rms_last              = {ml_metrics['ey_rms_last']:.6f}\n")

    # ------------------------------------------------------------------------------------------------------------------
    # ONE PLOT: Lap time vs iteration
    # ------------------------------------------------------------------------------------------------------------------
    iterations = list(range(numSS_it, Laps))
    plt.figure(figsize=(8, 5))

    if len(lap_times['LMPC_Standard']) > 0:
        plt.plot(iterations, lap_times['LMPC_Standard'], 'o-', label='Standard LMPC')
    if len(lap_times['LMPC_ML']) > 0:
        plt.plot(iterations, lap_times['LMPC_ML'], 's-', label='ML-Enhanced LMPC')

    plt.xlabel('LMPC Iteration (Lap #)')
    plt.ylabel('Lap Time [s]')
    plt.title('LMPC Lap Time vs Iteration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("="*100)
    print("SIMULATION COMPLETE")
    print("="*100 + "\n")


if __name__== "__main__":
    main()
