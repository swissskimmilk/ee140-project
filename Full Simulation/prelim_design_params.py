import math
import config
import numpy as np 

# ----- 1. SYSTEM SPECS (Imported from config.py) -----
VDDL_MAX = config.VDDL_MAX
VDDH_MAX = config.VDDH_MAX
G_CLOSED_LOOP = config.G_CLOSED_LOOP
RL = config.RL
CL = config.CL
TOTAL_ERROR_SPEC = config.TOTAL_ERROR_SPEC
POWER_MAX = config.POWER_MAX
PHASE_MARGIN_MIN = config.PHASE_MARGIN_MIN
OUTPUT_SWING_MIN = config.OUTPUT_SWING_MIN

# ----- 2. TIMING & ERROR BUDGET (Imported from config.py) -----
REFRESH_RATE = config.REFRESH_RATE
T_PIXEL = config.T_PIXEL
ERROR_SPLIT_STATIC_RATIO = config.ERROR_SPLIT_STATIC_RATIO
CC = config.CC
C_OUT_P = config.C_OUT_P
FIRST_STAGE_GAIN = config.FIRST_STAGE_GAIN
SECOND_STAGE_GAIN = config.SECOND_STAGE_GAIN
P2_FACTOR = config.P2_FACTOR

# ----- 3. CALCULATIONS & DERIVATIONS -----

# Calculate error due to RC time constant of output 
tau_load = RL * CL
f_3db_load = 1 / (2 * np.pi * tau_load)
ERROR_RC = np.e ** (-T_PIXEL / tau_load)
ERROR_STATIC = (TOTAL_ERROR_SPEC - ERROR_RC) * ERROR_SPLIT_STATIC_RATIO
ERROR_DYNAMIC = (TOTAL_ERROR_SPEC - ERROR_RC) * (1 - ERROR_SPLIT_STATIC_RATIO)

total_error_planned = ERROR_RC + ERROR_STATIC + ERROR_DYNAMIC

if TOTAL_ERROR_SPEC <= ERROR_RC:
    raise ValueError("Spec impossible: RC alone exceeds TOTAL_ERROR_SPEC")

if not (0 < ERROR_STATIC < 1 and 0 < ERROR_DYNAMIC < 1):
    raise ValueError("Bad error split: STATIC/DYNAMIC must be between 0 and 1")

# Compute gain requirements based on static error 
BETA = 1 / G_CLOSED_LOOP    # G_CLOSED_LOOP is set by the feedback, for sufficiently large gain
loop_gain_required = (1 / ERROR_STATIC) - 1
A0_open_loop_required = loop_gain_required / BETA

# Bandwidth Requirements (Dynamic Error) ---
tau_required = -T_PIXEL / math.log(ERROR_DYNAMIC)
f_3db_cl_required = 1 / (2 * math.pi * tau_required)

# Ensure that the amp pole is much faster than the RC output 
MARGIN_OVER_LOAD = 3.0
fu_from_dyn  = f_3db_cl_required / BETA
fu_from_load = MARGIN_OVER_LOAD * f_3db_load
f_u_required = max(fu_from_dyn, fu_from_load) # Applies to both the first and second stage 

# Stage 1 Design (Input Stage) - Miller Compensation
# In Miller Comp: f_u = gm1 / (2 * pi * Cc)
gm1_required = f_u_required * 2 * math.pi * CC

# Stage 2 Design (Output Stage) - Pole Splitting Requirement
# The non-dominant pole p2 is approx gm2 / CL (assuming Cgs2 << CL and Cdb1 << Cc)
# For 60 deg phase margin, we typically want p2 >= 2.2 * f_u (or at least > 2*fu)
# Let's target p2 = 2.5 * f_u to be safe.
p2_target = P2_FACTOR * f_u_required

C_OUT = CL + C_OUT_P + CC * (1 + 1/SECOND_STAGE_GAIN)
gm2_required_stability = p2_target * 2 * math.pi * C_OUT

# Nulling Resistor Design
# Rz is used to move the RHP zero.
# Z_RHP = gm2 / Cc. 
# With Rz: Zero = 1 / (Cc * (1/gm2 - Rz))
# Strategy 1: Push Zero to Infinity => Rz = 1/gm2
# Strategy 2: Cancel p2 (LHP) => Rz = (CL + Cc) / (gm2 * Cc) (approx)
Rz_infinity = 1 / gm2_required_stability
Rz_cancel_p2 = (C_OUT + CC) / (gm2_required_stability * CC)


def print_design_report():
    print(f"{'='*60}")
    print(f"{'DESIGN PARAMETER CALCULATION (Miller + Nulling R)':^60}")
    print(f"{'='*60}\n")
    
    print(f"--- Spec Verification ---")
    print(f"Total Error Spec: {TOTAL_ERROR_SPEC*100:.3f}% (Planned: {total_error_planned*100:.3f}%)")
    print(f"Pixel Time: {T_PIXEL*1e9:.1f} ns -> Tau Required: {tau_required*1e9:.1f} ns")
    print(f"Required f_u: {f_u_required/1e6:.2f} MHz")
    print(f"Required open loop gain (A0): {A0_open_loop_required:.2f} V/V")
    print("")

    print(f"--- Compensation Strategy (Miller) ---")
    print(f"Compensation Capacitor (Cc): {CC*1e12:.1f} pF")
    print(f"Required gm1 (for f_u): {gm1_required*1e6:.1f} uS")
    print("")

    print(f"--- Stage 2 (Output) Requirements for Stability ---")
    print(f"Output Capacitance (C_OUT): {C_OUT*1e12:.1f} pF")
    print(f"Target Non-Dominant Pole (p2): {p2_target/1e6:.1f} MHz (2.5x f_u)")
    print(f"Required gm2 (for p2): {gm2_required_stability*1e6:.1f} uS")
    print(f"  -> This ensures the output pole is high enough.")
    print("")

    print(f"--- Nulling Resistor (Rz) ---")
    print(f"Option A (Zero @ Inf): Rz = 1/gm2 = {Rz_infinity:.1f} Ohms")
    print(f"Option B (Cancel p2):  Rz = (CL+Cc)/(gm2*Cc) = {Rz_cancel_p2:.1f} Ohms")
    print("")

if __name__ == "__main__":
    print_design_report()
