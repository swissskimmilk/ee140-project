import math
import config
import numpy as np 

"""
Preliminary design calculations for the 2-stage Miller-compensated op-amp.

Goal:
  - Allocate error between:
      * RC load (irreducible)
      * static error (finite loop gain)
      * dynamic error (finite bandwidth of amp)
  - From that, derive:
      * Required DC loop gain A0
      * Required unity-gain frequency f_u
      * Required gm1 (for CC)
      * Required gm2 (for p2 placement)
      * Candidate Rz values
"""

# ----- 1. SYSTEM SPECS (Imported from config.py) -----
VDDL_MAX          = config.VDDL_MAX
VDDH_MAX          = config.VDDH_MAX
G_CLOSED_LOOP     = config.G_CLOSED_LOOP
RL                = config.RL
CL                = config.CL
TOTAL_ERROR_SPEC  = config.TOTAL_ERROR_SPEC
POWER_MAX         = config.POWER_MAX
PHASE_MARGIN_MIN  = config.PHASE_MARGIN_MIN
OUTPUT_SWING_MIN  = config.OUTPUT_SWING_MIN

# ----- 2. TIMING & ERROR BUDGET (Imported from config.py) -----
REFRESH_RATE              = config.REFRESH_RATE
T_PIXEL                   = config.T_PIXEL
ERROR_SPLIT_STATIC_RATIO  = config.ERROR_SPLIT_STATIC_RATIO
CC                        = config.CC
C_OUT_P                   = config.C_OUT_P
FIRST_STAGE_GAIN          = config.FIRST_STAGE_GAIN
SECOND_STAGE_GAIN         = config.SECOND_STAGE_GAIN
P2_FACTOR                 = config.P2_FACTOR

# Design choice: how much faster (in CLOSED LOOP) we want the amp
# compared to the load RC pole. This is a heuristic margin.
MARGIN_OVER_LOAD_CL = 2.2

# ----- 3. BASIC LOAD CALCS -----

# RC time constant at the pixel node
tau_load    = RL * CL
f_3db_load  = 1.0 / (2.0 * math.pi * tau_load)

# Dynamic error from the RC alone at T_PIXEL
ERROR_RC = math.e ** (-T_PIXEL / tau_load)

# Sanity: if RC alone exceeds spec, impossible
if TOTAL_ERROR_SPEC <= ERROR_RC:
    raise ValueError(
        f"Spec impossible: RC-only error at T_PIXEL is {ERROR_RC:.4e} "
        f"which exceeds TOTAL_ERROR_SPEC={TOTAL_ERROR_SPEC:.4e}"
    )

# Remaining error budget AFTER unavoidable RC behavior
remaining_error = TOTAL_ERROR_SPEC - ERROR_RC

# Split this remaining error between static and "extra" dynamic (amp-limited)
ERROR_STATIC  = remaining_error * ERROR_SPLIT_STATIC_RATIO
ERROR_DYNAMIC = remaining_error * (1.0 - ERROR_SPLIT_STATIC_RATIO)

if not (0 < ERROR_STATIC < 1 and 0 < ERROR_DYNAMIC < 1):
    raise ValueError(
        f"Bad error split: ERROR_STATIC={ERROR_STATIC}, "
        f"ERROR_DYNAMIC={ERROR_DYNAMIC} must both be between 0 and 1."
    )

# Planned total error if budgets are exactly met
total_error_planned = ERROR_RC + ERROR_STATIC + ERROR_DYNAMIC

# ----- 4. LOOP GAIN REQUIREMENT FROM STATIC ERROR -----

# β for gain-of-2 feedback
BETA = 1.0 / G_CLOSED_LOOP   # with large A0, ACL ≈ 1/β = G_CLOSED_LOOP

# Static error ≈ 1 / (1 + β * A0)
# Require: 1 / (1 + β A0) <= ERROR_STATIC
#   => 1 + β A0 >= 1 / ERROR_STATIC
#   => β A0 >= (1 / ERROR_STATIC) - 1
loop_gain_required = (1.0 / ERROR_STATIC) - 1.0
A0_open_loop_required = loop_gain_required / BETA

# ----- 5. BANDWIDTH REQUIREMENT FROM DYNAMIC ERROR -----
#
# Approximation:
#   - Treat amplifier's closed-loop response as first-order with τ_amp.
#   - Demand that its own dynamic error at T_PIXEL be <= ERROR_DYNAMIC.
#
# So: e^(-T_PIXEL / τ_amp) <= ERROR_DYNAMIC
#     => τ_amp <= -T_PIXEL / ln(ERROR_DYNAMIC)
#
# Then f_3dB,CL_amp = 1 / (2π τ_amp)
# and f_u ≈ f_3dB,CL_amp / BETA

tau_amp_required = -T_PIXEL / math.log(ERROR_DYNAMIC)
f_3db_cl_required = 1.0 / (2.0 * math.pi * tau_amp_required)
fu_from_dyn = f_3db_cl_required / BETA

# ----- 6. SPEED REQUIREMENT RELATIVE TO LOAD POLE -----
#
# We want the closed-loop amplifier BW to be some factor above the load BW:
#   f_3dB,CL_amp >= MARGIN_OVER_LOAD_CL * f_3dB_load
#
# But f_3dB,CL_amp ≈ f_u * BETA
#   => f_u >= (MARGIN_OVER_LOAD_CL * f_3dB_load) / BETA

fu_from_load = (MARGIN_OVER_LOAD_CL * f_3db_load) / BETA

# Final required unity-gain frequency: must satisfy BOTH constraints
f_u_required = max(fu_from_dyn, fu_from_load)

# ----- 7. STAGE 1 DESIGN (Miller Compensation) -----
#
# For classical Miller:
#   f_u ≈ gm1 / (2π Cc)  =>  gm1 = 2π Cc f_u

gm1_required = 2.0 * math.pi * CC * f_u_required

# ----- 8. STAGE 2 DESIGN (p2 placement for stability) -----
#
# For rough modeling:
#   p2 ≈ gm2 / C_OUT
# with
#   C_OUT = CL + C_OUT_P + Cc (1 + 1/Av2)
#
# We want p2 = P2_FACTOR * f_u_required (e.g. 2x, 2.5x, etc.)

p2_target = P2_FACTOR * f_u_required

CL_eff = CL / (1.0 + (2 * np.pi * p2_target * RL * CL)**2)

C_OUT = CL_eff + C_OUT_P + CC * (1.0 + 1.0 / SECOND_STAGE_GAIN)

gm2_required_stability = 2.0 * math.pi * C_OUT * p2_target

# ----- 9. NULLING RESISTOR DESIGN (Rz) -----
#
# RHP zero (without Rz): z_RHP = gm2 / Cc
# With Rz, zero moves to:
#   z = 1 / [Cc * (1/gm2 - Rz)]
#
# Strategy A (Rz_infinity): push zero to infinity => Rz = 1/gm2
# Strategy B (Rz_cancel_p2): approximate cancellation of p2

Rz_infinity = 1.0 / gm2_required_stability
Rz_cancel_p2 = (C_OUT + CC) / (gm2_required_stability * CC)


def print_design_report():
    print(f"{'='*60}")
    print(f"{'PRELIM DESIGN PARAMETER CALC (Miller + Nulling R)':^60}")
    print(f"{'='*60}\n")
    
    print(f"--- Spec & Error Budget ---")
    print(f"Total Error Spec:           {TOTAL_ERROR_SPEC*100:.3f}%")
    print(f"  RC-only error @ T_PIXEL:  {ERROR_RC*100:.3f}%")
    print(f"  Remaining error budget:   {remaining_error*100:.3f}%")
    print(f"    -> Static budget:       {ERROR_STATIC*100:.3f}%")
    print(f"    -> Dynamic (amp) budget:{ERROR_DYNAMIC*100:.3f}%")
    print(f"Planned total (RC+stat+dyn):{total_error_planned*100:.3f}%")
    print("")
    print(f"T_PIXEL:                    {T_PIXEL*1e9:.2f} ns")
    print(f"Load τ:                     {tau_load*1e9:.2f} ns")
    print(f"Load f_3dB:                 {f_3db_load/1e6:.2f} MHz")
    print("")
    
    print(f"--- Static Error → Loop Gain ---")
    print(f"β (for G=2):                {BETA:.3f}")
    print(f"Required loop gain βA0:     {loop_gain_required:.2f}")
    print(f"Required A0 (open-loop):    {A0_open_loop_required:.2f} V/V "
          f"({20*math.log10(A0_open_loop_required):.1f} dB)")
    print("")
    
    print(f"--- Dynamic Error → f_u ---")
    print(f"Required τ_amp from dyn:    {tau_amp_required*1e9:.2f} ns")
    print(f"Required f_3dB,CL (amp):    {f_3db_cl_required/1e6:.2f} MHz")
    print(f"f_u from dyn requirement:   {fu_from_dyn/1e6:.2f} MHz")
    print(f"f_u from load margin (CL):  {fu_from_load/1e6:.2f} MHz "
          f"(CL margin={MARGIN_OVER_LOAD_CL}x)")
    print(f"--> Final required f_u:     {f_u_required/1e6:.2f} MHz")
    print("")
    
    print(f"--- Stage 1 (Input) ---")
    print(f"Cc:                         {CC*1e12:.2f} pF")
    print(f"Required gm1:               {gm1_required*1e6:.2f} µS")
    print("")
    
    print(f"--- Stage 2 (Output) ---")
    print(f"C_OUT estimate:             {C_OUT*1e12:.2f} pF")
    print(f"Target p2:                  {p2_target/1e6:.2f} MHz "
          f"(= {P2_FACTOR:.2f} * f_u)")
    print(f"Required gm2 (stability):   {gm2_required_stability*1e6:.2f} µS")
    print("")
    
    print(f"--- Nulling Resistor (Rz) ---")
    print(f"Option A (zero @ ∞):        Rz = {Rz_infinity:.2f} Ω")
    print(f"Option B (cancel p2):       Rz = {Rz_cancel_p2:.2f} Ω")
    print("")

if __name__ == "__main__":
    print_design_report()
