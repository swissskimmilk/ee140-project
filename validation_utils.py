"""
Validation Utilities for Amplifier Design

This module contains utilities for validating and checking various aspects
of the amplifier design, including:
- Error budget impact analysis
- Configuration testing
- gm/gds range validation
- Slew rate checks
"""

import math
import config as cfg
import calculate_design_params as params
from look_up import *

# Test current configuration
def test_current_config():
    """Test the current configuration from config.py"""
    
    error_static = cfg.ERROR_SPLIT_STATIC
    error_dynamic = cfg.ERROR_SPLIT_DYNAMIC
    CC = cfg.CC
    VDDH = cfg.VDDH_MAX
    
    TOTAL_ERROR = cfg.TOTAL_ERROR_SPEC
    T_PIXEL = cfg.T_PIXEL
    RL = cfg.RL
    CL = cfg.CL
    BETA = 1 / cfg.G_CLOSED_LOOP
    VDDL = cfg.VDDL_MAX
    POWER_MAX = cfg.POWER_MAX
    
    print("="*70)
    print("CURRENT CONFIGURATION TEST")
    print("="*70)
    print(f"Static: {error_static*100:.3f}%, Dynamic: {error_dynamic*100:.3f}%")
    print(f"CC: {CC*1e12:.1f} pF, VDDH: {VDDH} V\n")
    
    # Stage 1 analysis
    loop_gain_req = (1/error_static) - 1
    A0_req = loop_gain_req / BETA
    tau_required = -T_PIXEL / math.log(error_dynamic)
    f_3db_cl = 1 / (2 * math.pi * tau_required)
    f_u = f_3db_cl / BETA
    gm1_req = f_u * 2 * math.pi * CC
    
    gm_id_stage1 = 25
    I_stage1 = 2 * (gm1_req / gm_id_stage1)
    P_stage1 = I_stage1 * VDDL
    
    print("Stage 1:")
    print(f"  A0 required: {A0_req:.0f} V/V")
    print(f"  f_u: {f_u/1e6:.2f} MHz")
    print(f"  gm1: {gm1_req*1e6:.2f} uS")
    print(f"  Power: {P_stage1*1e3:.3f} mW\n")
    
    # Stage 2 analysis
    p2_target = 2.5 * f_u
    gm2_stability = p2_target * 2 * math.pi * CL
    
    print("Stage 2:")
    print(f"  gm2 (stability): {gm2_stability*1e3:.2f} mS\n")
    
    for gm_id_stage2 in [28, 30, 32, 34]:
        I_stage2 = gm2_stability / gm_id_stage2
        P_stage2 = I_stage2 * VDDH
        P_total = P_stage1 + P_stage2
        
        print(f"  gm/ID = {gm_id_stage2}:")
        print(f"    I_stage2: {I_stage2*1e6:.1f} uA")
        print(f"    P_stage2: {P_stage2*1e3:.3f} mW")
        print(f"    P_total: {P_total*1e3:.3f} mW")
        print(f"    Within budget? {P_total <= POWER_MAX}")
        
        if P_total <= POWER_MAX:
            gm_gds = 15
            Rout_est = gm_gds / gm2_stability
            tau_actual = (Rout_est + RL) * CL
            error_at_tpixel = math.exp(-T_PIXEL / tau_actual)
            t_settle = -tau_actual * math.log(TOTAL_ERROR)
            
            print(f"    Rout: {Rout_est:.0f} Ohm")
            print(f"    tau: {tau_actual*1e9:.2f} ns")
            print(f"    Error at T_PIXEL: {error_at_tpixel*100:.3f}%")
            print(f"    Settling time: {t_settle*1e9:.1f} ns")
            print(f"    Meets settling? {t_settle <= T_PIXEL}")
            
            if t_settle <= T_PIXEL:
                FOM = 1 / (P_total * t_settle)
                print(f"    FOM: {FOM/1e9:.2f} x 10^9")
        print()

def check_error_budget_impact():
    """Analyze the impact of error budget allocation on settling time"""
    
    T_PIXEL = cfg.T_PIXEL
    RL = cfg.RL  
    CL = cfg.CL
    
    print("="*70)
    print("ERROR BUDGET IMPACT ON SETTLING TIME")
    print("="*70 + "\n")
    
    # Physical limit
    tau_min = RL * CL
    print(f"Physical Limit (Rout=0): tau_min = {tau_min*1e9:.2f} ns\n")
    
    # Different error tolerances
    for error_pct, label in [(cfg.ERROR_SPLIT_DYNAMIC, "DYNAMIC"), 
                              (cfg.TOTAL_ERROR_SPEC, "TOTAL")]:
        tau_req = -T_PIXEL / math.log(error_pct)
        settling_time_estimate = -tau_min * math.log(error_pct)
        
        print(f"If checking against {error_pct*100:.3f}% ({label}):")
        print(f"  Required tau: {tau_req*1e9:.2f} ns")
        print(f"  Best-case settling (Rout=0): {settling_time_estimate*1e9:.2f} ns")
        print(f"  Passes 180ns spec? {'YES' if settling_time_estimate < T_PIXEL else 'NO'}\n")
    
    print("="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    print(f"ERROR_SPLIT_STATIC  = {cfg.ERROR_SPLIT_STATIC*100:.3f}%")
    print(f"ERROR_SPLIT_DYNAMIC = {cfg.ERROR_SPLIT_DYNAMIC*100:.3f}%")
    print(f"TOTAL_ERROR_SPEC    = {cfg.TOTAL_ERROR_SPEC*100:.3f}%\n")

def check_gmgds_range():
    """Check gm/gds vs gm/ID for available devices"""
    
    print("="*70)
    print("DEVICE CHARACTERIZATION - gm/gds vs gm/ID")
    print("="*70 + "\n")
    
    nch = importdata('nch_2v.mat')
    pch = importdata('pch_2v.mat')
    
    print("Checking gm/gds vs gm/ID for L=0.15um, VDS=0.9V:\n")
    print("NMOS (nch_2v):")
    print(f"{'gm/ID':<10} {'gm/gds':<10}")
    print("-" * 20)
    for gm_id in [8, 10, 12, 15, 18, 20, 25, 30, 35]:
        try:
            gmgds = look_up_vs_gm_id(nch, 'GM_GDS', gm_id, l=0.15, vds=0.9)
            print(f"{gm_id:<10} {gmgds:<10.1f}")
        except:
            print(f"{gm_id:<10} FAILED")
    
    print("\nPMOS (pch_2v):")
    print(f"{'gm/ID':<10} {'gm/gds':<10}")
    print("-" * 20)
    for gm_id in [8, 10, 12, 15, 18, 20, 25, 30, 35]:
        try:
            gmgds = look_up_vs_gm_id(pch, 'GM_GDS', gm_id, l=0.15, vds=0.9)
            print(f"{gm_id:<10} {gmgds:<10.1f}")
        except:
            print(f"{gm_id:<10} FAILED")
    
    print("\n" + "="*70)
    print("Analysis:")
    print("- gm/ID = 30 is in weak inversion (high efficiency)")
    print("- At L_min = 0.15um, gm/gds ~ 50 is achievable in weak inversion")
    print("- For moderate inversion (gm/ID=15), gm/gds is lower")
    print("="*70 + "\n")

def check_slew_rate():
    """Check if Stage 1 current is sufficient to avoid slew limiting at CC"""
    
    print("="*70)
    print("SLEW RATE LIMITATION CHECK")
    print("="*70 + "\n")
    
    # Stage 1 parameters
    I_tail_stage1 = 4.75e-6  # A (from design)
    CC = cfg.CC
    
    # Calculate slew rate at CC
    SR_CC = I_tail_stage1 / CC
    
    print(f"Stage 1 Tail Current: {I_tail_stage1*1e6:.2f} uA")
    print(f"Compensation Capacitor: {CC*1e12:.2f} pF")
    print(f"Slew Rate at CC: SR = I_tail / CC = {SR_CC/1e6:.2f} V/us\n")
    
    # Check voltage swing at CC
    V_output_swing = 0.7  # V
    A2_estimate = params.A2_gain_est
    V_stage1_swing_est = V_output_swing / A2_estimate
    
    print("--- Voltage Swing Estimation ---")
    print(f"Output swing (test): {V_output_swing} V")
    print(f"Stage 2 gain (A2): {A2_estimate:.2f} V/V")
    print(f"Stage 1 output swing (estimate): {V_stage1_swing_est:.3f} V\n")
    
    # Time to slew
    t_slew = V_stage1_swing_est / SR_CC
    
    print("--- Slew Time Calculation ---")
    print(f"Time to slew {V_stage1_swing_est:.3f}V at {SR_CC/1e6:.2f} V/us:")
    print(f"  t_slew = {t_slew*1e9:.2f} ns\n")
    
    # Compare to spec
    T_PIXEL = cfg.T_PIXEL
    
    print("--- Comparison to Spec ---")
    print(f"Required settling time: {T_PIXEL*1e9:.1f} ns")
    print(f"Estimated slew time: {t_slew*1e9:.2f} ns\n")
    
    if t_slew > T_PIXEL * 0.5:
        print("[WARNING] Slew time is significant (> 50% of spec)!")
        print("  Design may be slew-rate limited at CC!")
        
        SR_required = V_stage1_swing_est / (T_PIXEL * 0.3)
        I_tail_required = SR_required * CC
        
        print(f"\nTo limit slewing to 30% of settling time:")
        print(f"  Required SR: {SR_required/1e6:.2f} V/us")
        print(f"  Required I_tail: {I_tail_required*1e6:.2f} uA")
        print(f"  Current I_tail: {I_tail_stage1*1e6:.2f} uA")
        print(f"  Need to INCREASE by: {(I_tail_required/I_tail_stage1 - 1)*100:.1f}%\n")
        
    elif t_slew > T_PIXEL * 0.2:
        print("[CAUTION] Slew time is moderate (20-50% of spec)")
        print("  May impact settling, but likely acceptable\n")
    else:
        print("[OK] Slew time is small (< 20% of spec)")
        print("  Settling is bandwidth-limited, not slew-limited\n")
    
    print("="*70)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "config":
            test_current_config()
        elif test_name == "error":
            check_error_budget_impact()
        elif test_name == "gmgds":
            check_gmgds_range()
        elif test_name == "slew":
            check_slew_rate()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: config, error, gmgds, slew")
    else:
        print("Running all validation checks:\n")
        test_current_config()
        print("\n" + "="*70 + "\n")
        check_error_budget_impact()
        print("\n" + "="*70 + "\n")
        check_gmgds_range()
        print("\n" + "="*70 + "\n")
        check_slew_rate()



