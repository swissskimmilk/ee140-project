"""
FOM Optimization for Two-Stage Miller-Compensated Amplifier

This script optimizes the Figure of Merit (FOM = 1/(Power * Settling_Time))
by sweeping error budget allocation, compensation capacitor, and supply voltage.

Key Insights:
- Two-stage settling = CC slewing + RL-CL settling
- RL-CL component is fixed by load (~155ns)
- CC slewing depends on I_tail and CC (design variables)
- Minimizing power with faster settling maximizes FOM
"""

import numpy as np
import math

# Constants
TOTAL_ERROR = 0.002
T_PIXEL = 180e-9  
RL = 1000
CL = 25e-12
BETA = 0.5
VDDL = 1.1
POWER_MAX = 1.25e-3

tau_settling = RL * CL  # 25 ns
t_settle_rlcl = -tau_settling * math.log(TOTAL_ERROR)  # ~155 ns

def calculate_design_metrics(error_static, error_dynamic, CC, VDDH):
    """
    Calculate power, settling time, and FOM for given configuration.
    
    Returns dict with metrics or None if infeasible.
    """
    
    if abs(error_static + error_dynamic - TOTAL_ERROR) > 1e-6:
        return None
    
    if error_static <= 0 or error_dynamic <= 0:
        return None
    
    # Stage 1 requirements
    loop_gain_req = (1/error_static) - 1
    A0_req = loop_gain_req / BETA
    
    tau_required = -T_PIXEL / math.log(error_dynamic)
    f_3db_cl = 1 / (2 * math.pi * tau_required)
    f_u = f_3db_cl / BETA
    
    gm1_req = f_u * 2 * math.pi * CC
    
    # Stage 1 power (optimal gm/ID = 26 for efficiency)
    gm_id_stage1 = 26
    I_stage1 = 2 * (gm1_req / gm_id_stage1)
    I_tail_stage1 = I_stage1
    P_stage1 = I_stage1 * VDDL
    
    # CC slewing time
    SR_CC = I_tail_stage1 / CC
    A2_est = 5.0
    V_stage1_swing = 0.7 / A2_est
    t_slew_CC = V_stage1_swing / SR_CC
    
    # Stage 2 requirements (stability)
    p2_target = 2.5 * f_u
    gm2_req = p2_target * 2 * math.pi * CL
    
    # Stage 2 power (optimal gm/ID = 31 for 2V devices)
    gm_id_stage2 = 31
    I_stage2 = gm2_req / gm_id_stage2
    P_stage2 = I_stage2 * VDDH
    
    P_total = P_stage1 + P_stage2
    
    if P_total > POWER_MAX:
        return None
    
    # Total settling time
    t_settle_rlcl = -tau_settling * math.log(TOTAL_ERROR)
    t_settle_total = t_slew_CC + t_settle_rlcl
    
    if t_settle_total > T_PIXEL:
        return None
    
    FOM = 1 / (P_total * t_settle_total)
    
    return {
        'error_static': error_static,
        'error_dynamic': error_dynamic,
        'CC': CC,
        'VDDH': VDDH,
        'A0': A0_req,
        'f_u': f_u,
        'gm1': gm1_req,
        'gm2': gm2_req,
        'P_stage1': P_stage1,
        'P_stage2': P_stage2,
        'P_total': P_total,
        't_slew_CC': t_slew_CC,
        't_settle_rlcl': t_settle_rlcl,
        't_settle_total': t_settle_total,
        'FOM': FOM,
        'I_tail': I_tail_stage1
    }

def optimize_fom():
    """Run FOM optimization sweep"""
    
    print("="*80)
    print("FOM OPTIMIZATION - Two-Stage Settling Model")  
    print("="*80)
    print(f"\nKey Insight: Two-stage settling = CC slewing + RL-CL settling")
    print(f"  RL-CL component: {t_settle_rlcl*1e9:.1f} ns (fixed by load)")
    print(f"  CC slewing: depends on I_tail and CC (design variables)")
    print(f"\nOptimization goal: MINIMIZE (Power x Total_Settling_Time)\n")
    
    # Search space
    static_range = np.linspace(0.00075, 0.00125, 30)
    CC_range = [0.8, 1.0, 1.2, 1.5, 2.0]  # pF
    VDDH_range = [1.5, 1.6, 1.7, 1.8]  # V
    
    all_results = []
    
    print("Searching parameter space...")
    for error_static in static_range:
        error_dynamic = TOTAL_ERROR - error_static
        
        if error_dynamic <= 0:
            continue
        
        for CC_pF in CC_range:
            for VDDH in VDDH_range:
                result = calculate_design_metrics(error_static, error_dynamic, CC_pF * 1e-12, VDDH)
                if result:
                    all_results.append(result)
    
    print(f"Found {len(all_results)} feasible designs\n")
    
    if not all_results:
        print("No feasible designs found!")
        return None
    
    # Sort by FOM (highest first)
    all_results.sort(key=lambda x: x['FOM'], reverse=True)
    
    # Display top 15
    print("="*80)
    print("TOP 15 DESIGNS (Highest FOM)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Static%':<9} {'Dynamic%':<10} {'CC(pF)':<8} {'VDDH(V)':<9} "
          f"{'Power(mW)':<11} {'FOM(10^9)':<10}")
    print("-"*80)
    
    for i, r in enumerate(all_results[:15]):
        print(f"{i+1:<5} {r['error_static']*100:<9.3f} {r['error_dynamic']*100:<10.3f} "
              f"{r['CC']*1e12:<8.1f} {r['VDDH']:<9.2f} "
              f"{r['P_total']*1e3:<11.3f} {r['FOM']/1e9:<10.2f}")
    
    optimal = all_results[0]
    
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION")
    print("="*80)
    print(f"\nUpdate config.py with:")
    print(f"  ERROR_SPLIT_STATIC  = {optimal['error_static']:.6f}  # {optimal['error_static']*100:.3f}%")
    print(f"  ERROR_SPLIT_DYNAMIC = {optimal['error_dynamic']:.6f}  # {optimal['error_dynamic']*100:.3f}%")
    print(f"  CC = {optimal['CC']*1e12:.1f}e-12  # {optimal['CC']*1e12:.1f} pF")
    print(f"\nAnd target VDDH in design_output_stage.py:")
    print(f"  VDDH = {optimal['VDDH']} V")
    
    print(f"\nExpected Performance:")
    print(f"  DC Gain: {optimal['A0']:.0f} V/V ({20*math.log10(optimal['A0']):.1f} dB)")
    print(f"  Unity Gain Freq: {optimal['f_u']/1e6:.2f} MHz")
    print(f"  Stage 1 gm1: {optimal['gm1']*1e6:.2f} uS")
    print(f"  Stage 2 gm2: {optimal['gm2']*1e3:.2f} mS")
    print(f"\n  Stage 1 Power: {optimal['P_stage1']*1e3:.3f} mW")
    print(f"  Stage 2 Power: {optimal['P_stage2']*1e3:.3f} mW")
    print(f"  Total Power: {optimal['P_total']*1e3:.3f} mW")
    print(f"  Power Margin: {(POWER_MAX - optimal['P_total'])/POWER_MAX*100:.1f}%")
    print(f"\n  CC Slewing: {optimal['t_slew_CC']*1e9:.1f} ns")
    print(f"  RL-CL Settling: {optimal['t_settle_rlcl']*1e9:.1f} ns")
    print(f"  Total Settling: {optimal['t_settle_total']*1e9:.1f} ns")
    print(f"  Settling Margin: {(T_PIXEL - optimal['t_settle_total'])/T_PIXEL*100:.1f}%")
    print(f"\n  FOM = {optimal['FOM']/1e9:.2f} x 10^9 W^-1 s^-1\n")
    
    return optimal

if __name__ == "__main__":
    result = optimize_fom()
    if result:
        print("="*80)
        print("Optimization complete!")
        print("="*80)
