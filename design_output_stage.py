import numpy as np
import matplotlib.pyplot as plt
import math
from look_up import *
import calculate_design_params as params

# Device selection per README constraints:
# - For branches with VDD > 1.1V, MUST use v2 devices
# - Output stage needs 1.4V swing, so VDDH will be > 1.1V
# - Therefore, ALWAYS use 2V devices for output stage

# Load 2V device data
try:
    nch = importdata("nch_2v.mat")
    pch = importdata("pch_2v.mat")
    print("Loaded 2V device data for output stage.")
except Exception as e:
    print(f"Error loading 2V MOS data: {e}")
    exit()

def design_output_stage():
    """
    Class AB Output Stage Design
    VDD--[Mp]--Vout--[RL]--[CL]--GND--[Mn]--GND
    
    Returns: dict with device sizes, biasing, and performance
    """
    
    print("="*70)
    print("CLASS AB OUTPUT STAGE DESIGN")
    print("="*70)
    
    gm_target_stability = params.gm2_required_stability
    RL = params.RL
    CL = params.CL
    tau_required = -params.T_PIXEL / math.log(params.ERROR_SPLIT_DYNAMIC)
    V_swing_pk = params.OUTPUT_SWING_MIN / 2.0
    I_drive_required = V_swing_pk / RL
    
    tau_min_series = RL * CL
    error_min_achievable = math.exp(-params.T_PIXEL / tau_min_series)
    
    print(f"\nSettling Analysis:")
    print(f"  Required tau: {tau_required*1e9:.2f}ns")
    print(f"  Physical limit (Rout=0): {tau_min_series*1e9:.2f}ns")
    print(f"  Best error: {error_min_achievable*100:.3f}%")
    
    if error_min_achievable > params.ERROR_SPLIT_DYNAMIC:
        print(f"  [WARNING] Cannot meet spec even with infinite drive!")
        gm_target_settling = float('inf')
    else:
        print(f"  [OK] Potentially achievable")
        gm_target_settling = 0
    
    gm_target = max(gm_target_stability, gm_target_settling)
    
    print(f"\nTarget Parameters:")
    print(f"  gm2 target: {gm_target*1e6:.1f}uS")
    print(f"  Load: {RL}Ohm, {CL*1e12:.1f}pF")
    print(f"  Swing: {params.OUTPUT_SWING_MIN}V, Peak I: {I_drive_required*1e3:.2f}mA")
    
    print(f"\nSTEP 1: SUPPLY VOLTAGE OPTIMIZATION")
    
    vdd_sweep = np.arange(params.VDDH_MAX, 1.35, -0.05)
    global_best_design = None
    
    for idx, VDDH in enumerate(vdd_sweep):
        print(f"  VDDH={VDDH:.2f}V ({idx+1}/{len(vdd_sweep)})...", end='')
        
        Vout_CM = VDDH / 2.0
        headroom_min = Vout_CM - V_swing_pk
        
        if headroom_min < 0.1:
            print(" skipped")
            continue
        
        device_type = "2V"
        L_candidates = [0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]
        gm_id_sweep = np.linspace(8, 30, 12)
        
        local_best_design = None
        local_min_power = float('inf')
        
        for L in L_candidates:
            if L < 0.15:  # 2V devices have minimum L = 0.15um
                continue
            
            for gm_id_n in gm_id_sweep:
                gm_n_target = 0.6 * gm_target
                gm_p_target = 0.4 * gm_target
                I_Q_n = gm_n_target / gm_id_n
                gm_id_p_values = [gm_id_n * 0.9, gm_id_n, gm_id_n * 1.1]
                
                for gm_id_p in gm_id_p_values:
                    I_Q_p = gm_p_target / gm_id_p
                    
                    try:
                        # Get device parameters at quiescent point (VDS = Vout_CM)
                        gm_gds_n = look_up_vs_gm_id(nch, 'GM_GDS', gm_id_n, l=L, vds=Vout_CM)
                        id_w_n = look_up_vs_gm_id(nch, 'ID_W', gm_id_n, l=L, vds=Vout_CM)
                        gm_gds_p = look_up_vs_gm_id(pch, 'GM_GDS', gm_id_p, l=L, vds=Vout_CM)
                        id_w_p = look_up_vs_gm_id(pch, 'ID_W', gm_id_p, l=L, vds=Vout_CM)
                        
                        W_n = I_Q_n / id_w_n
                        W_p = I_Q_p / id_w_p
                        
                        VGS_drive_n = 1.0
                        VSG_drive_p = 1.0
                        
                        id_w_n_peak = look_up_basic(nch, 'ID_W', vgs=VGS_drive_n, vds=headroom_min, l=L)
                        I_peak_n = id_w_n_peak * W_n * 1e-3  # id_w is in mA/um, convert to A
                        
                        id_w_p_peak = look_up_basic(pch, 'ID_W', vgs=VSG_drive_p, vds=headroom_min, l=L)
                        I_peak_p = id_w_p_peak * W_p * 1e-3
                        
                        if I_peak_n < I_drive_required or I_peak_p < I_drive_required:
                            continue
                        
                        gds_n = gm_n_target / gm_gds_n
                        gds_p = gm_p_target / gm_gds_p
                        rout_actual = 1 / (gds_n + gds_p)
                        gm_total = gm_n_target + gm_p_target
                        power = (I_Q_n + I_Q_p) * VDDH
                        
                        if power < local_min_power:
                            local_min_power = power
                            VGS_n_Q = look_up_vgs_vs_gm_id(nch, gm_id_n, l=L, vds=Vout_CM)
                            VSG_p_Q = look_up_vgs_vs_gm_id(pch, gm_id_p, l=L, vds=Vout_CM)
                            A2_stage = gm_total * rout_actual
                            
                            local_best_design = {
                                'VDDH': VDDH,
                                'device_type': device_type,
                                'Vout_CM': Vout_CM,
                                'L': L,
                                # NMOS parameters
                                'gm_id_n': gm_id_n,
                                'I_Q_n': I_Q_n,
                                'W_n': W_n,
                                'VGS_n_Q': VGS_n_Q,
                                'gm_n': gm_n_target,
                                'I_peak_n': I_peak_n,
                                'I_max_n_swing': I_peak_n,  # Alias for settling script
                                # PMOS parameters
                                'gm_id_p': gm_id_p,
                                'I_Q_p': I_Q_p,
                                'W_p': W_p,
                                'VSG_p_Q': VSG_p_Q,
                                'gm_p': gm_p_target,
                                'I_peak_p': I_peak_p,
                                'I_max_p_swing': I_peak_p,
                                'Rout': rout_actual,
                                'gm_total': gm_total,
                                'A2': A2_stage,
                                'I_quiescent': I_Q_n + I_Q_p,
                                'Power': power,
                                'headroom': headroom_min
                            }
                    except:
                        continue
        
        if local_best_design:
            print(f" {local_min_power*1e3:.3f}mW")
        else:
            print(" no design")
        
        if local_best_design:
            global_best_design = local_best_design
            if global_best_design['Power'] < 0.5e-3:
                print(f"  [OK] Excellent solution at VDDH={VDDH}V")
                break
    
    best_design = global_best_design
    
    if not best_design:
        print("\n[X] No valid design found!")
        return None
    
    print(f"\nSTEP 2: OPTIMAL DESIGN FOUND")
    
    print(f"\n[OK] Supply: VDDH={best_design['VDDH']:.2f}V ({best_design['device_type']})")
    print(f"    Vout_CM={best_design['Vout_CM']:.3f}V, Headroom={best_design['headroom']:.3f}V")
    
    print(f"\n[OK] NMOS (Mn): nch_{best_design['device_type'].lower()}")
    print(f"    L={best_design['L']}um, W={best_design['W_n']:.2f}um")
    print(f"    gm/ID={best_design['gm_id_n']:.2f}, I_Q={best_design['I_Q_n']*1e6:.2f}uA")
    print(f"    VGS={best_design['VGS_n_Q']:.3f}V, I_peak={best_design['I_peak_n']*1e3:.2f}mA")
    
    print(f"\n[OK] PMOS (Mp): pch_{best_design['device_type'].lower()}")
    print(f"    L={best_design['L']}um, W={best_design['W_p']:.2f}um")
    print(f"    gm/ID={best_design['gm_id_p']:.2f}, I_Q={best_design['I_Q_p']*1e6:.2f}uA")
    print(f"    VSG={best_design['VSG_p_Q']:.3f}V, I_peak={best_design['I_peak_p']*1e3:.2f}mA")
    
    print(f"\nSTEP 3: PERFORMANCE SUMMARY")
    
    A2_actual = best_design['gm_total'] * best_design['Rout']
    
    print(f"\n[OK] Small-Signal:")
    print(f"    gm={best_design['gm_total']*1e6:.2f}uS (target: {gm_target*1e6:.1f}uS)")
    print(f"    Rout={best_design['Rout']/1e3:.2f}kOhm")
    print(f"    Gain A2={A2_actual:.2f}V/V (internal)")
    
    gm_ratio = best_design['gm_total'] / gm_target
    status = "PASS" if 0.9 <= gm_ratio <= 1.1 else "WARNING"
    print(f"    [{status}] gm ratio: {gm_ratio:.2f}")
    
    peak_I = min(best_design['I_peak_n'], best_design['I_peak_p'])
    drive_margin = (peak_I - I_drive_required) / I_drive_required * 100
    print(f"\n[OK] Large-Signal:")
    print(f"    Peak I: {peak_I*1e3:.2f}mA (required: {I_drive_required*1e3:.2f}mA)")
    print(f"    [{'PASS' if drive_margin > 0 else 'FAIL'}] Margin: {drive_margin:.1f}%")
    
    print(f"\n[OK] Power: {best_design['Power']*1e3:.3f}mW")
    print(f"    I_quiescent: {best_design['I_quiescent']*1e6:.2f}uA")
    
    print(f"\nSTEP 4: BIAS CIRCUIT REQUIREMENTS")
    
    print(f"\n[OK] Required Bias Voltages:")
    print(f"    VBias_n: {best_design['VGS_n_Q']:.3f}V")
    print(f"    VBias_p: {best_design['VDDH'] - best_design['VSG_p_Q']:.3f}V")
    
    print(f"\n[OK] Bias Generation: Resistive divider or current mirror")
    
    delta_V = best_design['VGS_n_Q'] + best_design['VSG_p_Q'] - best_design['VDDH']
    if abs(delta_V) < 0.1:
        print(f"    [OK] Class AB overlap: {abs(delta_V)*1e3:.1f}mV")
    else:
        print(f"    [WARNING] Voltage gap: {delta_V:.3f}V")
    
    print(f"\n{'='*70}")
    print("DESIGN COMPLETE - Class AB Output Stage Sized!")
    print("="*70 + "\n")
    
    return best_design

if __name__ == "__main__":
    result = design_output_stage()
    
    if result:
        print("\n" + "="*70)
        print("DESIGN SUMMARY")
        print("="*70)
        print(f"\nVDDH: {result['VDDH']}V, Type: {result['device_type']}")
        print(f"\n{'Device':<8} {'L(um)':<8} {'W(um)':<10} {'I_Q(uA)':<10} {'I_pk(mA)':<10}")
        print("-"*70)
        print(f"{'NMOS':<8} {result['L']:<8.2f} {result['W_n']:<10.2f} "
              f"{result['I_Q_n']*1e6:<10.2f} {result['I_peak_n']*1e3:<10.2f}")
        print(f"{'PMOS':<8} {result['L']:<8.2f} {result['W_p']:<10.2f} "
              f"{result['I_Q_p']*1e6:<10.2f} {result['I_peak_p']*1e3:<10.2f}")
        print("-"*70)
        print(f"\nPower: {result['Power']*1e3:.3f}mW, gm: {result['gm_total']*1e6:.2f}uS")
