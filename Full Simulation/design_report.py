# -*- coding: utf-8 -*-
"""
COMPREHENSIVE DESIGN REPORT
==========================
LCD Driver Amplifier - Two-Stage Miller-Compensated Design

This script provides a complete evaluation of the amplifier design:
1. Design Parameters Summary
2. Stage 1 & 2 Designs
3. Stability Analysis (Bode Plots & Phase Margin)
4. Settling Time Analysis
5. Power Dissipation & Figure of Merit
6. Specification Compliance Check

Pipeline:
1. Start with prelim_design_params.py to calculate required parameters
2. Find optimal design for telescopic first stage
3. Find optimal design for AB class output stage
4. Use design data in calculate_settling and analyze_stability
"""

import sys
import io
# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import math

# Import config first to check plot settings
import config
# Set matplotlib backend before importing pyplot
# Use non-interactive backend if plots are disabled to prevent them from showing
if not config.GENERATE_STABILITY_PLOTS and not config.GENERATE_SETTLING_PLOTS:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend - plots won't show
import matplotlib.pyplot as plt

# Import other design modules
import prelim_design_params as params
import telescopic_combined_fixed_cm as stage1
import class_ab_output as stage2
import calculate_settling
import analyze_stability
import design_helpers as helpers

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def print_pass_fail(condition, pass_msg, fail_msg):
    """Print pass/fail status"""
    if condition:
        print(f"  [OK] [PASS] {pass_msg}")
    else:
        print(f"  [X] [FAIL] {fail_msg}")

print_header("LCD DRIVER AMPLIFIER DESIGN REPORT")

# ==============================================================================
# SECTION 1: DESIGN SPECIFICATIONS & PARAMETERS
# ==============================================================================
print_header("SECTION 1: DESIGN SPECIFICATIONS & PARAMETERS")

# First, run the preliminary design parameter calculations
print("\n--- Running Preliminary Design Parameter Calculations ---")
params.print_design_report()

print("\n--- Target Specifications ---")
print(f"Closed-Loop Gain: {params.G_CLOSED_LOOP} V/V")
print(f"Output Swing Required: {params.OUTPUT_SWING_MIN} V")
print(f"Load: {params.RL} Ohm || {params.CL*1e12:.1f} pF")
print(f"Total Error Budget: {params.TOTAL_ERROR_SPEC*100:.3f}%")
print(f"  - Static Error: {helpers.format_error_fraction(params.ERROR_STATIC, 3)}")
print(f"  - Dynamic Error: {helpers.format_error_fraction(params.ERROR_DYNAMIC, 3)}")
print(f"Settling Time Required: {params.T_PIXEL*1e9:.1f} ns")
print(f"Phase Margin Target: {params.PHASE_MARGIN_MIN} deg")
print(f"Maximum Power: {params.POWER_MAX*1e3:.2f} mW")
print(f"Supply Voltages: VDD_L={params.VDDL_MAX}V, VDD_H={params.VDDH_MAX}V")

print("\n--- Calculated Design Parameters ---")
print(f"Required Open-Loop Gain (A0): {params.A0_open_loop_required:.1f} V/V "
      f"({helpers.gain_to_dB(params.A0_open_loop_required):.1f} dB)")
print(f"Required Unity Gain Frequency (f_u): {helpers.format_frequency_Hz(params.f_u_required)}")
print(f"Required Closed-Loop 3dB Bandwidth: {helpers.format_frequency_Hz(params.f_3db_cl_required)}")
print(f"Miller Compensation Capacitor (CC): {params.CC*1e12:.2f} pF")
print(f"Required Stage 1 Transconductance (gm1): {params.gm1_required*1e6:.2f} uS")
print(f"Required Stage 2 Transconductance (gm2): {params.gm2_required_stability*1e6:.2f} uS")

print("\n--- Nulling Resistor Options ---")
print(f"Rz (Zero at Infinity): {params.Rz_infinity:.1f} Ohm")
print(f"Rz (Cancel p2): {params.Rz_cancel_p2:.1f} Ohm")

# ==============================================================================
# SECTION 2: STAGE DESIGNS
# ==============================================================================
print_header("SECTION 2: AMPLIFIER STAGE DESIGNS")

# --- Stage 2 Design (run first to determine required input CM for Stage 1) ---
print("\n" + "-"*80)
print("STAGE 2: CLASS AB OUTPUT STAGE")
print("-"*80)

stage2_design = stage2.design_output_stage()

if not stage2_design:
    raise RuntimeError("Stage 2 Design Failed! Cannot proceed without stage 2 design.")

print("\n[OK] Stage 2 Design Successful!")
print(f"  Total gm: {stage2_design['gm_total']*1e6:.2f} uS")
print(f"  Output Resistance: {helpers.format_resistance_Ohm(stage2_design['rout'])}")
print(f"  Quiescent Current: {stage2_design['Iq_total']*1e6:.2f} uA")
print(f"  Stage 2 Gain (A2): {stage2_design['A2']:.2f} V/V")
print(f"  Peak Drive Current: {stage2_design['I_pk_available']*1e3:.2f} mA")
print(f"  Required Input CM (Vgate_n): {stage2_design['Vgate_n']:.3f} V")

# Extract key Stage 2 parameters
gm2_actual = stage2_design['gm_total']
# For common-source: A2 already calculated in design (includes RL effect)
A2_actual = stage2_design['A2']
Rout2_actual = stage2_design['rout']
# Power: VDDH * Iq_total (DC quiescent power)
P_stage2 = params.VDDH_MAX * stage2_design['Iq_total']
I_max = stage2_design['I_pk_available']
VDDH_actual = params.VDDH_MAX

# Get actual output swing from the design
# The output stage design calculates this based on actual headroom constraints
swing_available = stage2_design['swing_actual']
swing_meets_spec = stage2_design['swing_meets_spec']
if not swing_meets_spec:
    print(f"\n[WARNING] Output stage design cannot achieve required swing!")
    print(f"  Required: {params.OUTPUT_SWING_MIN:.3f} V")
    print(f"  Achievable: {swing_available:.3f} V")

# --- Stage 1 Design (uses Stage 2 required input CM as its output CM target) ---
print("\n" + "-"*80)
print("STAGE 1: TELESCOPIC CASCODE DIFFERENTIAL AMPLIFIER")
print("-"*80)

Vout_target_stage1 = stage2_design["Vgate_n"]
print(f"  Target Stage-1 Output CM for Stage-2 input: {Vout_target_stage1:.3f} V")

# Call the design function with the required output common-mode
stage1_design = stage1.design_telescopic(
    Vout_target=Vout_target_stage1,
    mode=config.TELESCOPIC_MODE,
)

if not stage1_design:
    raise RuntimeError("Stage 1 Design Failed! Cannot proceed without stage 1 design.")

print("\n[OK] Stage 1 Design Successful!")
print(f"  Voltage Gain (A1): {stage1_design['A1_gain']:.1f} V/V ({helpers.gain_to_dB(stage1_design['A1_gain']):.1f} dB)")
print(f"  Output Resistance: {helpers.format_resistance_Ohm(stage1_design['Rout_total'])}")
print(f"  Branch Current: {stage1_design['Id_branch']*1e6:.2f} uA")
print(f"  Tail Current: {stage1_design['Id_branch']*2*1e6:.2f} uA")
print(f"  Common-Mode Voltage (inputs): {stage1_design['Vcm_in']:.3f} V")
print(f"  Output CM Voltage (Stage 1): {stage1_design['Vout']:.3f} V")

# Extract key Stage 1 parameters
# Calculate gm1 from the design: gm1 = gm_ID_in * Id_branch
gm1_actual = stage1_design['gm_ID_in'] * stage1_design['Id_branch']
A1_actual = stage1_design['A1_gain']
Rout1_actual = stage1_design['Rout_total']
I_tail_stage1 = stage1_design['Id_branch'] * 2  # Total tail current
# Calculate power using actual supply voltage from design
VDD_stage1 = stage1_design['VDD_eff']
P_stage1 = VDD_stage1 * I_tail_stage1

# ==============================================================================
# SECTION 3: STABILITY ANALYSIS
# ==============================================================================
print_header("SECTION 3: STABILITY ANALYSIS")

print("\n--- Transfer Function Parameters ---")
A0_actual = helpers.calculate_total_gain(A1_actual, A2_actual)
print(f"Stage 1 Gain (A1): {A1_actual:.1f} V/V ({helpers.gain_to_dB(A1_actual):.1f} dB)")
print(f"Stage 2 Gain (A2): {A2_actual:.2f} V/V ({helpers.gain_to_dB(A2_actual):.1f} dB)")
print(f"Total DC Gain (A0): {A0_actual:.1f} V/V ({helpers.gain_to_dB(A0_actual):.1f} dB)")

# Calculate static error from actual gain
loop_gain_actual = helpers.calculate_loop_gain(A0_actual, params.G_CLOSED_LOOP)
error_static_actual = helpers.calculate_static_error(A0_actual, params.G_CLOSED_LOOP)
print(f"\n--- Static Error from Actual Gain ---")
print(f"Loop Gain: {loop_gain_actual:.1f} V/V ({helpers.gain_to_dB(loop_gain_actual):.1f} dB)")
print(f"Static Error: {helpers.format_error_fraction(error_static_actual)} (from actual A0={A0_actual:.1f} V/V)")

# Calculate poles and zeros
CC = params.CC
CL = params.CL
p1 = helpers.calculate_dominant_pole(gm1_actual, A2_actual, CC)
p2 = helpers.calculate_non_dominant_pole(gm2_actual, CL)
f_p1 = helpers.calculate_pole_frequency(p1)
f_p2 = helpers.calculate_pole_frequency(p2)
f_u_actual = helpers.calculate_unity_gain_freq(gm1_actual, CC)

# Determine Rz value based on config setting
if config.RZ_SETTING == "infinity":
    Rz_value = params.Rz_infinity
    Rz_label = f"Rz = 1/gm2 = {Rz_value:.1f} Ohm (zero at infinity)"
elif config.RZ_SETTING == "cancel_p2":
    Rz_value = params.Rz_cancel_p2
    Rz_label = f"Rz = {Rz_value:.1f} Ohm (cancel p2)"
else:  # "none"
    Rz_value = 0.0
    Rz_label = "Rz = 0 (no nulling resistor)"

print(f"\n--- Pole/Zero Locations ({Rz_label}) ---")
print(f"Dominant Pole (p1): {helpers.format_frequency_Hz(f_p1)}")
print(f"Non-Dominant Pole (p2): {helpers.format_frequency_Hz(f_p2)}")
print(f"Unity Gain Frequency: {helpers.format_frequency_Hz(f_u_actual)}")
print(f"Pole Separation (p2/f_u): {helpers.calculate_pole_separation_ratio(f_p2, f_u_actual):.2f}x")

# Estimate phase margin (simplified for two-pole system)
pm_estimate = helpers.estimate_phase_margin(f_u_actual, f_p2)

print(f"\n--- Stability Metrics ---")
print(f"Estimated Phase Margin: {pm_estimate:.1f} deg")
print_pass_fail(pm_estimate >= params.PHASE_MARGIN_MIN,
                f"Phase Margin meets target ({params.PHASE_MARGIN_MIN} deg)",
                f"Phase Margin below target ({params.PHASE_MARGIN_MIN} deg)")

if pm_estimate < params.PHASE_MARGIN_MIN:
    print("  Recommendations:")
    print("    - Increase CC to improve pole separation")
    print("    - Increase gm2 to push p2 to higher frequency")
    print("    - Optimize Rz for zero cancellation")

# Generate stability plots if enabled
if config.GENERATE_STABILITY_PLOTS:
    print("\n--- Generating Stability Bode Plots ---")
    try:
        # Set module-level variables in analyze_stability for the functions to use
        analyze_stability.gm1 = gm1_actual
        analyze_stability.gm2 = gm2_actual
        analyze_stability.A1 = A1_actual
        analyze_stability.A2 = A2_actual
        analyze_stability.A0 = A0_actual
        analyze_stability.CC = CC
        analyze_stability.CL = CL
        analyze_stability.Rz_infinity = params.Rz_infinity
        analyze_stability.Rz_cancel_p2 = params.Rz_cancel_p2
        
        # Generate plot with selected Rz setting
        fig_stab, results_stab, metrics_stab = analyze_stability.generate_bode_plot(
            Rz=Rz_value,
            plot_title=f"(Rz setting: {config.RZ_SETTING})"
        )
        plt.savefig("stability_bode_design_report.png", dpi=300, bbox_inches="tight")
        print("  Stability Bode plot saved as 'stability_bode_design_report.png'")
        plt.close(fig_stab)
        # Explicitly clear any remaining figures to prevent them from showing
        plt.close('all')
    except Exception as e:
        print(f"  [WARNING] Could not generate stability plots: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')  # Make sure no plots are left open
else:
    print("\nNote: Set GENERATE_STABILITY_PLOTS=True in config.py to generate Bode plots")

# ==============================================================================
# SECTION 4: SETTLING TIME ANALYSIS
# ==============================================================================
print_header("SECTION 4: SETTLING TIME ANALYSIS")

print("\n--- Output Stage Drive Capability ---")
print(f"Maximum Drive Current: {I_max*1e3:.2f} mA")
print(f"Output Resistance: {Rout2_actual:.1f} Ohm")

# Calculate settling time
V_step = 0.7  # 700mV step (half of 1.4V swing)
error_tolerance = params.TOTAL_ERROR_SPEC

print(f"\n--- Settling Time Simulation ---")
print(f"Test Step Size: {V_step} V")
print(f"Error Tolerance: {error_tolerance*100:.3f}%")

try:
    # Calculate static error from actual gain (not from prelim budget)
    A0_actual_for_settling = helpers.calculate_total_gain(A1_actual, A2_actual)
    error_static_actual = helpers.calculate_static_error(A0_actual_for_settling, params.G_CLOSED_LOOP)
    
    # Get actual fu from stage 1 design
    fu_actual = stage1_design['fu_actual']
    
    # Pass all required parameters to analyze_settling
    result = calculate_settling.analyze_settling(
        V_step=V_step,
        I_max=I_max,
        R_out=Rout2_actual,
        R_L=params.RL,
        C_L=params.CL,
        error_total=error_tolerance,
        I_tail_stage1=I_tail_stage1,
        CC=params.CC,
        A2=A2_actual,
        error_static=error_static_actual,  # Use actual static error from real gain
        fu_actual=fu_actual,  # Use actual fu from design
        G_CLOSED_LOOP=params.G_CLOSED_LOOP  # Required to calculate f_3db_cl from fu_actual
    )
    
    # Use total settling time (includes CC slewing)
    t_settle = result['settling_time_total'] if 'settling_time_total' in result else result['settling_time']
    t_settle_rlcl = result['settling_time']
    t_slew_cc = result['t_slew_CC'] if 't_slew_CC' in result else 0.0
    
    if t_settle is not None:
        if t_slew_cc > 1e-9:
            print(f"\n[OK] Total Settling Time: {t_settle*1e9:.2f} ns")
            print(f"     (CC slewing: {t_slew_cc*1e9:.2f} ns + RL-CL: {t_settle_rlcl*1e9:.2f} ns)")
        else:
            print(f"\n[OK] Settling Time: {t_settle*1e9:.2f} ns")
        
        # Plot the response
        if config.GENERATE_SETTLING_PLOTS:
            calculate_settling.plot_settling_response(result, save=True, show=False)
            print("  Settling plot saved as 'settling_plot.png'")
        else:
            print("  (Set GENERATE_SETTLING_PLOTS=True in config.py to generate plot)")
        
except Exception as e:
    print(f"\n[X] Error in settling analysis: {e}")
    import traceback
    traceback.print_exc()
    raise

# Compare to spec
print(f"\n--- Specification Check ---")
print(f"Required Settling Time: {params.T_PIXEL*1e9:.1f} ns")
if t_settle != float('inf'):
    print(f"Actual Settling Time: {t_settle*1e9:.2f} ns")
    if t_settle <= params.T_PIXEL:
        margin = (params.T_PIXEL - t_settle) / params.T_PIXEL * 100
        print_pass_fail(True, f"Settling Time meets requirement ({margin:.1f}% margin)", "")
    else:
        shortfall = (t_settle - params.T_PIXEL) / params.T_PIXEL * 100
        print_pass_fail(False, "", f"Settling Time exceeds requirement ({shortfall:.1f}% over)")
        print("  Recommendation: Increase output stage gm2 (wider devices)")
else:
    print_pass_fail(False, "", "Did not settle")

# ==============================================================================
# SECTION 5: POWER DISSIPATION & FIGURE OF MERIT
# ==============================================================================
print_header("SECTION 5: POWER DISSIPATION & FIGURE OF MERIT")

print("\n--- Power Breakdown ---")
print(f"Stage 1 Power: {P_stage1*1e3:.3f} mW")
print(f"Stage 2 Power: {P_stage2*1e3:.3f} mW")

P_total = P_stage1 + P_stage2
print(f"\n--- Total Power Dissipation ---")
print(f"Total Power: {P_total*1e3:.3f} mW")
print(f"Power Budget: {params.POWER_MAX*1e3:.2f} mW")

if P_total <= params.POWER_MAX:
    margin = (params.POWER_MAX - P_total) / params.POWER_MAX * 100
    print_pass_fail(True, f"Power within budget ({margin:.1f}% margin)", "")
else:
    excess = (P_total - params.POWER_MAX) / params.POWER_MAX * 100
    print_pass_fail(False, "", f"Power exceeds budget ({excess:.1f}% over)")

# Calculate Figure of Merit
print(f"\n--- Figure of Merit ---")

# PRIMARY FOM: 1 / (Power x Settling_Time)
# This is the main optimization metric
# Use total settling time (includes CC slewing)
if t_settle < float('inf') and P_total > 0:
    FOM_primary = 1 / (P_total * t_settle)
    print(f"PRIMARY FOM = 1/(Power x Settling_Time)")
    if t_slew_cc > 1e-9:
        print(f"  FOM = 1/({P_total*1e3:.3f} mW x {t_settle*1e9:.2f} ns)")
        print(f"        [includes CC slewing: {t_slew_cc*1e9:.2f} ns]")
    else:
        print(f"  FOM = 1/({P_total*1e3:.3f} mW x {t_settle*1e9:.2f} ns)")
    print(f"  **FOM = {FOM_primary/1e9:.2f} x 10^9 W^-1 s^-1**")
    print()
else:
    FOM_primary = 0
    print(f"PRIMARY FOM: Cannot calculate (settling failed)")
    print()

# Additional FOMs for reference:
print(f"Additional Metrics:")

# FOM 1: Gain-Bandwidth Product per Power
GBW = f_u_actual
FOM_GBW = GBW / P_total
print(f"  GBW/Power: {FOM_GBW/1e6:.2f} MHz/mW")

# FOM 2: Small-Signal FOM
FOM_SS = GBW / (P_total * params.CL)
print(f"  Small-Signal FOM: {FOM_SS/1e15:.2f} x 10^15 Hz/(W·F)")

# ==============================================================================
# SECTION 6: SPECIFICATION COMPLIANCE SUMMARY
# ==============================================================================
print_header("SECTION 6: SPECIFICATION COMPLIANCE SUMMARY")

print("\n--- Requirements Checklist ---")

# Create a checklist
checks = []

# 1. DC Gain
checks.append(("DC Gain", 
               A0_actual >= params.A0_open_loop_required,
               f"{A0_actual:.1f} V/V",
               f">= {params.A0_open_loop_required:.1f} V/V"))

# 2. Unity Gain Frequency
checks.append(("Unity Gain Frequency",
               abs(f_u_actual - params.f_u_required) / params.f_u_required < 0.2,
               f"{helpers.format_frequency_Hz(f_u_actual)}",
               f"~{helpers.format_frequency_Hz(params.f_u_required)}"))

# 3. Phase Margin
checks.append(("Phase Margin",
               pm_estimate >= params.PHASE_MARGIN_MIN,
               f"{pm_estimate:.1f} deg",
               f">= {params.PHASE_MARGIN_MIN} deg"))

# 4. Settling Time
checks.append(("Settling Time",
               t_settle <= params.T_PIXEL if t_settle != float('inf') else False,
               f"{t_settle*1e9:.1f} ns" if t_settle != float('inf') else "N/A",
               f"<= {params.T_PIXEL*1e9:.1f} ns"))

# 5. Output Swing (calculated from actual design)
checks.append(("Output Swing",
               swing_meets_spec,
               f"{swing_available:.2f} V",
               f">= {params.OUTPUT_SWING_MIN:.1f} V"))

# 6. Power
checks.append(("Power Consumption",
               P_total <= params.POWER_MAX,
               f"{P_total*1e3:.3f} mW",
               f"<= {params.POWER_MAX*1e3:.2f} mW"))

# Print checklist table
print(f"\n{'Specification':<25} {'Status':<10} {'Actual':<20} {'Requirement':<20}")
print("-" * 75)

pass_count = 0
for spec_name, passed, actual, requirement in checks:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{spec_name:<25} {status:<10} {actual:<20} {requirement:<20}")
    if passed:
        pass_count += 1

print("-" * 75)
print(f"Overall: {pass_count}/{len(checks)} specifications met")

if pass_count == len(checks):
    print("\n*** [OK] DESIGN MEETS ALL SPECIFICATIONS! ***")
else:
    print(f"\n*** [X] DESIGN ISSUES: {len(checks) - pass_count} specification(s) not met ***")
    print("\nRecommendations for improvement:")
    
    for spec_name, passed, actual, requirement in checks:
        if not passed:
            if "Phase Margin" in spec_name:
                print(f"  - {spec_name}: Increase CC or optimize Rz")
            elif "Settling" in spec_name:
                print(f"  - {spec_name}: Increase output stage gm2 (wider devices)")
            elif "Power" in spec_name:
                print(f"  - {spec_name}: Reduce bias currents or optimize gm/ID")
            elif "Gain" in spec_name:
                print(f"  - {spec_name}: Increase stage 1 output resistance or gm")

# ==============================================================================
# SECTION 7: DESIGN SUMMARY
# ==============================================================================
print_header("SECTION 7: DESIGN SUMMARY")

print("\n--- Architecture ---")
print("Two-Stage Miller-Compensated Amplifier")
# Get telescopic mode from design
telescopic_mode_used = stage1_design['mode']
mode_description = "high-swing PMOS load" if telescopic_mode_used == "high_swing" else "self-biased diode load"
print(f"  - Stage 1: Telescopic cascode differential amplifier ({telescopic_mode_used} mode: {mode_description})")
print("  - Stage 2: Class AB common-source output stage")
# Get Rz setting description
if config.RZ_SETTING == "infinity":
    rz_description = f"Rz = 1/gm2 = {params.Rz_infinity:.1f} Ohm (zero at infinity)"
elif config.RZ_SETTING == "cancel_p2":
    rz_description = f"Rz = {params.Rz_cancel_p2:.1f} Ohm (cancel p2)"
else:
    rz_description = "Rz = 0 (no nulling resistor)"
print(f"  - Compensation: Miller capacitor (CC = {params.CC*1e12:.2f} pF) with {rz_description}")

print("\n--- Key Design Parameters ---")
print(f"Supply Voltages:")
# Get actual VDD used by stage 1 from design
VDD_stage1_actual = stage1_design['VDD_eff']
print(f"  - VDD_L (Stage 1): {VDD_stage1_actual:.2f} V (effective, from design)")
print(f"  - VDD_H (Stage 2): {VDDH_actual:.2f} V")
print(f"\nCompensation:")
print(f"  - Miller Capacitor (CC): {params.CC*1e12:.2f} pF")
print(f"  - Recommended Rz: {params.Rz_infinity:.1f} - {params.Rz_cancel_p2:.1f} Ohm")

print("\n--- Performance Summary ---")
print(f"DC Gain: {helpers.gain_to_dB(A0_actual):.1f} dB")
print(f"Unity Gain Frequency: {helpers.format_frequency_Hz(f_u_actual)}")
print(f"Phase Margin: {pm_estimate:.1f} deg")
if t_settle != float('inf'):
    print(f"Settling Time: {helpers.format_time_s(t_settle)}")
print(f"Total Power: {helpers.format_power_W(P_total)}")
if t_settle < float('inf') and P_total > 0:
    FOM_main = 1 / (P_total * t_settle)
    print(f"FOM: {FOM_main/1e9:.2f} x 10^9 W^-1 s^-1")

print("\n--- Next Steps ---")
print("1. Run full stability analysis: python analyze_stability.py")
print("2. Perform SPICE simulation to verify hand calculations")
print("3. Corner analysis (process, voltage, temperature)")
print("4. Layout design and parasitic extraction")
print("5. Verify CMRR, PSRR, and noise performance")

print_header("END OF REPORT")

print("\nGenerated/Updated Files:")
print("  [OK] settling_plot.png (settling time transient)")
print("  - Run analyze_stability.py for Bode plots")
print("  - Run individual stage scripts for detailed analysis")

print("\n[OK] Design report complete!")
