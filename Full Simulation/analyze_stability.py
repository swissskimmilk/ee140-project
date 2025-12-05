"""
Stability Analysis for Miller-Compensated Two-Stage Amplifier
Two-pole + one-zero open-loop model (A_ol), with optional nulling resistor.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import prelim_design_params as params

# ==============================================================================
# DESIGN-LEVEL PARAMETERS (FROM YOUR SIZING SCRIPTS)
# ==============================================================================
# NOTE: These are default values for standalone execution.
# When called from design_report.py, these are overridden with actual design values.

# Transconductances (from gm/ID / operating point)
gm1 = params.gm1_required              # Stage 1 transconductance (S)
# gm2_required_stability removed - will be set by design_report.py with actual gm2
# Default to a placeholder value (will be overridden)
try:
    gm2 = params.gm2_required_stability    # Stage 2 transconductance (S) - OLD, deprecated
except AttributeError:
    # If gm2_required_stability doesn't exist, use a default placeholder
    # This will be overridden by design_report.py with actual gm2 from design
    gm2 = 1e-3  # Placeholder - will be overridden

# Stage gains (FEED THESE IN FROM YOUR OTHER SCRIPTS)
A1 = params.FIRST_STAGE_GAIN                 # Stage 1 DC gain (V/V)
A2 = params.SECOND_STAGE_GAIN              # Stage 2 DC gain (V/V)
A0 = A1 * A2                           # Total DC gain (open-loop, V/V)

# Compensation / load
CC = params.CC                         # Miller compensation capacitor (F)
CL = params.CL                         # Load capacitance seen at output node (F)
RL = params.RL                         # External load resistance (Ohm) - used only for printing

# Nulling resistor options (will be overridden by design_report.py with actual values)
try:
    Rz_infinity = params.Rz_infinity       # Rz that pushes zero to infinity (~1/gm2)
    Rz_cancel = params.Rz_cancel_p2        # Rz that approximately cancels p2
except AttributeError:
    # If Rz values don't exist, use placeholders (will be overridden)
    Rz_infinity = 100.0  # Placeholder
    Rz_cancel = 200.0   # Placeholder

# Target specs
f_u_target = params.f_u_required       # Target unity gain frequency (Hz) for loop
PM_target = params.PHASE_MARGIN_MIN    # Target phase margin (deg)

# Crude parasitic estimate at node 1 (kept for info only)
Cdb1 = 0.1e-12
Cgs2 = 0.2e-12
C1 = Cdb1 + Cgs2

print(f"{'='*70}")
print(f"{'STABILITY ANALYSIS - MILLER COMPENSATED AMPLIFIER (OPEN-LOOP MODEL)':^70}")
print(f"{'='*70}\n")

print(f"--- Amplifier Parameters (from sizing scripts) ---")
print(f"Stage 1 gm1:              {gm1*1e6:8.2f} uS")
print(f"Stage 2 gm2:              {gm2*1e6:8.2f} uS")
print(f"Stage 1 Gain A1:          {A1:8.1f} V/V ({20*math.log10(A1):6.1f} dB)")
print(f"Stage 2 Gain A2:          {A2:8.1f} V/V ({20*math.log10(A2):6.1f} dB)")
print(f"Total DC Gain A0:         {A0:8.1f} V/V ({20*math.log10(A0):6.1f} dB)")
print(f"\nCompensation Capacitor CC: {CC*1e12:8.2f} pF")
print(f"Load Capacitor CL:        {CL*1e12:8.2f} pF")
print(f"Load Resistance RL:       {RL/1e3:8.2f} kOhm (informational only)")
print(f"Parasitic Cap at Node 1:  {C1*1e12:8.2f} pF (crude estimate)\n")


# ==============================================================================
# POLES / ZERO CALCULATION
# ==============================================================================

def calculate_poles_zeros(Rz=0.0):
    """
    Calculate poles and zero for Miller-compensated two-stage open-loop amplifier.

    Model:
        A_ol(s) = A0 * (1 ± s/wz) / ((1 + s/wp1)(1 + s/wp2))

    Approximations:
        - Unity-gain freq (amp alone):
              f_u_ol ≈ gm1 / (2π * CC)
        - Dominant pole:
              |p1| ≈ ω_u_ol / A0  (single-pole response up to f_u_ol)
        - Non-dominant pole at output:
              |p2| ≈ gm2 / CL
        - Zero from CC and Rz:
              ω_z = gm2 / [CC * (1 - gm2 * Rz)]
    """

    # Unity-gain freq of amplifier alone (open-loop model, no feedback factor)
    f_u_ol = gm1 / (2 * math.pi * CC)
    w_u_ol = 2 * math.pi * f_u_ol

    # Dominant pole: approximate from fu and DC gain
    # |p1| ≈ w_u_ol / A0
    wp1 = w_u_ol / max(A0, 1.0)  # protect against A0 <= 0
    p1 = -wp1

    # Non-dominant pole at output
    # Simple gm2/CL estimate
    wp2 = gm2 / max(CL, 1e-30)
    p2 = -wp2

    # Zero from Rz / CC
    if Rz == 0.0:
        # Pure Miller: RHP zero
        wz = gm2 / max(CC, 1e-30)
        z1 = +wz
        z1_type = "RHP"
    else:
        # General expression: ωz = gm2 / [CC * (1 - gm2 * Rz)]
        denom = 1.0 - gm2 * Rz
        if abs(denom) < 1e-18:
            # Zero at infinity
            z1 = float('inf')
            z1_type = "Infinity"
        else:
            wz = gm2 / (CC * denom)
            if wz > 0:
                # RHP zero
                z1 = +wz
                z1_type = "RHP"
            else:
                # LHP zero
                z1 = -wz   # store magnitude as positive
                z1_type = "LHP"

    # Convert to Hz
    f_p1 = abs(p1) / (2 * math.pi)
    f_p2 = abs(p2) / (2 * math.pi)
    if z1 == float('inf'):
        f_z1 = float('inf')
    else:
        f_z1 = abs(z1) / (2 * math.pi)

    return {
        "p1": p1,
        "p2": p2,
        "z1": z1,
        "z1_type": z1_type,
        "f_p1": f_p1,
        "f_p2": f_p2,
        "f_z1": f_z1,
        "f_u_ol": f_u_ol,
    }


# ==============================================================================
# BODE PLOT GENERATION (OPEN-LOOP A_ol)
# ==============================================================================

def generate_bode_plot(Rz=0.0, plot_title=""):
    """
    Generate Bode plots of the open-loop amplifier A_ol(s).

    A_ol(s) = A0 * (1 ± s/wz) / ((1 + s/wp1)(1 + s/wp2))

    NOTE: This is **not** including the feedback factor β.
          It's a shape / intuition tool for A_ol only.
    """

    results = calculate_poles_zeros(Rz)

    # Frequency range: 1 Hz to 10 GHz
    f_min = 1.0
    f_max = 1e10
    f = np.logspace(np.log10(f_min), np.log10(f_max), 10000)
    s = 2j * np.pi * f

    # Poles / zero
    p1 = results["p1"]
    p2 = results["p2"]
    z1 = results["z1"]
    z1_type = results["z1_type"]

    wp1 = abs(p1)
    wp2 = abs(p2)

    # Build A_ol(s)
    if z1_type == "Infinity":
        # No finite zero
        H = A0 / ((1 + s/wp1) * (1 + s/wp2))
    elif z1_type == "RHP":
        wz = abs(z1)
        H = A0 * (1 - s/wz) / ((1 + s/wp1) * (1 + s/wp2))
    else:  # LHP
        wz = abs(z1)
        H = A0 * (1 + s/wz) / ((1 + s/wp1) * (1 + s/wp2))

    # Magnitude / phase
    mag_dB = 20 * np.log10(np.abs(H))
    phase_rad = np.unwrap(np.angle(H))
    phase_deg = phase_rad * 180.0 / np.pi

    f_p1 = results["f_p1"]
    f_p2 = results["f_p2"]
    f_z1 = results["f_z1"]
    f_u_ol = results["f_u_ol"]

    # Find unity-gain point of A_ol(s) (0 dB crossing, approx)
    idx_unity = np.argmin(np.abs(mag_dB))  # closest to 0 dB
    f_unity = f[idx_unity]
    phase_at_unity = phase_deg[idx_unity]
    phase_margin = 180.0 + phase_at_unity  # assuming closed-loop cross at same fu

    # Gain margin: freq where phase ~ -180°
    idx_180 = np.argmin(np.abs(phase_deg + 180.0))
    f_180 = f[idx_180]
    gain_at_180 = mag_dB[idx_180]
    gain_margin = -gain_at_180

    # Console report
    print(f"--- Stability (Open-Loop Model) {plot_title} ---")
    print(f"Dominant Pole  p1: {f_p1/1e3:8.2f} kHz")
    print(f"Non-dominant p2: {f_p2/1e6:8.2f} MHz")
    if f_z1 != float("inf"):
        print(f"Zero z1:         {f_z1/1e6:8.2f} MHz ({z1_type})")
    else:
        print(f"Zero z1:         Infinity")
    print(f"Unity-gain fu(A_ol): {f_unity/1e6:8.2f} MHz")
    print(f"(Rough fu_target for loop): {f_u_target/1e6:8.2f} MHz")
    print(f"Phase Margin (approx): {phase_margin:5.1f} deg  [target {PM_target} deg]")
    print(f"Gain Margin  (approx): {gain_margin:5.1f} dB")

    if phase_margin >= PM_target:
        stability_status = "STABLE [OK]"
    elif phase_margin >= 45.0:
        stability_status = "MARGINAL (needs improvement)"
    else:
        stability_status = "UNSTABLE [FAIL]"
    print(f"Status (open-loop model): {stability_status}\n")

    # Plot
    A0_dB = 20 * math.log10(abs(A0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Magnitude
    ax1.semilogx(f, mag_dB, linewidth=2, label="|A_ol(jω)|")
    ax1.axhline(0, color="r", linestyle="--", linewidth=1.5, label="0 dB")
    ax1.axvline(f_unity, color="g", linestyle="--", linewidth=1.0,
                label=f"fu ≈ {f_unity/1e6:.2f} MHz")

    # Mark poles / zero
    ax1.plot(f_p1, A0_dB - 3.0, "ro", markersize=8,
             label=f"p1: {f_p1/1e3:.1f} kHz")
    ax1.plot(f_p2, mag_dB[np.argmin(np.abs(f - f_p2))], "mo", markersize=8,
             label=f"p2: {f_p2/1e6:.1f} MHz")
    if f_z1 != float("inf"):
        ax1.plot(f_z1, mag_dB[np.argmin(np.abs(f - f_z1))], "go", markersize=8,
                 label=f"z1: {f_z1/1e6:.1f} MHz ({z1_type})")

    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title(f"Open-Loop Bode Magnitude {plot_title}", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_ylim([-60, A0_dB + 20])

    # Phase
    ax2.semilogx(f, phase_deg, linewidth=2, label="∠A_ol(jω)")
    ax2.axhline(-180, color="r", linestyle="--", linewidth=1.5, label="-180°")
    ax2.axvline(f_unity, color="g", linestyle="--", linewidth=1.0,
                label=f"fu ≈ {f_unity/1e6:.2f} MHz")

    # Mark phase margin
    ax2.plot(f_unity, phase_at_unity, "go", markersize=10)
    ax2.annotate(f"PM ≈ {phase_margin:.1f}°",
                 xy=(f_unity, phase_at_unity),
                 xytext=(f_unity*3, phase_at_unity+20),
                 arrowprops=dict(arrowstyle="->", lw=2, color="green"))

    # Mark LHP zero "phase bump" if present
    if z1_type == "LHP" and f_z1 != float("inf"):
        z_idx = np.argmin(np.abs(f - f_z1))
        ax2.plot(f_z1, phase_deg[z_idx], "mo", markersize=8)
        ax2.annotate("LHP zero\nphase lead",
                     xy=(f_z1, phase_deg[z_idx]),
                     xytext=(f_z1*0.4, phase_deg[z_idx]-20),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="purple"),
                     fontsize=10, color="purple")

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_title(f"Open-Loop Bode Phase {plot_title}", fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.set_ylim([-270, 0])

    plt.tight_layout()

    return fig, results, {
        "f_unity": f_unity,
        "phase_margin": phase_margin,
        "gain_margin": gain_margin,
        "stability_status": stability_status,
    }


# ==============================================================================
# ANALYSIS SCENARIOS
# ==============================================================================

print("\n" + "="*70)
print("SCENARIO 1: NO NULLING RESISTOR (Rz = 0)")
print("="*70)
fig1, results1, metrics1 = generate_bode_plot(Rz=0.0, plot_title="(No Rz)")
plt.savefig("stability_bode_no_rz.png", dpi=300, bbox_inches="tight")
print("Bode plot saved as 'stability_bode_no_rz.png'")

print("\n" + "="*70)
print("SCENARIO 2: Rz ≈ 1/gm2 (zero at infinity)")
print("="*70)
fig2, results2, metrics2 = generate_bode_plot(Rz=Rz_infinity,
                                              plot_title=f"(Rz ≈ 1/gm2 ≈ {Rz_infinity:.1f} Ω)")
plt.savefig("stability_bode_rz_infinity.png", dpi=300, bbox_inches="tight")
print("Bode plot saved as 'stability_bode_rz_infinity.png'")

print("\n" + "="*70)
print("SCENARIO 3: Rz chosen to approx cancel p2")
print("="*70)
fig3, results3, metrics3 = generate_bode_plot(Rz=Rz_cancel,
                                              plot_title=f"(Rz_cancel ≈ {Rz_cancel:.1f} Ω)")
plt.savefig("stability_bode_rz_cancel.png", dpi=300, bbox_inches="tight")
print("Bode plot saved as 'stability_bode_rz_cancel.png'")

# ==============================================================================
# COMPARISON TABLE
# ==============================================================================

print("\n" + "="*70)
print("SCENARIO COMPARISON (OPEN-LOOP MODEL)")
print("="*70)
print(f"{'Scenario':<30} {'fu (MHz)':<12} {'PM (deg)':<12} {'GM (dB)':<12} {'Status':<20}")
print("-"*70)
print(f"{'No Rz':<30}"
      f"{metrics1['f_unity']/1e6:<12.2f}"
      f"{metrics1['phase_margin']:<12.1f}"
      f"{metrics1['gain_margin']:<12.1f}"
      f"{metrics1['stability_status']:<20}")
print(f"{'Rz ≈ 1/gm2':<30}"
      f"{metrics2['f_unity']/1e6:<12.2f}"
      f"{metrics2['phase_margin']:<12.1f}"
      f"{metrics2['gain_margin']:<12.1f}"
      f"{metrics2['stability_status']:<20}")
print(f"{'Rz cancel p2':<30}"
      f"{metrics3['f_unity']/1e6:<12.2f}"
      f"{metrics3['phase_margin']:<12.1f}"
      f"{metrics3['gain_margin']:<12.1f}"
      f"{metrics3['stability_status']:<20}")
print("="*70)


# ==============================================================================
# SENSITIVITY: CC VS STABILITY (KEEP fu_target FIXED)
# ==============================================================================

print("\n" + "="*70)
print("SENSITIVITY ANALYSIS - CC FOR GIVEN fu_target")
print("="*70)

def analyze_cc_sensitivity():
    """
    Analyze how changing CC (with gm1, gm2 fixed to current design)
    affects fu, p2/fu, and a rough PM estimate.

    This answers:
      - If I shrink CC, fu increases but p2/fu drops => PM drops.
      - If I grow CC, fu decreases but p2/fu increases => PM improves.
    """

    # Use current gm1, gm2 as designed
    gm1_fixed = gm1
    gm2_fixed = gm2

    # Sweep CC around your current value (0.8 pF)
    CC_values = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.5]) * 1e-12  # F
    results_table = []

    for cc_test in CC_values:
        # Open-loop fu for this CC (no feedback factor)
        f_u_test = gm1_fixed / (2 * math.pi * cc_test)

        # Dominant pole ~ fu / A0
        w_u_test = 2 * math.pi * f_u_test
        wp1_test = w_u_test / max(A0, 1.0)
        f_p1_test = wp1_test / (2 * math.pi)

        # Non-dominant pole
        wp2_test = gm2_fixed / max(CL, 1e-30)
        f_p2_test = wp2_test / (2 * math.pi)

        # Ratio p2 / fu
        ratio = f_p2_test / f_u_test

        # Rough two-pole PM estimate (zero at infinity)
        pm_est = 90.0 - math.degrees(math.atan(f_u_test / f_p2_test))

        results_table.append({
            "CC_pF": cc_test * 1e12,
            "f_p1_kHz": f_p1_test / 1e3,
            "f_p2_MHz": f_p2_test / 1e6,
            "f_u_MHz": f_u_test / 1e6,
            "p2_fu_ratio": ratio,
            "PM_est": pm_est,
        })

    print(f"\n{'CC (pF)':<10} {'f_p1 (kHz)':<12} {'f_p2 (MHz)':<12} "
          f"{'fu (MHz)':<12} {'p2/fu':<10} {'PM (deg)':<10}")
    print("-" * 80)

    for r in results_table:
        status = "[OK]" if r["PM_est"] >= PM_target else ""
        print(f"{r['CC_pF']:<10.2f}"
              f"{r['f_p1_kHz']:<12.2f}"
              f"{r['f_p2_MHz']:<12.2f}"
              f"{r['f_u_MHz']:<12.2f}"
              f"{r['p2_fu_ratio']:<10.2f}"
              f"{r['PM_est']:<10.1f} {status}")

    print("\nRule of thumb: want p2/fu ≳ 2–3 and PM_est ≥ target.\n")

    # Pick CC with highest fu that still meets PM target
    feasible = [r for r in results_table if r["PM_est"] >= PM_target]
    if feasible:
        best = max(feasible, key=lambda r: r["f_u_MHz"])
        print(f"Suggested CC (toy model): {best['CC_pF']:.2f} pF")
        print(f"  -> fu ≈ {best['f_u_MHz']:.2f} MHz")
        print(f"  -> PM_est ≈ {best['PM_est']:.1f} deg")
    else:
        print("No CC in this sweep meets the PM target in this simple model.")



analyze_cc_sensitivity()


# ==============================================================================
# POLE-ZERO MAP (TEXT)
# ==============================================================================

print("\n" + "="*70)
print("POLE-ZERO SUMMARY (Rz ≈ 1/gm2 → zero at infinity)")
print("="*70)

results_map = calculate_poles_zeros(Rz=Rz_infinity)

print("\nPoles (x) / Zero (o) in frequency domain:")
print(f"  p1: {results_map['f_p1']/1e3:8.2f} kHz")
print(f"  p2: {results_map['f_p2']/1e6:8.2f} MHz")
if results_map["f_z1"] != float("inf"):
    print(f"  z1: {results_map['f_z1']/1e6:8.2f} MHz ({results_map['z1_type']})")
else:
    print(f"  z1: Infinity (with Rz ≈ 1/gm2)")

print(f"\nPole separation p2/p1: {results_map['f_p2']/results_map['f_p1']:.1f}×")
print("For classic Miller comp, you usually want p2/p1 >> 10 for generous PM.\n")

print("="*70)
print("Analysis (open-loop model) complete.")
print("="*70)

plt.show()
