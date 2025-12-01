"""
LCD Driver Amplifier Design Configuration
===========================================

This file contains all designer-adjustable parameters for the amplifier design.
Modify these values to tune your design, then re-run the design scripts.

All scripts (prelim_design_params.py, telescopic_combined.py, class_ab_output.py, 
etc.) import from this file to ensure consistency.
"""

import math

# ==============================================================================
# 1. SYSTEM SPECIFICATIONS (from project handout)
# ==============================================================================
# These are the hard specs from the assignment. Only change if requirements change.

# Stage 1 supply bumped to high-rail (per updated headroom analysis)
VDDL_MAX = 1.1          # V - Stage 1 supply 
VDDH_MAX = 1.8          # V - Maximum high voltage supply
G_CLOSED_LOOP = 2       # V/V - Closed-loop gain requirement
RL = 1e3                # Ohms - Load resistance
CL = 25e-12             # Farads - Load capacitance
TOTAL_ERROR_SPEC = 0.002 # 0.2% - Total allowable error
POWER_MAX = 1.25e-3     # Watts - Maximum power consumption (EE140 spec)
PHASE_MARGIN_MIN = 60   # Degrees - Minimum phase margin (targeting 60 deg for robustness)
OUTPUT_SWING_MIN = 1.4  # V - Minimum required output swing

# ==============================================================================
# 2. TIMING SPECIFICATIONS
# ==============================================================================
# Display and timing requirements

REFRESH_RATE = 60       # Hz - Display refresh rate
T_PIXEL = 180e-9        # seconds - Time to settle each pixel (180 ns)

# ==============================================================================
# 3. PRELIMINARY DESIGN BUDGETS & ESTIMATES
# ==============================================================================
# These are used for initial design parameter calculations in prelim_design_params.py
# They are estimates/budgets that guide the design, but actual values come from
# the optimized designs.

ERROR_SPLIT_STATIC_RATIO = 0.5  # Fraction of non-RC error allocated to static error
                                 # (rest goes to dynamic error)

# --- Miller Compensation Capacitor ---
# This is a key design variable! Affects:
#  - Unity gain frequency: f_u = gm1 / (2*pi*CC)
#  - Required gm1 (and thus stage 1 power)
#  - Stability (pole separation)
# OPTIMIZED: Smaller CC reduces gm1 requirement and Stage 1 power
CC = 0.8e-12  # Farads - Miller compensation capacitor (0.8 pF)

# Estimates of parasitic capacitances and gains (used for preliminary calculations)
C_OUT_P = 2e-12         # Farads - Estimated output parasitic capacitance
FIRST_STAGE_GAIN = 200  # V/V - Estimated Stage 1 gain (actual from telescopic design may differ)
SECOND_STAGE_GAIN = 37  # V/V - Estimated Stage 2 gain (actual from output stage design may differ)
P2_FACTOR = 2           # Factor for non-dominant pole placement (p2 = P2_FACTOR * f_u)

# ==============================================================================
# 4. DESIGN SCRIPT SETTINGS
# ==============================================================================
# These control which design options are used when running the design scripts.

# Telescopic Stage 1 Mode
# Options: "high_swing" or "standard"
#   - "high_swing": High-swing PMOS load (top PMOS gate near Vout)
#   - "standard": Simple self-biased load (both PMOS devices diode-connected)
TELESCOPIC_MODE = "standard"

# Nulling Resistor (Rz) Setting
# Options: "infinity", "cancel_p2", or "none"
#   - "infinity": Rz = 1/gm2 (pushes zero to infinity)
#   - "cancel_p2": Rz chosen to approximately cancel p2
#   - "none": No nulling resistor (Rz = 0)
RZ_SETTING = "infinity"


# ==============================================================================
# 5. PLOTTING OPTIONS
# ==============================================================================
# Control whether design_report.py generates plots

GENERATE_STABILITY_PLOTS = False   # Generate Bode plots for stability analysis
GENERATE_SETTLING_PLOTS = False    # Generate settling time transient plots

# ==============================================================================
# VALIDATION
# ==============================================================================
# Sanity checks on configuration
if PHASE_MARGIN_MIN < 45:
    print(f"WARNING: Phase margin target ({PHASE_MARGIN_MIN} deg) is quite low. "
          f"Consider targeting >= 45 deg for stability.")

