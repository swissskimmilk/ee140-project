"""
Design Helper Functions
=======================

Common calculations and utilities for amplifier design analysis.
These functions make the code more readable and avoid duplication.
"""

import math
import numpy as np


# ==============================================================================
# FEEDBACK & ERROR CALCULATIONS
# ==============================================================================

def calculate_beta(G_closed_loop):
    """
    Calculate feedback factor from closed-loop gain.
    
    Args:
        G_closed_loop: Closed-loop gain (V/V)
    
    Returns:
        Beta: Feedback factor (dimensionless)
    """
    return 1.0 / G_closed_loop


def calculate_loop_gain(A0, G_closed_loop):
    """
    Calculate loop gain from open-loop gain and closed-loop gain.
    
    Args:
        A0: Open-loop DC gain (V/V)
        G_closed_loop: Closed-loop gain (V/V)
    
    Returns:
        Loop gain: A0 * beta (V/V)
    """
    beta = calculate_beta(G_closed_loop)
    return A0 * beta


def calculate_static_error(A0, G_closed_loop):
    """
    Calculate static error from actual amplifier gain.
    
    Static error = 1 / (1 + loop_gain)
    where loop_gain = A0 * beta and beta = 1 / G_closed_loop
    
    Args:
        A0: Open-loop DC gain (V/V)
        G_closed_loop: Closed-loop gain (V/V)
    
    Returns:
        Static error: Fraction (0-1)
    """
    loop_gain = calculate_loop_gain(A0, G_closed_loop)
    return 1.0 / (1.0 + loop_gain)


def calculate_required_A0(error_static, G_closed_loop):
    """
    Calculate required open-loop gain from static error budget.
    
    Args:
        error_static: Static error budget (fraction, 0-1)
        G_closed_loop: Closed-loop gain (V/V)
    
    Returns:
        Required A0: Open-loop gain (V/V)
    """
    beta = calculate_beta(G_closed_loop)
    loop_gain_required = (1.0 / error_static) - 1.0
    return loop_gain_required / beta


# ==============================================================================
# FREQUENCY & BANDWIDTH CALCULATIONS
# ==============================================================================

def calculate_unity_gain_freq(gm1, CC):
    """
    Calculate unity gain frequency for Miller-compensated amplifier.
    
    f_u = gm1 / (2 * pi * CC)
    
    Args:
        gm1: Stage 1 transconductance (S)
        CC: Miller compensation capacitor (F)
    
    Returns:
        Unity gain frequency (Hz)
    """
    return gm1 / (2.0 * math.pi * CC)


def calculate_pole_frequency(pole_value):
    """
    Convert pole value (rad/s) to frequency (Hz).
    
    f_p = |p| / (2 * pi)
    
    Args:
        pole_value: Pole value (rad/s, typically negative)
    
    Returns:
        Pole frequency (Hz)
    """
    return abs(pole_value) / (2.0 * math.pi)


def calculate_dominant_pole(gm1, A2, CC):
    """
    Calculate dominant pole for Miller-compensated two-stage amplifier.
    
    p1 = -gm1 / (A2 * CC)
    
    Args:
        gm1: Stage 1 transconductance (S)
        A2: Stage 2 gain (V/V)
        CC: Miller compensation capacitor (F)
    
    Returns:
        Dominant pole (rad/s, negative)
    """
    return -gm1 / (A2 * CC)


def calculate_non_dominant_pole(gm2, CL):
    """
    Calculate non-dominant pole at output node.
    
    p2 = -gm2 / CL
    
    Args:
        gm2: Stage 2 transconductance (S)
        CL: Load capacitance (F)
    
    Returns:
        Non-dominant pole (rad/s, negative)
    """
    return -gm2 / CL


def calculate_closed_loop_bandwidth(f_u, G_closed_loop):
    """
    Calculate closed-loop 3dB bandwidth from unity gain frequency.
    
    f_3dB_cl ≈ f_u / G_closed_loop (for single-pole system)
    
    Args:
        f_u: Unity gain frequency (Hz)
        G_closed_loop: Closed-loop gain (V/V)
    
    Returns:
        Closed-loop 3dB bandwidth (Hz)
    """
    return f_u / G_closed_loop


# ==============================================================================
# STABILITY CALCULATIONS
# ==============================================================================

def estimate_phase_margin(f_u, f_p2):
    """
    Estimate phase margin for two-pole system (simplified).
    
    PM ≈ 90° - atan(f_u / f_p2)
    
    This is a simplified estimate assuming:
    - Dominant pole at much lower frequency
    - Non-dominant pole at f_p2
    - No zeros in the range of interest
    
    Args:
        f_u: Unity gain frequency (Hz)
        f_p2: Non-dominant pole frequency (Hz)
    
    Returns:
        Estimated phase margin (degrees)
    """
    return 90.0 - math.atan(f_u / f_p2) * 180.0 / math.pi


def calculate_pole_separation_ratio(f_p2, f_u):
    """
    Calculate pole separation ratio (p2/f_u).
    
    Higher ratio generally means better phase margin.
    Typically want p2/f_u >= 2-3 for good stability.
    
    Args:
        f_p2: Non-dominant pole frequency (Hz)
        f_u: Unity gain frequency (Hz)
    
    Returns:
        Pole separation ratio (dimensionless)
    """
    return f_p2 / f_u


# ==============================================================================
# UNIT CONVERSIONS & FORMATTING
# ==============================================================================

def gain_to_dB(gain):
    """
    Convert gain (V/V) to decibels.
    
    dB = 20 * log10(gain)
    
    Args:
        gain: Voltage gain (V/V)
    
    Returns:
        Gain in decibels (dB)
    """
    return 20.0 * math.log10(gain)


def dB_to_gain(dB):
    """
    Convert decibels to gain (V/V).
    
    gain = 10^(dB/20)
    
    Args:
        dB: Gain in decibels
    
    Returns:
        Voltage gain (V/V)
    """
    return 10.0 ** (dB / 20.0)


def format_frequency_Hz(f_Hz):
    """
    Format frequency in Hz to human-readable string with appropriate units.
    
    Args:
        f_Hz: Frequency (Hz)
    
    Returns:
        Formatted string (e.g., "1.5 MHz", "250 kHz", "50 Hz")
    """
    if f_Hz >= 1e9:
        return f"{f_Hz/1e9:.2f} GHz"
    elif f_Hz >= 1e6:
        return f"{f_Hz/1e6:.2f} MHz"
    elif f_Hz >= 1e3:
        return f"{f_Hz/1e3:.2f} kHz"
    else:
        return f"{f_Hz:.2f} Hz"


def format_capacitance_F(C_F):
    """
    Format capacitance in Farads to human-readable string.
    
    Args:
        C_F: Capacitance (F)
    
    Returns:
        Formatted string (e.g., "1.5 pF", "250 fF")
    """
    if C_F >= 1e-9:
        return f"{C_F/1e-9:.2f} nF"
    elif C_F >= 1e-12:
        return f"{C_F/1e-12:.2f} pF"
    elif C_F >= 1e-15:
        return f"{C_F/1e-15:.2f} fF"
    else:
        return f"{C_F:.2e} F"


def format_resistance_Ohm(R_Ohm):
    """
    Format resistance in Ohms to human-readable string.
    
    Args:
        R_Ohm: Resistance (Ohm)
    
    Returns:
        Formatted string (e.g., "1.5 MOhm", "250 kOhm", "50 Ohm")
    """
    if R_Ohm >= 1e6:
        return f"{R_Ohm/1e6:.2f} MOhm"
    elif R_Ohm >= 1e3:
        return f"{R_Ohm/1e3:.2f} kOhm"
    else:
        return f"{R_Ohm:.2f} Ohm"


def format_transconductance_S(gm_S):
    """
    Format transconductance in Siemens to human-readable string.
    
    Args:
        gm_S: Transconductance (S)
    
    Returns:
        Formatted string (e.g., "1.5 mS", "250 uS", "50 nS")
    """
    if gm_S >= 1e-3:
        return f"{gm_S/1e-3:.2f} mS"
    elif gm_S >= 1e-6:
        return f"{gm_S/1e-6:.2f} uS"
    elif gm_S >= 1e-9:
        return f"{gm_S/1e-9:.2f} nS"
    else:
        return f"{gm_S:.2e} S"


def format_current_A(I_A):
    """
    Format current in Amperes to human-readable string.
    
    Args:
        I_A: Current (A)
    
    Returns:
        Formatted string (e.g., "1.5 mA", "250 uA", "50 nA")
    """
    if I_A >= 1e-3:
        return f"{I_A/1e-3:.2f} mA"
    elif I_A >= 1e-6:
        return f"{I_A/1e-6:.2f} uA"
    elif I_A >= 1e-9:
        return f"{I_A/1e-9:.2f} nA"
    else:
        return f"{I_A:.2e} A"


def format_power_W(P_W):
    """
    Format power in Watts to human-readable string.
    
    Args:
        P_W: Power (W)
    
    Returns:
        Formatted string (e.g., "1.5 mW", "250 uW")
    """
    if P_W >= 1e-3:
        return f"{P_W/1e-3:.3f} mW"
    elif P_W >= 1e-6:
        return f"{P_W/1e-6:.3f} uW"
    else:
        return f"{P_W:.3e} W"


def format_time_s(t_s):
    """
    Format time in seconds to human-readable string.
    
    Args:
        t_s: Time (s)
    
    Returns:
        Formatted string (e.g., "1.5 ms", "250 us", "50 ns")
    """
    if t_s >= 1e-3:
        return f"{t_s/1e-3:.2f} ms"
    elif t_s >= 1e-6:
        return f"{t_s/1e-6:.2f} us"
    elif t_s >= 1e-9:
        return f"{t_s/1e-9:.2f} ns"
    else:
        return f"{t_s:.2e} s"


# ==============================================================================
# GAIN CALCULATIONS
# ==============================================================================

def calculate_total_gain(A1, A2):
    """
    Calculate total open-loop gain from stage gains.
    
    A0 = A1 * A2
    
    Args:
        A1: Stage 1 gain (V/V)
        A2: Stage 2 gain (V/V)
    
    Returns:
        Total open-loop gain (V/V)
    """
    return A1 * A2


def calculate_stage_gain_from_gm_rout(gm, rout, R_load=None):
    """
    Calculate stage gain from transconductance and output resistance.
    
    If R_load is provided: A = gm * (rout || R_load)
    Otherwise: A = gm * rout
    
    Args:
        gm: Transconductance (S)
        rout: Output resistance (Ohm)
        R_load: Optional load resistance (Ohm)
    
    Returns:
        Stage gain (V/V)
    """
    if R_load is not None:
        R_eff = 1.0 / (1.0 / rout + 1.0 / R_load)
        return gm * R_eff
    else:
        return gm * rout


# ==============================================================================
# ERROR PERCENTAGE FORMATTING
# ==============================================================================

def format_error_fraction(error_fraction, decimal_places=4):
    """
    Format error fraction to percentage string.
    
    Args:
        error_fraction: Error as fraction (0-1)
        decimal_places: Number of decimal places
    
    Returns:
        Formatted string (e.g., "0.2000%")
    """
    return f"{error_fraction * 100:.{decimal_places}f}%"
