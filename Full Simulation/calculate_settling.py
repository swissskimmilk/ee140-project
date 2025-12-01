import numpy as np
import matplotlib.pyplot as plt
import math
import prelim_design_params as params
import class_ab_output as design_tool


def simulate_settling(
    V_step,
    I_max,
    R_L,
    C_L,
    error_total,
    tau_amp=None,
    error_static=0.0,
    t_max=None,
    slew_rate_amp=None,
):
    """
    Simulate settling of the pixel node including:
      - finite static error (A0)
      - finite amplifier bandwidth (1st-order, tau_amp)
      - amplifier slew rate limiting (slew_rate_amp)
      - RL-CL load with current limiting (I_max)

    Model:
      V_cmd = V_step (ideal commanded pixel voltage)
      V_inf = V_step * (1 - error_static)    # final pixel level due to finite A0
      V_amp(t) = V_inf * (1 - exp(-t/tau_amp))  if tau_amp is not None
                 V_inf                          if tau_amp is None (instant amp)
      
      If slew_rate_amp is provided, V_amp is also slew-rate limited:
        dV_amp/dt_max = slew_rate_amp

      i_desired = (V_amp(t) - V_CL) / R_L
      i_out = clamp(i_desired, -I_max, +I_max)
      
      If slew_rate_amp is provided, also limit current based on slew rate:
        I_slew_max = slew_rate_amp * C_L (approximate, for charging CL)
        i_out = clamp(i_out, -I_slew_max, +I_slew_max)
      
      dV_CL/dt = i_out / C_L

    Settling band is defined around V_step with TOTAL error:
      [V_step * (1 - error_total), V_step * (1 + error_total)]
    """

    # RC time constant
    tau_rc = R_L * C_L

    # crude t_max guess if not given
    if t_max is None:
        # RC-only time to get within error_total
        t_est_rc = -tau_rc * math.log(max(error_total, 1e-6)) * 1.5
        # pure slew estimate: V_step on C_L with I_max
        t_est_slew = (abs(V_step) * C_L / max(I_max, 1e-12)) * 1.5
        t_max = max(t_est_rc, t_est_slew, 10 * tau_rc)

    # time vector
    n_points = 10000
    t = np.linspace(0, t_max, n_points)
    dt = t[1] - t[0]

    # state arrays
    V_CL = np.zeros(n_points)
    V_out = np.zeros(n_points)
    I_out = np.zeros(n_points)
    is_limited = np.zeros(n_points, dtype=bool)

    # final DC target due to static error
    V_inf = V_step * (1.0 - error_static)
    
    # Calculate slew-rate-limited current if slew rate is provided
    I_slew_max = None
    if slew_rate_amp is not None and slew_rate_amp > 0:
        # Maximum current available when amplifier is slewing at max rate
        # Approximate: I = C * dV/dt, so I_max_slew = slew_rate * C_L
        I_slew_max = slew_rate_amp * C_L

    # integration (forward Euler is fine here)
    V_amp_prev = 0.0  # Track previous V_amp for slew rate calculation
    for i in range(n_points - 1):
        v_cl = V_CL[i]

        # amplifier output vs time
        if tau_amp is not None and tau_amp > 0:
            V_amp_ideal = V_inf * (1.0 - math.exp(-t[i] / tau_amp))
        else:
            V_amp_ideal = V_inf
        
        # Apply slew rate limiting to V_amp
        if slew_rate_amp is not None and slew_rate_amp > 0 and i > 0:
            dt = t[i] - t[i-1]
            dV_max = slew_rate_amp * dt
            dV_available = V_amp_ideal - V_amp_prev
            if abs(dV_available) > dV_max:
                # Slew rate limited - clamp the change
                V_amp = V_amp_prev + math.copysign(dV_max, dV_available)
            else:
                V_amp = V_amp_ideal
        else:
            V_amp = V_amp_ideal
        
        V_amp_prev = V_amp

        # desired current through RL into CL
        i_desired = (V_amp - v_cl) / R_L

        # current limiting: first by I_max (output stage capability)
        if i_desired > I_max:
            i_out = I_max
            is_limited[i] = True
        elif i_desired < -I_max:
            i_out = -I_max
            is_limited[i] = True
        else:
            i_out = i_desired
            is_limited[i] = False
        
        # Additional limiting by slew rate if provided
        if I_slew_max is not None:
            if i_out > I_slew_max:
                i_out = I_slew_max
                is_limited[i] = True
            elif i_out < -I_slew_max:
                i_out = -I_slew_max
                is_limited[i] = True

        # capacitor update
        dv_cl_dt = i_out / C_L
        V_CL[i + 1] = v_cl + dv_cl_dt * dt

        # "amplifier output" node (pixel side of RL)
        V_out[i] = V_CL[i] + i_out * R_L
        I_out[i] = i_out

    # last sample
    V_out[-1] = V_out[-2]
    I_out[-1] = I_out[-2]

    # total error band around the commanded step (what the spec cares about)
    upper_bound = V_step * (1.0 + error_total)
    lower_bound = V_step * (1.0 - error_total)

    within_bounds = (V_CL >= lower_bound) & (V_CL <= upper_bound)

    settling_time = None
    settling_idx = None
    for i in range(len(within_bounds)):
        if np.all(within_bounds[i:]):
            settling_time = t[i]
            settling_idx = i
            break

    return {
        "t": t,
        "v_cl": V_CL,
        "v_out": V_out,
        "i_out": I_out,
        "is_limited": is_limited,
        "settling_time": settling_time,
        "settling_idx": settling_idx,
        "v_target_cmd": V_step,
        "v_final_dc": V_inf,
        "error_tolerance_total": error_total,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
    }


def plot_settling_response(result, save=True, show=True):
    """Plot the settling response with total error band."""

    t_ns = result["t"] * 1e9
    v_cl = result["v_cl"]
    v_out = result["v_out"]
    i_out = result["i_out"] * 1e3  # mA
    v_cmd = result["v_target_cmd"]
    v_final = result["v_final_dc"]
    error_tol = result["error_tolerance_total"]
    t_settle = result["settling_time"]
    upper = result["upper_bound"]
    lower = result["lower_bound"]

    t_slew_cc = result.get("t_slew_CC", 0.0)
    t_settle_total = result.get("settling_time_total", t_settle)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Voltages
    ax1.plot(t_ns, v_out, "b-", label="Amp Output (V_out)", linewidth=1.5, alpha=0.7)
    ax1.plot(t_ns, v_cl, "r-", label="Pixel Cap (V_CL)", linewidth=2)

    ax1.axhline(v_cmd, color="k", linestyle=":", alpha=0.7, label=f"Command: {v_cmd:.3f} V")
    ax1.axhline(v_final, color="c", linestyle="--", alpha=0.7, label=f"Final DC: {v_final:.3f} V")

    ax1.axhline(upper, color="g", linestyle=":", alpha=0.4, linewidth=1)
    ax1.axhline(lower, color="g", linestyle=":", alpha=0.4, linewidth=1)
    ax1.fill_between(
        t_ns,
        lower,
        upper,
        color="green",
        alpha=0.15,
        label=f"Total Error Band: ±{error_tol*100:.2f}%",
    )

    if t_settle is not None:
        ax1.axvline(
            t_settle * 1e9,
            color="red",
            linestyle="--",
            label=f"Settled (total err): {t_settle*1e9:.2f} ns",
            linewidth=2,
            alpha=0.7,
        )
        settle_idx = result["settling_idx"]
        if settle_idx is not None:
            ax1.plot(
                t_settle * 1e9,
                v_cl[settle_idx],
                "ro",
                markersize=8,
                label="Settling Point",
                zorder=10,
            )

        if t_slew_cc > 1e-9 and t_settle_total is not None:
            ax1.text(
                0.98,
                0.05,
                f"Total settling = {t_slew_cc*1e9:.1f} ns (CC) + "
                f"{t_settle*1e9:.1f} ns (RL-CL) = {t_settle_total*1e9:.1f} ns",
                transform=ax1.transAxes,
                fontsize=9,
                va="bottom",
                ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    ax1.set_ylabel("Voltage (V)")
    ax1.set_xlabel("Time (ns)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    if t_slew_cc > 1e-9:
        ax1.set_title(
            "Pixel Settling Response (Total Error)\n"
            f"(Includes ~{t_slew_cc*1e9:.1f} ns CC slewing delay)"
        )
    else:
        ax1.set_title("Pixel Settling Response (Total Error)")

    # Second plot: current + error
    ax2_twin = ax2.twinx()

    # Current
    ax2.plot(t_ns, i_out, "g-", linewidth=1.5, label="Output Current (mA)")

    if np.any(result["is_limited"]):
        ax2.fill_between(
            t_ns,
            0,
            i_out,
            where=result["is_limited"],
            color="orange",
            alpha=0.3,
            label="Current Limited (Slew)",
        )

    ax2.set_ylabel("Current (mA)", color="g")
    ax2.set_xlabel("Time (ns)")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2.grid(True, alpha=0.3)

    # Error percentage (vs command V_step)
    error_percent = np.abs(v_cl - v_cmd) / max(abs(v_cmd), 1e-9) * 100.0
    ax2_twin.plot(t_ns, error_percent, "m-", linewidth=1, alpha=0.7, label="Total Error %")
    ax2_twin.axhline(
        error_tol * 100,
        color="m",
        linestyle=":",
        alpha=0.6,
        label=f"Spec: ±{error_tol*100:.2f}%",
    )
    ax2_twin.set_ylabel("Error (%)", color="m")
    ax2_twin.tick_params(axis="y", labelcolor="m")
    ax2_twin.set_yscale("log")

    if t_settle is not None:
        ax2.axvline(t_settle * 1e9, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax2_twin.axvline(t_settle * 1e9, color="red", linestyle="--", linewidth=2, alpha=0.7)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()

    if save:
        plt.savefig("settling_plot.png", dpi=150, bbox_inches="tight")
        print("Plot saved to settling_plot.png")

    if show:
        plt.show()

    return fig


def analyze_settling(
    V_step,
    I_max,
    R_out,
    R_L,
    C_L,
    error_total,
    I_tail_stage1=None,
    CC=None,
    A2=None,
    f_3db_cl=None,
    error_static=None,
    fu_actual=None,
    G_CLOSED_LOOP=None,
):
    """
    Top-level settling-time analysis wrapper.

    V_step: commanded pixel step (output)
    error_total: TOTAL_ERROR_SPEC (fraction)
    error_static: ERROR_STATIC (fraction, final DC undershoot fraction)
    f_3db_cl: closed-loop amplifier 3dB bandwidth (Hz) - if provided, used directly
    fu_actual: actual unity gain frequency from design (Hz) - if provided, used to calculate f_3db_cl
    G_CLOSED_LOOP: closed-loop gain - required if fu_actual is provided
    """

    print("\n" + "=" * 70)
    print(f"{'SETTLING TIME ANALYSIS (TOTAL ERROR)':^70}")
    print("=" * 70 + "\n")

    if error_static is None:
        error_static = getattr(params, "ERROR_STATIC", 0.0)

    # Calculate f_3db_cl from actual fu if provided
    if fu_actual is not None and fu_actual > 0:
        if G_CLOSED_LOOP is None:
            G_CLOSED_LOOP = getattr(params, "G_CLOSED_LOOP", None)
        if G_CLOSED_LOOP is not None and G_CLOSED_LOOP > 0:
            f_3db_cl_amp_actual = fu_actual / G_CLOSED_LOOP
            print(f"  Using actual fu:         {fu_actual/1e6:.3f} MHz")
            print(f"  Calculated f_3dB (CL):    {f_3db_cl_amp_actual/1e6:.3f} MHz (fu_actual / G_CLOSED_LOOP)")
            f_3db_cl = f_3db_cl_amp_actual
        else:
            print(f"  [WARNING] fu_actual provided but G_CLOSED_LOOP not available, using f_3db_cl parameter")
    elif f_3db_cl is None:
        # if you store it as params.f_3db_cl_required, use that
        f_3db_cl = getattr(params, "f_3db_cl_required", None)

    if f_3db_cl is not None and f_3db_cl > 0:
        tau_amp = 1.0 / (2.0 * math.pi * f_3db_cl)
    else:
        tau_amp = None

    print("Circuit Parameters:")
    print(f"  Commanded step (V_step): {V_step:.3f} V")
    print(f"  Driver I_max:            {I_max*1e3:.2f} mA")
    print(f"  Amp Output R (R_out):    {R_out:.1f} Ohm (info)")
    print(f"  Load Resistance (R_L):   {R_L:.1f} Ohm")
    print(f"  Load Capacitance (C_L):  {C_L*1e12:.1f} pF")
    print(f"  TOTAL Error Spec:        ±{error_total*100:.3f}%")
    print(f"  Static Error (A0):       {error_static*100:.3f}%")
    if f_3db_cl is not None:
        print(f"  Amp f_3dB (closed-loop): {f_3db_cl/1e6:.3f} MHz")
    else:
        print(f"  Amp f_3dB not provided -> treating amp as instant (no BW error)")

    if I_tail_stage1 is not None and CC is not None:
        print("\nTwo-Stage Parameters:")
        print(f"  Stage 1 tail current:    {I_tail_stage1*1e6:.2f} uA")
        print(f"  Compensation cap (CC):   {CC*1e12:.2f} pF")
        if A2 is not None:
            print(f"  Output stage gain A2:    {A2:.2f} V/V")

    # Basic RC numbers
    tau = R_L * C_L
    I_required_steady = V_step / R_L

    print("\nAnalytical Estimates (RC + Current Source):")
    print(f"  RL * CL time constant:   {tau*1e9:.2f} ns")
    print(f"  Steady-state current:    {I_required_steady*1e3:.2f} mA")

    if I_required_steady > I_max:
        print("\n  Regime: SLEW LIMITED (output)")
        print(f"    Required I = {I_required_steady*1e3:.2f} mA > "
              f"I_max = {I_max*1e3:.2f} mA")
        print(f"    Initial slew rate:     {(I_max/C_L):.2e} V/s")
    else:
        print("\n  Regime: LINEAR (not output-current-limited)")
        print(f"    Required I = {I_required_steady*1e3:.2f} mA < "
              f"I_max = {I_max*1e3:.2f} mA")

    # CC slewing
    t_slew_CC = 0.0
    slew_rate_amp = None
    if I_tail_stage1 is not None and CC is not None and A2 is not None and A2 > 0:
        print("\n--- Stage 1 (CC) Slewing Check ---")
        SR_CC = I_tail_stage1 / CC
        print(f"  Slew rate at CC:         {SR_CC/1e6:.2f} V/us")

        V_stage1_swing = V_step / A2
        t_slew_CC = V_stage1_swing / SR_CC
        print(f"  Est. Stage 1 swing:      {V_stage1_swing:.3f} V")
        print(f"  CC slewing time:         {t_slew_CC*1e9:.2f} ns")

        if t_slew_CC > 1e-9:
            print(f"  [NOTE] CC slewing adds ~{t_slew_CC*1e9:.2f} ns before RL-CL settles")
        
        # Calculate effective slew rate at output (stage 1 slew rate * A2)
        # This is the maximum rate at which the amplifier output can change
        slew_rate_amp = SR_CC * A2
        print(f"  Effective output slew rate: {slew_rate_amp/1e6:.2f} V/us (SR_CC * A2)")

    # Simulate pixel settling (amp BW, static error, RL-CL+Imax, with slew rate limiting)
    result = simulate_settling(
        V_step=V_step,
        I_max=I_max,
        R_L=R_L,
        C_L=C_L,
        error_total=error_total,
        tau_amp=tau_amp,
        error_static=error_static,
        slew_rate_amp=slew_rate_amp,
    )

    print("\n--- Numerical Settling (Total Error) ---")
    if result["settling_time"] is not None:
        print(f"  Time to enter & stay within ±TOTAL err: "
              f"{result['settling_time']*1e9:.2f} ns")

        if np.any(result["is_limited"]):
            slew_duration = np.sum(result["is_limited"]) * (result["t"][1] - result["t"][0])
            print(f"  Output current limited for: {slew_duration*1e9:.2f} ns")
        else:
            print("  Output slewing: None (linear RL-CL)")

        # add CC slewing
        if t_slew_CC > 1e-9:
            result["settling_time_total"] = result["settling_time"] + t_slew_CC
            result["t_slew_CC"] = t_slew_CC
            print("\n--- Total Settling Time (incl. CC) ---")
            print(f"  CC slewing:              {t_slew_CC*1e9:.2f} ns")
            print(f"  RL-CL settle (total err):{result['settling_time']*1e9:.2f} ns")
            print(f"  TOTAL:                   {result['settling_time_total']*1e9:.2f} ns")
        else:
            result["settling_time_total"] = result["settling_time"]
            result["t_slew_CC"] = 0.0

    else:
        print("  [FAIL] Did not reach TOTAL error band in simulation window")
        result["settling_time_total"] = None
        result["t_slew_CC"] = t_slew_CC

    print("\n" + "=" * 70 + "\n")
    return result


if __name__ == "__main__":
    """
    Standalone execution of settling analysis.
    
    NOTE: This will call the design functions, which can take a while.
    For integrated analysis, use design_report.py instead, which calls
    the designs once and passes all data to analyze_settling().
    """
    print("Fetching design parameters...")
    print("NOTE: This will run the design optimizations, which may take time.")
    print("      For faster analysis, use design_report.py instead.\n")

    try:
        best_design = design_tool.design_output_stage()

        if best_design:
            # Use correct field names from class_ab_output return structure
            I_max = best_design["I_pk_available"]
            R_out = best_design["rout"]

            C_L = params.CL
            R_L = params.RL
            error_total = params.TOTAL_ERROR_SPEC

            # example step (you can tie this to your real pixel swing)
            V_step = 0.7

            # Stage 1 / CC info
            try:
                import telescopic_combined as design_stage1
                import config
                stage1_design = design_stage1.design_telescopic(mode=config.TELESCOPIC_MODE)
                I_tail_stage1 = (
                    stage1_design["Id_branch"] * 2 if stage1_design else None
                )
                # Calculate actual A1 from design
                A1_actual = stage1_design.get("A1_gain", getattr(params, "FIRST_STAGE_GAIN", 200.0))
            except Exception:
                I_tail_stage1 = None
                A1_actual = getattr(params, "FIRST_STAGE_GAIN", 200.0)

            # Use actual A2 from design, fallback to config estimate
            A2 = best_design.get("A2", getattr(params, "SECOND_STAGE_GAIN", 37.0))
            f_3db_cl = getattr(params, "f_3db_cl_required", None)
            
            # Get actual fu from stage 1 design if available
            fu_actual = stage1_design.get("fu_actual", None) if stage1_design else None
            
            # Calculate static error from actual gain (not from preliminary budget)
            import design_helpers as helpers
            A0_actual = helpers.calculate_total_gain(A1_actual, A2)
            error_static = helpers.calculate_static_error(A0_actual, params.G_CLOSED_LOOP)

            result = analyze_settling(
                V_step=V_step,
                I_max=I_max,
                R_out=R_out,
                R_L=R_L,
                C_L=C_L,
                error_total=error_total,
                I_tail_stage1=I_tail_stage1,
                CC=params.CC,
                A2=A2,
                f_3db_cl=f_3db_cl,
                error_static=error_static,
                fu_actual=fu_actual,  # Use actual fu from design
                G_CLOSED_LOOP=params.G_CLOSED_LOOP  # Required to calculate f_3db_cl from fu_actual
            )

            plot_settling_response(result, save=True, show=True)

            # spec check
            t_spec = params.T_PIXEL
            print("Specification Check:")
            print(f"  T_PIXEL spec:            {t_spec*1e9:.1f} ns")

            t_settle_check = result.get("settling_time_total", result["settling_time"])
            if t_settle_check is not None:
                print(f"  Total settling (incl CC): {t_settle_check*1e9:.2f} ns")
                if t_settle_check <= t_spec:
                    margin = (t_spec - t_settle_check) / t_spec * 100.0
                    print(f"  [PASS] Margin:          {margin:.1f}%")
                else:
                    shortfall = (t_settle_check - t_spec) / t_spec * 100.0
                    print(f"  [FAIL] Exceeds spec by: {shortfall:.1f}%")
            else:
                print("  [FAIL] Did not settle within TOTAL error band.")
        else:
            print("Could not retrieve a valid output stage design.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

