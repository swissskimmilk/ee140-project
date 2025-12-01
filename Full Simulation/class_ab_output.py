import numpy as np
import math
from look_up import *
import prelim_design_params as params
import csv

"""
Improved Class-AB common-source output stage sizing script (Stage 2)

Features:
  - Uses gm/Id-based headroom: VDS,sat ≈ Vov ≈ 1 / (gm/Id).
  - Computes actual saturation-limited swing window [V_low_sat, V_high_sat].
  - Enforces that required swing [Vout_min_req, Vout_max_req] fits inside that window.
  - Sweeps VDDH (e.g. 1.8V, 1.7V, 1.6V, ...) and includes VDDH in CSV.
  - Picks best design by minimizing static power = VDDH * Iq_total.
  - Computes gate bias voltages:
      * Vgate_n: required NMOS gate DC voltage (== required input common-mode).
      * Vgate_p: PMOS gate DC voltage.
      * Vbias_g_p_minus_g_n: DC gate-to-gate bias (PMOS gate = NMOS gate + Vbias).
"""

# Load 2V device data for output stage
try:
    nch_2v = importdata("nch_2v.mat")
    pch_2v = importdata("pch_2v.mat")
    print("Loaded 2V device data for output stage.")
except Exception as e:
    print(f"Error loading 2V MOS data: {e}")
    raise SystemExit


def design_output_stage():
    # ---------------------------------------------------------------------
    # Project / spec parameters
    # ---------------------------------------------------------------------
    RL = params.RL                         # pixel series resistor (Ohm), e.g. 1k
    CL = params.CL                         # pixel cap (F), e.g. 25e-12
    VDDH_MAX = params.VDDH_MAX             # maximum allowed VDDH (e.g. 1.8 V)
    deltaV = params.OUTPUT_SWING_MIN       # required swing at pixel, e.g. 1.4 V
    gm2_target = params.gm2_required_stability  # total gm of stage 2 needed for AC

    # ---------------------------------------------------------------------
    # VDDH sweep: try max and some lower options (must be ≤ VDDH_MAX)
    # ---------------------------------------------------------------------
    VDDH_candidates = sorted({VDDH_MAX, 1.7, 1.6, 1.5})
    VDDH_candidates = [v for v in VDDH_candidates if 0.9 * deltaV < v <= VDDH_MAX]

    # ---------------------------------------------------------------------
    # Design knobs
    # ---------------------------------------------------------------------
    A2_min = 15                   # minimum gain of output stage (V/V)
    alpha_Ipk = 1.5               # safety factor on peak current vs ΔV/RL
    k_AB = 20.0                   # Class-AB ratio: I_pk_available ≈ k_AB * Iq

    # gm/Id and L sweeps for both devices
    gm_id_n_range = np.linspace(6, 16, 6)  # NMOS gm/Id candidates [1/V]
    gm_id_p_range = np.linspace(6, 16, 6)  # PMOS gm/Id candidates [1/V]
    L_n_candidates = [0.15, 0.18, 0.2, 0.3, 0.5]  # um
    L_p_candidates = [0.15, 0.18, 0.2, 0.3, 0.5]

    # ---------------------------------------------------------------------
    # Physics: RL–CL and current requirement
    # ---------------------------------------------------------------------
    tau_RLCL = RL * CL
    print("=" * 70)
    print("OUTPUT STAGE DESIGN (Class AB common-source)")
    print("=" * 70)
    print(f"RL      = {RL} Ω")
    print(f"CL      = {CL*1e12:.2f} pF")
    print(f"ΔV_pix  = {deltaV:.2f} V")
    print(f"τ_RLCL  = {tau_RLCL*1e9:.2f} ns  (physical load time constant)")
    print()
    print(f"VDDH candidates = {VDDH_candidates}")
    print()

    # Peak current needed through RL to support ΔV step
    I_pk_min = deltaV / RL                      # bare minimum (A)
    I_pk_target = alpha_Ipk * I_pk_min          # margin (A)
    print(f"Minimum peak current I_pk_min = {I_pk_min*1e3:.3f} mA")
    print(f"Target I_pk (with margin)      = {I_pk_target*1e3:.3f} mA (α={alpha_Ipk:.1f})")
    print(f"Class-AB ratio k_AB           = {k_AB:.1f} (I_pk ≈ k_AB * Iq)")
    print(f"Stage-2 gm target (total)     = {gm2_target*1e6:.2f} µS")
    print()

    designs = []

    # =====================================================================
    # Sweep over VDDH, gm/Id, L
    # =====================================================================
    for VDDH in VDDH_candidates:
        # We choose output common-mode at mid-supply for this stage
        Vout_CM = VDDH / 2.0
        Vout_min_req = Vout_CM - deltaV / 2.0
        Vout_max_req = Vout_CM + deltaV / 2.0

        print("-" * 70)
        print(f"VDDH = {VDDH:.3f} V")
        print(f"  Vout_CM      = {Vout_CM:.3f} V")
        print(f"  Vout_min_req = {Vout_min_req:.3f} V")
        print(f"  Vout_max_req = {Vout_max_req:.3f} V")
        print("  (Requested endpoints at the amplifier output node.)")
        print()

        for L_n in L_n_candidates:
            for L_p in L_p_candidates:
                for gm_id_n in gm_id_n_range:
                    for gm_id_p in gm_id_p_range:
                        try:
                            # -------------------------------------------------
                            # SINGLE BRANCH CURRENT:
                            # gm2_target = gm_n + gm_p = Iq * (gm_id_n + gm_id_p)
                            # → Iq set by gm2_target and gm/Id choices.
                            # -------------------------------------------------
                            denom = gm_id_n + gm_id_p
                            if denom <= 0:
                                continue

                            Iq = gm2_target / denom        # A (branch current)
                            gm_n = gm_id_n * Iq            # S
                            gm_p = gm_id_p * Iq            # S
                            gm_total = gm_n + gm_p         # ≈ gm2_target

                            # Current density J_D = ID/W from LUT, at VDS ≈ Vout_CM
                            JD_n = look_up_vs_gm_id(
                                nch_2v, "ID_W", gm_id_n, l=L_n, vds=Vout_CM
                            )  # A/um
                            JD_p = look_up_vs_gm_id(
                                pch_2v, "ID_W", gm_id_p, l=L_p, vds=(VDDH - Vout_CM)
                            )  # A/um (VSD as VDS for pch LUT)

                            if JD_n <= 0 or JD_p <= 0:
                                continue

                            # Widths (same Iq flows in N and P at DC)
                            Wn = Iq / JD_n  # um
                            Wp = Iq / JD_p  # um

                            # Intrinsic gain: gm/gds → gds = gm / (gm/gds)
                            gm_gds_n = look_up_vs_gm_id(
                                nch_2v, "GM_GDS", gm_id_n, l=L_n, vds=Vout_CM
                            )
                            gm_gds_p = look_up_vs_gm_id(
                                pch_2v, "GM_GDS", gm_id_p, l=L_p, vds=(VDDH - Vout_CM)
                            )

                            gm_gds_n = float(gm_gds_n)
                            gm_gds_p = float(gm_gds_p)
                            if gm_gds_n <= 0:
                                gm_gds_n = 1e-3
                            if gm_gds_p <= 0:
                                gm_gds_p = 1e-3

                            gds_n = gm_n / gm_gds_n
                            gds_p = gm_p / gm_gds_p
                            rout = 1.0 / (gds_n + gds_p)
                            A2 = gm_total * rout  # V/V

                            # Slew capability: rough Class-AB estimate
                            I_pk_available = k_AB * Iq   # A

                            # -------------------------------------------------
                            # Headroom / swing calculation using gm/Id
                            # Approximate VDS,sat ≈ Vov ≈ 1 / (gm/Id)
                            # -------------------------------------------------
                            if gm_id_n <= 0 or gm_id_p <= 0:
                                continue

                            Vov_n = 1.0 / gm_id_n   # ≈ VDSAT_n
                            Vov_p = 1.0 / gm_id_p   # ≈ VDSAT_p

                            V_low_sat = Vov_n
                            V_high_sat = VDDH - Vov_p
                            swing_actual = V_high_sat - V_low_sat

                            # -------------------------------------------------
                            # Compute gate bias voltages for this operating point
                            # -------------------------------------------------
                            # VGS_n and VGS_p from gm/Id tables at chosen VDS, L
                            VGS_n = float(
                                look_up_vgs_vs_gm_id(
                                    nch_2v, gm_id_n, vds=Vout_CM, l=L_n
                                )
                            )
                            VGS_p_mag = float(
                                look_up_vgs_vs_gm_id(
                                    pch_2v, gm_id_p, vds=(VDDH - Vout_CM), l=L_p
                                )
                            )

                            # NMOS: source at 0 V → gate DC = VGS_n
                            Vgate_n = VGS_n

                            # PMOS: source at VDDH, VSG magnitude = VGS_p_mag
                            # Gate is below VDDH by that magnitude
                            Vgate_p = VDDH - VGS_p_mag

                            # Gate-to-gate DC bias:
                            # Vbias = Vgp - Vgn, so PMOS gate = NMOS gate + Vbias
                            Vbias_g_p_minus_g_n = Vgate_p - Vgate_n

                            # -------------------------------------------------
                            # Constraints:
                            # 1) Slew: need I_pk_available >= I_pk_target
                            # 2) Gain: need A2 >= A2_min
                            # 3) Swing: [Vout_min_req, Vout_max_req] inside [V_low_sat, V_high_sat]
                            # -------------------------------------------------
                            if I_pk_available < I_pk_target:
                                continue
                            if A2 < A2_min:
                                continue
                            if (Vout_min_req < V_low_sat) or (Vout_max_req > V_high_sat):
                                continue

                            # Store valid design
                            designs.append(
                                dict(
                                    VDDH=VDDH,
                                    Vout_CM=Vout_CM,
                                    V_low_sat=V_low_sat,
                                    V_high_sat=V_high_sat,
                                    swing_actual=swing_actual,

                                    L_n=L_n,
                                    gm_id_n=gm_id_n,
                                    Wn=Wn,
                                    Iq_n=Iq,

                                    L_p=L_p,
                                    gm_id_p=gm_id_p,
                                    Wp=Wp,
                                    Iq_p=Iq,

                                    Iq_total=Iq,      # branch / supply current at DC
                                    gm_n=gm_n,
                                    gm_p=gm_p,
                                    gm_total=gm_total,
                                    rout=rout,
                                    A2=A2,
                                    I_pk_available=I_pk_available,

                                    VGS_n=VGS_n,
                                    VGS_p=VGS_p_mag,
                                    Vgate_n=Vgate_n,
                                    Vgate_p=Vgate_p,
                                    Vbias_g_p_minus_g_n=Vbias_g_p_minus_g_n,
                                )
                            )

                        except Exception:
                            # Skip anything that causes lookup to blow up
                            continue

    # ---------------------------------------------------------------------
    # Print & CSV export
    # ---------------------------------------------------------------------
    if not designs:
        print("[X] No valid designs found for the given constraints.")
        return None

    # Sort by static power VDDH * Iq_total (ascending)
    designs.sort(key=lambda d: d["VDDH"] * d["Iq_total"])

    # ---------------------------------------------------------------------
    # Export all valid designs to CSV
    # ---------------------------------------------------------------------
    csv_filename = "output_stage_designs.csv"

    fieldnames = [
        "VDDH [V]",

        "L_n [um]",
        "gm_id_n [1/V]",
        "Wn [um]",

        "L_p [um]",
        "gm_id_p [1/V]",
        "Wp [um]",

        "A2 [V/V]",
        "Iq_total [A]",
        "gm_n [S]",
        "gm_p [S]",
        "gm_total [S]",
        "rout [ohm]",
        "I_pk_available [A]",

        "Vout_CM [V]",
        "V_low_sat [V]",
        "V_high_sat [V]",
        "swing_actual [V]",

        "VGS_n [V]",
        "VGS_p [V]",
        "Vgate_n [V]",
        "Vgate_p [V]",
        "Vbias_pg_minus_ng [V]",
    ]

    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for d in designs:
            writer.writerow({
                "VDDH [V]": d["VDDH"],

                "L_n [um]": d["L_n"],
                "gm_id_n [1/V]": d["gm_id_n"],
                "Wn [um]": d["Wn"],

                "L_p [um]": d["L_p"],
                "gm_id_p [1/V]": d["gm_id_p"],
                "Wp [um]": d["Wp"],

                "A2 [V/V]": d["A2"],
                "Iq_total [A]": d["Iq_total"],

                "gm_n [S]": d["gm_n"],
                "gm_p [S]": d["gm_p"],
                "gm_total [S]": d["gm_total"],

                "rout [ohm]": d["rout"],
                "I_pk_available [A]": d["I_pk_available"],

                "Vout_CM [V]": d["Vout_CM"],
                "V_low_sat [V]": d["V_low_sat"],
                "V_high_sat [V]": d["V_high_sat"],
                "swing_actual [V]": d["swing_actual"],

                "VGS_n [V]": d["VGS_n"],
                "VGS_p [V]": d["VGS_p"],
                "Vgate_n [V]": d["Vgate_n"],
                "Vgate_p [V]": d["Vgate_p"],
                "Vbias_pg_minus_ng [V]": d["Vbias_g_p_minus_g_n"],
            })

    print(f"\n[OK] Exported {len(designs)} designs to '{csv_filename}'")

    # Best design (lowest static power)
    best = designs[0]

    print("\n" + "=" * 70)
    print("BEST DESIGN (MINIMUM VDDH * Iq_total)")
    print("=" * 70)
    print(f"VDDH        = {best['VDDH']:.3f} V")
    print(f"Vout_CM     = {best['Vout_CM']:.3f} V")
    print(f"V_low_sat   = {best['V_low_sat']:.3f} V")
    print(f"V_high_sat  = {best['V_high_sat']:.3f} V")
    print(f"Swing_act   = {best['swing_actual']:.3f} V (required ≥ {deltaV:.3f} V)")
    print()
    print(f"L_n         = {best['L_n']:.3f} um")
    print(f"gmId_n      = {best['gm_id_n']:.2f} V^-1")
    print(f"Wn          = {best['Wn']:.2f} um")
    print()
    print(f"L_p         = {best['L_p']:.3f} um")
    print(f"gmId_p      = {best['gm_id_p']:.2f} V^-1")
    print(f"Wp          = {best['Wp']:.2f} um")
    print()
    print(f"Iq (branch) = {best['Iq_total']*1e6:.2f} uA")
    print(f"A2          = {best['A2']:.2f} V/V")
    print(f"gm_total    = {best['gm_total']*1e6:.2f} uS")
    print(f"rout        = {best['rout']/1e3:.2f} kΩ")
    print(f"I_pk_av     ≈ {best['I_pk_available']*1e3:.2f} mA "
          f"(target ≥ {I_pk_target*1e3:.2f} mA)")
    P_mW = best['VDDH'] * best['Iq_total'] * 1e3
    print()
    print(f"VGS_n       = {best['VGS_n']:.3f} V")
    print(f"VGS_p       = {best['VGS_p']:.3f} V")
    print(f"Vgate_n     = {best['Vgate_n']:.3f} V  (required input common-mode)")
    print(f"Vgate_p     = {best['Vgate_p']:.3f} V")
    print(f"Vbias(p-n)  = {best['Vbias_g_p_minus_g_n']:.3f} V  (PMOS gate = NMOS gate + Vbias)")
    print(f"P_out,DC    ≈ {P_mW:.2f} mW at VDDH={best['VDDH']:.2f} V")
    print("=" * 70)

    # Flags for caller
    best['swing_meets_spec'] = (best['swing_actual'] >= deltaV)

    return best


if __name__ == "__main__":
    design_output_stage()
