import numpy as np
from look_up import *
import matplotlib.pyplot as plt

# Load MOS data
try:
    nch_2v = importdata("nch_2v.mat")  # 2V NMOS for inputs
    nch    = importdata("nch_1v.mat")  # 1V NMOS for cascades, tail, etc.
    pch    = importdata("pch_1v.mat")  # 1V PMOS
    print("MOS data loaded successfully for telescopic design.")
except Exception as e:
    print(f"Error loading MOS data: {e}")
    raise

def design_telescopic():
    # ====== SPECIFICATION ======
    fu = 1e8                 # desired UGF
    ft = fu * 10             # device ft requirement (~10× UGF)
    VDD = 1.1
    CL  = 1e-12              # load cap seen by first stage

    V_tail   = 0.15          # VDS for tail device (M9, nch_1v)
    V_casc_n = 0.18          # min VDS for NMOS cascode (M3/M4, nch_1v)
    V_casc_p = 0.18          # min VSD for PMOS cascode/mirror (M5–M8, pch_1v)

    # Sweep output DC target
    Vout_range = np.arange(0.35, 0.75, 0.05)

    # L ranges for each device family
    L_in_range = nch_2v['L']   # inputs use 2V devices
    L_n_range  = nch['L']      # 1V NMOS for cascodes, tail
    L_p_range  = pch['L']      # 1V PMOS for cascodes/mirrors

    gm_ID_range = np.arange(6, 26, 1)

    best_gain = -1.0
    best_cfg  = None

    # =====================================================================
    #           SWEEP OUTPUT VOLTAGE TARGET (Vout)
    # =====================================================================
    for Vout in Vout_range:

        # Voltage budget downward:
        #   M9 (tail, nch_1v) + M1 (input, nch_2v) + M3 (cascode, nch_1v) = Vout
        V_in_avail = Vout - V_tail - V_casc_n
        if V_in_avail <= 0.10:
            continue

        # Voltage budget upward:
        #   M6 (PMOS cascode, pch_1v) + M8 (PMOS mirror, pch_1v) = VDD - Vout
        V_pmos_avail = VDD - Vout
        if V_pmos_avail <= 2 * V_casc_p:
            continue

        # =================================================================
        #  STEP 1: Input Transistors (M1/M2) — nch_2v
        # =================================================================
        best_input_here = None

        for L_in in L_in_range:
            for gm_ID_in in gm_ID_range:

                # Choose a VDS that fits in the NMOS budget
                VDS_in = min(0.20, V_in_avail)

                # ft requirement using 2V devices
                ft_in = look_up_vs_gm_id(
                    nch_2v, 'GM_CGG', gm_ID_in, l=L_in, vds=VDS_in
                ) / (2 * np.pi)

                if ft_in < ft:
                    continue

                # Av = gm/gds for input device (2V NMOS)
                Av_in = look_up_vs_gm_id(
                    nch_2v, 'GM_GDS', gm_ID_in, l=L_in, vds=VDS_in
                )

                # Pick highest gm/gds among those meeting ft
                if best_input_here is None or Av_in > best_input_here['Av']:
                    best_input_here = {
                        'L':     L_in,
                        'gm_ID': gm_ID_in,
                        'VDS':   VDS_in,
                        'Av':    Av_in,
                    }

        if best_input_here is None:
            continue

        L_in    = best_input_here['L']
        gm_ID_in = best_input_here['gm_ID']
        VDS_in  = best_input_here['VDS']

        # Target gm from UGF spec
        gm_target = 2 * np.pi * fu * CL

        # Branch current in one input device
        ID_in = gm_target / gm_ID_in

        # Width for M1/M2 from current density, using 2V NMOS data
        JD_in = look_up_vs_gm_id(
            nch_2v, 'ID_W', gm_ID_in, l=L_in, vds=VDS_in
        )
        W_in = ID_in / JD_in

        # =================================================================
        #  STEP 2: NMOS Cascode (M3/M4) — nch_1v
        # =================================================================
        best_n_casc = None

        for Lc in L_n_range:
            for gm_IDc in gm_ID_range:

                VDS_c = V_casc_n

                ft_c = look_up_vs_gm_id(
                    nch, 'GM_CGG', gm_IDc, l=Lc, vds=VDS_c
                ) / (2 * np.pi)
                if ft_c < ft:
                    continue

                Av_c = look_up_vs_gm_id(
                    nch, 'GM_GDS', gm_IDc, l=Lc, vds=VDS_c
                )
                gm_c = gm_IDc * ID_in
                ro_c = Av_c / gm_c      # ro = (gm/gds)/gm = 1/gds

                # cascode boost factor ~ 1 + gm_c * ro_c ≈ 1 + Av_c
                boost = 1.0 + gm_c * ro_c

                if best_n_casc is None or boost > best_n_casc['boost']:
                    best_n_casc = {
                        'L':     Lc,
                        'gm_ID': gm_IDc,
                        'VDS':   VDS_c,
                        'boost': boost,
                        'ro':    ro_c,
                        'gm':    gm_c,
                    }

        if best_n_casc is None:
            continue

        # =================================================================
        #  STEP 3: PMOS Cascode / Mirror (M5–M8) — pch_1v
        # =================================================================
        best_p_casc = None

        for Lp in L_p_range:
            for gm_IDp in gm_ID_range:

                VSD_p = V_casc_p

                # Voltage feasibility: cascode + mirror must fit
                if 2 * VSD_p > V_pmos_avail:
                    continue

                ft_p = look_up_vs_gm_id(
                    pch, 'GM_CGG', gm_IDp, l=Lp, vds=VSD_p
                ) / (2 * np.pi)
                if ft_p < ft:
                    continue

                Av_p = look_up_vs_gm_id(
                    pch, 'GM_GDS', gm_IDp, l=Lp, vds=VSD_p
                )
                gm_p = gm_IDp * ID_in
                ro_p = Av_p / gm_p

                boost_p = 1.0 + gm_p * ro_p

                if best_p_casc is None or boost_p > best_p_casc['boost']:
                    best_p_casc = {
                        'L':     Lp,
                        'gm_ID': gm_IDp,
                        'VSD':   VSD_p,
                        'boost': boost_p,
                        'ro':    ro_p,
                        'gm':    gm_p,
                    }

        if best_p_casc is None:
            continue

        # =================================================================
        #  STEP 4: Small-signal Gain Estimate
        # =================================================================
        # ro of input transistor (from 2V NMOS LUT)
        Av_in = look_up_vs_gm_id(
            nch_2v, 'GM_GDS', gm_ID_in, l=L_in, vds=VDS_in
        )
        gm_M1 = gm_ID_in * ID_in
        ro_in = Av_in / gm_M1

        # DOWNWARD resistance: input (2V NMOS) cascoded by NMOS cascode (1V NMOS)
        # Tail source (1V NMOS) assumed high ro → treat ro_in as base
        ro_down_base = ro_in
        Rout_down = ro_down_base * best_n_casc['boost']

        # UPWARD resistance: PMOS mirror + cascode (both 1V PMOS)
        Rout_up = best_p_casc['ro'] * best_p_casc['boost']

        # Total output resistance (single-ended)
        Rout_total = 1.0 / (1.0 / Rout_down + 1.0 / Rout_up)

        # Effective transconductance from one input to single-ended output
        # (Vin+ AC=1, Vin- at AC 0) → gm_eff ≈ gm_M1 / 2
        gm_eff = gm_M1 / 2.0
        A1 = gm_eff * Rout_total

        if A1 > best_gain:
            best_gain = A1
            best_cfg = {
                'Vout':   Vout,
                'A1':     A1,
                'input':  best_input_here,
                'n_casc': best_n_casc,
                'p_casc': best_p_casc,
                'ID':     ID_in,
                'W_in':   W_in,
                'gm_M1':  gm_M1,
                'Rout':   Rout_total,
                'V_tail': V_tail,
            }

    # =====================================================================
    #                  WIDTH COMPUTATION FOR ALL DEVICES
    # =====================================================================
    if best_cfg is None:
        print("No valid telescopic design found for the given constraints.")
        return None

        # === Bias / common-mode voltages ===
    # Input device VGS (2V NMOS)
    VGS_in = look_up_vgs_vs_gm_id(
        nch_2v,
        best_cfg['input']['gm_ID'],
        l=best_cfg['input']['L'],
        vds=best_cfg['input']['VDS']
    )

    # NMOS cascode VGS (1V NMOS)
    VGS_casc_n = look_up_vgs_vs_gm_id(
        nch,
        best_cfg['n_casc']['gm_ID'],
        l=best_cfg['n_casc']['L'],
        vds=best_cfg['n_casc']['VDS']
    )

    V_tail = best_cfg['V_tail']
    VDS_in  = best_cfg['input']['VDS']
    VDS_casc_n = best_cfg['n_casc']['VDS']

    # Ideal input CM (M1/M2 gates)
    Vcm_in_ideal = V_tail + VGS_in

    # Cascode gate bias for M3/M4
    VB_casc = V_tail + VDS_in + VGS_casc_n

    # Ideal output CM (node at drain of M4 / source of M6)
    Vcm_out_ideal = V_tail + VDS_in + VDS_casc_n


    # Input pair M1/M2 width already known:
    W_in = best_cfg['W_in']

    # NMOS cascode widths (M3/M4), nch_1v
    JD_n_casc = look_up_vs_gm_id(
        nch,
        'ID_W',
        best_cfg['n_casc']['gm_ID'],
        l=best_cfg['n_casc']['L'],
        vds=best_cfg['n_casc']['VDS']
    )
    W_n_casc = best_cfg['ID'] / JD_n_casc

    # PMOS cascode / mirror widths (M5–M8), pch_1v
    JD_p_casc = look_up_vs_gm_id(
        pch,
        'ID_W',
        best_cfg['p_casc']['gm_ID'],
        l=best_cfg['p_casc']['L'],
        vds=best_cfg['p_casc']['VSD']
    )
    W_p_casc = best_cfg['ID'] / JD_p_casc

    # Tail device M9, nch_1v
    gm_ID_tail = 10.0
    L_tail = 1.0
    ID_tail = 2.0 * best_cfg['ID']
    JD_tail = look_up_vs_gm_id(
        nch,
        'ID_W',
        gm_ID_tail,
        l=L_tail,
        vds=best_cfg['V_tail']
    )
    W_tail = ID_tail / JD_tail

    # =====================================================================
    #                  PRINT FINAL RESULTS
    # =====================================================================

    print("\n======= TELESCOPIC DIFF-AMP DESIGN =======")
    print(f"Best Vout target: {best_cfg['Vout']:.3f} V")
    print(f"Gain estimate:    {best_cfg['A1']:.1f} V/V "
          f"({20*np.log10(best_cfg['A1']):.1f} dB)")
    print(f"gm(M1, 2V dev):   {best_cfg['gm_M1']*1e6:.2f} uS")
    print(f"Rout (single-end):{best_cfg['Rout']/1e6:.2f} MΩ")

    print("\nInput pair (M1/M2, nch_2v):")
    print(f"  L   = {best_cfg['input']['L']:.3f} um")
    print(f"  gm/ID = {best_cfg['input']['gm_ID']:.2f} V⁻¹")
    print(f"  VDS = {best_cfg['input']['VDS']:.3f} V")
    print(f"  W   = {W_in:.2f} um per device")
    print(f"  ID  = {best_cfg['ID']*1e6:.2f} uA per device")

    print("\nNMOS cascode (M3/M4, nch_1v):")
    print(f"  L   = {best_cfg['n_casc']['L']:.3f} um")
    print(f"  gm/ID = {best_cfg['n_casc']['gm_ID']:.2f} V⁻¹")
    print(f"  VDS = {best_cfg['n_casc']['VDS']:.3f} V")
    print(f"  W   = {W_n_casc:.2f} um per device")
    print(f"  gm*ro (boost) ≈ {best_cfg['n_casc']['boost']:.1f}")

    print("\nPMOS cascode / mirror (M5–M8, pch_1v):")
    print(f"  L   = {best_cfg['p_casc']['L']:.3f} um")
    print(f"  gm/ID = {best_cfg['p_casc']['gm_ID']:.2f} V⁻¹")
    print(f"  VSD = {best_cfg['p_casc']['VSD']:.3f} V")
    print(f"  W   = {W_p_casc:.2f} um per device")
    print(f"  gm*ro (boost) ≈ {best_cfg['p_casc']['boost']:.1f}")

    print("\nTail current source (M9, nch_1v):")
    print(f"  L   = {L_tail:.3f} um")
    print(f"  gm/ID = {gm_ID_tail:.2f} V⁻¹")
    print(f"  VDS = {best_cfg['V_tail']:.3f} V")
    print(f"  W   = {W_tail:.2f} um")
    print(f"  ID  = {ID_tail*1e6:.2f} uA")

    print("\nBias / Common-mode voltages:")
    print(f"  V_B (M3/M4 gate bias):        {VB_casc:.3f} V")
    print(f"  Ideal input CM (Vcm_in):      {Vcm_in_ideal:.3f} V")
    print(f"  Ideal output CM (Vcm_out):    {Vcm_out_ideal:.3f} V")


    # Common-mode input from gm/ID LUT (2V devices)
    vcm = look_up_vgs_vs_gm_id(
        nch_2v,
        best_cfg['input']['gm_ID'],
        l=best_cfg['input']['L'],
        vds=best_cfg['input']['VDS']
    ) + best_cfg['V_tail']

    print(f"\nEstimated input common-mode Vcm ≈ {vcm:.3f} V")
    print("==========================================\n")

    return {
        'config': best_cfg,
        'W_in': W_in,
        'W_n_casc': W_n_casc,
        'W_p_casc': W_p_casc,
        'W_tail': W_tail,
        'L_tail': L_tail,
        'gm_ID_tail': gm_ID_tail,
    }

if __name__ == "__main__":
    design_telescopic()
