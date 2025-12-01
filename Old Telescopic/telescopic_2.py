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
    # ====== DESIGN KNOBS / SPECS ======
    VDD = 1.1

    # Target *input gm* (per side) – this indirectly sets "speed" / noise.
    # You can tweak this if you want faster or slower front-end.
    gm_target = 100e-6   # 100 µS

    # Headroom assumptions
    V_tail   = 0.15      # VDS for tail device (M9, nch_1v)
    V_casc_n = 0.18      # min VDS for NMOS cascode (M3/M4, nch_1v)
    V_casc_p = 0.18      # min VSD for PMOS cascode/mirror (M5–M8, pch_1v)

    # "High enough" gain threshold – used to ignore garbage designs
    A1_min = 100.0       # V/V

    # Sweep output DC target
    Vout_range = np.arange(0.35, 0.75, 0.05)

    # L ranges for each device family
    L_in_range = nch_2v['L']   # inputs use 2V devices
    L_n_range  = nch['L']      # 1V NMOS for cascodes, tail
    L_p_range  = pch['L']      # 1V PMOS for cascodes/mirrors

    # gm/ID sweep
    gm_ID_range = np.arange(6, 31, 1)   # [V^-1]

    best_FOM = -np.inf
    best_cfg = None

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
        #  Choose gm/ID, L that maximize "intrinsic gain per power-ish"
        # =================================================================
        candidates_input = []

        for L_in in L_in_range:
            for gm_ID_in in gm_ID_range:

                VDS_in = min(0.20, V_in_avail)
                if VDS_in <= 0:
                    continue

                # Intrinsic gain Av_in = gm/gds from LUT
                try:
                    Av_in = look_up_vs_gm_id(
                        nch_2v, 'GM_GDS', gm_ID_in, l=L_in, vds=VDS_in
                    )
                except Exception:
                    continue

                # crude score ~ Av * gm_ID (since for fixed gm_target,
                # FOM ~ Av * gm_ID; see gm/ID theory)
                score = Av_in * gm_ID_in

                candidates_input.append({
                    'L': L_in,
                    'gm_ID': gm_ID_in,
                    'VDS': VDS_in,
                    'Av': Av_in,
                    'score': score
                })

        if not candidates_input:
            continue

        # Sort input candidates by score descending, keep top few to avoid
        # being locked into a single local choice
        candidates_input.sort(key=lambda d: d['score'], reverse=True)
        top_inputs = candidates_input[:5]  # keep top 5 input options

        # =================================================================
        #  For each top input choice, design cascades & evaluate FOM
        # =================================================================
        for input_choice in top_inputs:

            L_in    = input_choice['L']
            gm_ID_in = input_choice['gm_ID']
            VDS_in  = input_choice['VDS']
            Av_in   = input_choice['Av']

            # For fixed gm_target, branch current in one input device:
            ID_in = gm_target / gm_ID_in

            if ID_in <= 0:
                continue

            # Width for M1/M2 from current density, using 2V NMOS data
            try:
                JD_in = look_up_vs_gm_id(
                    nch_2v, 'ID_W', gm_ID_in, l=L_in, vds=VDS_in
                )
            except Exception:
                continue

            if JD_in <= 0:
                continue

            W_in = ID_in / JD_in

            # =================================================================
            #  STEP 2: NMOS Cascode (M3/M4) — nch_1v
            # =================================================================
            best_n_casc = None

            for Lc in L_n_range:
                for gm_IDc in gm_ID_range:

                    VDS_c = V_casc_n

                    # Saturation check: VDS_c should be > ~1.2 * VDS_sat
                    VDS_sat_c = 2.0 / gm_IDc  # crude gm/ID-based estimate
                    if VDS_c < 1.2 * VDS_sat_c:
                        continue

                    try:
                        Av_c = look_up_vs_gm_id(
                            nch, 'GM_GDS', gm_IDc, l=Lc, vds=VDS_c
                        )
                    except Exception:
                        continue

                    gm_c = gm_IDc * ID_in
                    ro_c = Av_c / gm_c      # ro = (gm/gds)/gm = 1/gds

                    # Cascode boost factor
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

                    if 2 * VSD_p > V_pmos_avail:
                        continue

                    # Rough saturation check for PMOS
                    VSD_sat_p = 2.0 / gm_IDp
                    if VSD_p < 1.2 * VSD_sat_p:
                        continue

                    try:
                        Av_p = look_up_vs_gm_id(
                            pch, 'GM_GDS', gm_IDp, l=Lp, vds=VSD_p
                        )
                    except Exception:
                        continue

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
            #  STEP 4: Small-signal Gain + Power + FOM
            # =================================================================
            gm_M1 = gm_target

            # ro of input transistor (from 2V NMOS LUT)
            ro_in = Av_in / gm_M1

            # DOWNWARD resistance: input cascoded by NMOS cascode
            ro_down_base = ro_in          # tail treated as high-ro
            Rout_down = ro_down_base * best_n_casc['boost']

            # UPWARD resistance: PMOS mirror+cascode
            Rout_up = best_p_casc['ro'] * best_p_casc['boost']

            # Total Rout at single-ended output
            Rout_total = 1.0 / (1.0 / Rout_down + 1.0 / Rout_up)

            # Effective gm from Vin+ to single-ended output (Vin- at AC0)
            gm_eff = gm_M1 / 2.0
            A1 = gm_eff * Rout_total

            if A1 < A1_min:
                continue

            # Stage-1 power:
            # I(M1) = ID_in, I(M2) = ID_in, tail ≈ 2*ID_in
            # P ≈ VDD * (2*ID_in + 2*ID_in) = 4*ID_in*VDD
            P_stage1 = 4.0 * ID_in * VDD

            # FOM: gain per watt
            FOM = A1 / P_stage1

            if FOM > best_FOM:
                best_FOM = FOM
                best_cfg = {
                    'Vout':   Vout,
                    'A1':     A1,
                    'input':  input_choice,
                    'n_casc': best_n_casc,
                    'p_casc': best_p_casc,
                    'ID':     ID_in,
                    'W_in':   W_in,
                    'gm_M1':  gm_M1,
                    'Rout':   Rout_total,
                    'V_tail': V_tail,
                    'Power':  P_stage1,
                    'FOM':    FOM,
                }

    # =====================================================================
    #                  WIDTH COMPUTATION FOR ALL DEVICES
    # =====================================================================
    if best_cfg is None:
        print("No valid telescopic design found for the given constraints.")
        return None

    cfg = best_cfg

    # Input pair M1/M2
    W_in = cfg['W_in']

    # NMOS cascode widths (M3/M4), nch_1v
    JD_n_casc = look_up_vs_gm_id(
        nch,
        'ID_W',
        cfg['n_casc']['gm_ID'],
        l=cfg['n_casc']['L'],
        vds=cfg['n_casc']['VDS']
    )
    W_n_casc = cfg['ID'] / JD_n_casc

    # PMOS cascode / mirror widths (M5–M8), pch_1v
    JD_p_casc = look_up_vs_gm_id(
        pch,
        'ID_W',
        cfg['p_casc']['gm_ID'],
        l=cfg['p_casc']['L'],
        vds=cfg['p_casc']['VSD']
    )
    W_p_casc = cfg['ID'] / JD_p_casc

    # Tail device M9, nch_1v
    gm_ID_tail = 10.0
    L_tail = 1.0
    ID_tail = 2.0 * cfg['ID']
    JD_tail = look_up_vs_gm_id(
        nch,
        'ID_W',
        gm_ID_tail,
        l=L_tail,
        vds=cfg['V_tail']
    )
    W_tail = ID_tail / JD_tail

    # =====================================================================
    #        BIAS / COMMON-MODE VOLTAGES (V_B, Vcm_in, Vcm_out)
    # =====================================================================
    # Input VGS (2V NMOS)
    VGS_in = look_up_vgs_vs_gm_id(
        nch_2v,
        cfg['input']['gm_ID'],
        l=cfg['input']['L'],
        vds=cfg['input']['VDS']
    )

    # NMOS cascode VGS (1V NMOS)
    VGS_casc_n = look_up_vgs_vs_gm_id(
        nch,
        cfg['n_casc']['gm_ID'],
        l=cfg['n_casc']['L'],
        vds=cfg['n_casc']['VDS']
    )

    V_tail_sel   = cfg['V_tail']
    VDS_in_sel   = cfg['input']['VDS']
    VDS_casc_sel = cfg['n_casc']['VDS']

    # Ideal input CM (M1/M2 gates)
    Vcm_in_ideal = V_tail_sel + VGS_in

    # Cascode gate bias for M3/M4
    VB_casc = V_tail_sel + VDS_in_sel + VGS_casc_n

    # Ideal output CM (drain of M4 / source of M6)
    Vcm_out_ideal = V_tail_sel + VDS_in_sel + VDS_casc_sel

    # =====================================================================
    #                  PRINT FINAL RESULTS
    # =====================================================================

    print("\n======= TELESCOPIC DIFF-AMP DESIGN (High gain / Low power) =======")
    print(f"FOM (A1/P):       {cfg['FOM']:.3e}  [V/V per W]")
    print(f"Best Vout target: {cfg['Vout']:.3f} V")
    print(f"Gain estimate:    {cfg['A1']:.1f} V/V "
          f"({20*np.log10(cfg['A1']):.1f} dB)")
    print(f"gm(M1, target):   {cfg['gm_M1']*1e6:.2f} uS")
    print(f"Rout (single-end):{cfg['Rout']/1e6:.2f} MΩ")
    print(f"Stage-1 Power:    {cfg['Power']*1e3:.3f} mW")
    print(f"A1_min used:      {A1_min:.1f} V/V")

    print("\nInput pair (M1/M2, nch_2v):")
    print(f"  L   = {cfg['input']['L']:.3f} um")
    print(f"  gm/ID = {cfg['input']['gm_ID']:.2f} V⁻¹")
    print(f"  VDS = {cfg['input']['VDS']:.3f} V")
    print(f"  W   = {W_in:.2f} um per device")
    print(f"  ID  = {cfg['ID']*1e6:.2f} uA per device")

    print("\nNMOS cascode (M3/M4, nch_1v):")
    print(f"  L   = {cfg['n_casc']['L']:.3f} um")
    print(f"  gm/ID = {cfg['n_casc']['gm_ID']:.2f} V⁻¹")
    print(f"  VDS = {cfg['n_casc']['VDS']:.3f} V")
    print(f"  W   = {W_n_casc:.2f} um per device")
    print(f"  gm*ro (boost) ≈ {cfg['n_casc']['boost']:.1f}")

    print("\nPMOS cascode / mirror (M5–M8, pch_1v):")
    print(f"  L   = {cfg['p_casc']['L']:.3f} um")
    print(f"  gm/ID = {cfg['p_casc']['gm_ID']:.2f} V⁻¹")
    print(f"  VSD = {cfg['p_casc']['VSD']:.3f} V")
    print(f"  W   = {W_p_casc:.2f} um per device")
    print(f"  gm*ro (boost) ≈ {cfg['p_casc']['boost']:.1f}")

    print("\nTail current source (M9, nch_1v):")
    print(f"  L   = {L_tail:.3f} um")
    print(f"  gm/ID = {gm_ID_tail:.2f} V⁻¹")
    print(f"  VDS = {cfg['V_tail']:.3f} V")
    print(f"  W   = {W_tail:.2f} um")
    print(f"  ID  = {ID_tail*1e6:.2f} uA")

    print("\nBias / Common-mode voltages:")
    print(f"  V_B (M3/M4 gate bias):     {VB_casc:.3f} V")
    print(f"  Ideal input CM (Vcm_in):   {Vcm_in_ideal:.3f} V")
    print(f"  Ideal output CM (Vcm_out): {Vcm_out_ideal:.3f} V")

    print("===================================================================\n")

    return {
        'config': cfg,
        'W_in': W_in,
        'W_n_casc': W_n_casc,
        'W_p_casc': W_p_casc,
        'W_tail': W_tail,
        'L_tail': L_tail,
        'gm_ID_tail': gm_ID_tail,
        'VB_casc': VB_casc,
        'Vcm_in_ideal': Vcm_in_ideal,
        'Vcm_out_ideal': Vcm_out_ideal,
    }

if __name__ == "__main__":
    design_telescopic()
