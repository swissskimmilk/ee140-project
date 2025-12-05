import numpy as np
from look_up import *
import matplotlib.pyplot as plt
import prelim_design_params as params

# Load MOS data
try:
    nch_2v = importdata("nch_2v.mat")  # 2V NMOS for inputs (M1/M2)
    nch = importdata("nch_1v.mat")  # 1V NMOS for cascodes/tail (M3/M4/M9)
    pch = importdata("pch_1v.mat")  # 1V PMOS for loads/cascodes (M5–M8)
    print("MOS data loaded successfully for telescopic design.")
except Exception as e:
    print("Error loading MOS data:", e)
    raise


def vov_from_gm_id(gm_id, k=2.0):
    """Approximate Vov from gm/ID using gm/ID ≈ 2/Vov → Vov ≈ k/gm_id."""
    return k / gm_id


def design_telescopic(
    Vout_target,
    ft_fu_ratio=5,
    sat_margin_n=0.05,
    sat_margin_p=0.05,
    VDD_margin=0.03,
    mode="high_swing",
):
    """
    Telescopic stage-1 designer with PMOS active load, constrained to a
    specific single-ended output common-mode (Vout_target).

    mode:
      - "high_swing": high-swing PMOS load:
            top PMOS gate ≈ Vout (other side),
            bottom PMOS gate uses VBP0.
      - "standard":   SIMPLE SELF-BIASED LOAD:
            BOTH PMOS devices are diode-connected on the signal side
            → gate = drain (they bias themselves, no external VBP1/VBP0).
    """

    high_swing = (mode == "high_swing")
    diode_load = (mode == "standard")

    # ====== SYSTEM-LEVEL SPECS ======
    fu = params.f_u_required
    ft = fu * ft_fu_ratio
    Cint = params.CC

    VDD_spec = params.VDDL_MAX  # e.g. 1.1 V
    VDD_eff = VDD_spec - VDD_margin

    # Headroom model / constraints
    tail_headroom = 0.15   # VDS for tail device (M9)
    casc_headroom_n = 0.15   # VDS for NMOS cascode (M3/M4)
    # NOTE: we NO LONGER treat headroom_p_bot as literal VSD.
    # It can be interpreted as an *extra* "comfort margin" if you want.
    headroom_p_bot = 0.00   # extra slack beyond Vov; keep 0 or small

    # gm/ID search grids
    gm_ID_bot_p_grid = np.arange(6, 22, 1)   # bottom PMOS gm/ID candidates
    gm_ID_casc_n_grid = np.arange(8, 22, 1)   # NMOS cascode gm/ID candidates

    # Lengths
    L_casc_n = nch['L'][0]        # NMOS cascode (shortest 1V NMOS)

    # Sweep ranges — now Vout is fixed to Vout_target
    Vout_range = np.array([Vout_target])
    L_in_range = nch_2v['L']               # input NMOS L (2V device)
    L_p_range = pch['L']                  # PMOS L
    gm_ID_range_in = np.arange(5, 30, 1)       # gm/ID sweep for input NMOS
    gm_ID_top_p_grid = np.arange(5, 30, 1)       # gm/ID grid for top PMOS

    # Storage
    best_p_l = np.zeros(len(Vout_range))
    best_Av_norm = np.zeros((len(Vout_range), len(L_in_range)))
    best_n_gm_id = np.zeros((len(Vout_range), len(L_in_range)))
    p_top_gm_id_opt = np.zeros(len(Vout_range))
    p_bot_gm_id_opt = np.zeros(len(Vout_range))
    gm_ID_casc_n_opt = np.zeros((len(Vout_range), len(L_in_range)))

    best_Av_norm[:] = np.nan
    best_n_gm_id[:] = np.nan
    best_p_l[:] = np.nan
    p_top_gm_id_opt[:] = np.nan
    p_bot_gm_id_opt[:] = np.nan
    gm_ID_casc_n_opt[:] = np.nan

    # =====================================================================
    #   SWEEP (single) Vout_target: SIZE PMOS STACK FOR BEST Rout
    # =====================================================================
    for i, vout in enumerate(Vout_range):
        # PMOS stack nodes (top → bottom):
        # VDD → P_top → VZ → P_bot → Vout

        if high_swing:
            # High-swing: top PMOS gate near Vout, so VSG_top must match.
            VSG_top_req = VDD_eff - vout

        p_gds_id_best = None
        best_Lp_idx = None
        best_gmID_top_p_here = None
        best_gmID_bot_p_here = None

        # ----- PMOS stack sizing for this Vout (here just Vout_target) -----
        for gm_ID_bot_p in gm_ID_bot_p_grid:

            # *** KEY CHANGE: compute VSD_bot_p from Vov(gm/ID), not fixed ***
            vov_bot_est = vov_from_gm_id(gm_ID_bot_p)
            # VSD ≈ Vov + margin (+ optional headroom_p_bot slack)
            VSD_bot_p = vov_bot_est + sat_margin_p + headroom_p_bot

            # Node VZ is now Vout + that VSD_bot_p
            VZ = vout + VSD_bot_p
            if VZ >= VDD_eff:
                # This gm/ID would force too large VSD and violate supply
                continue

            VSD_top_p = VDD_eff - VZ   # actual VSD at top PMOS

            for idx_lp, Lp in enumerate(L_p_range):
                # Bottom PMOS (P_bot)
                p_bot_gds_id = look_up_vs_gm_id(
                    pch, 'GDS_ID', gm_ID_bot_p, vds=VSD_bot_p, l=Lp
                )
                p_bot_ft = look_up_vs_gm_id(
                    pch, 'GM_CGG', gm_ID_bot_p, vds=VSD_bot_p, l=Lp
                ) / (2 * np.pi)

                # Choose top PMOS gm/ID
                gm_ID_top_p_eff = None
                p_top_Av = None
                p_top_ft = None

                if high_swing:
                    # High-swing: top PMOS gate near Vout, so VSG_top must match.
                    best_gmID_top = None
                    best_vsg_err = None

                    for gmID_t in gm_ID_top_p_grid:
                        vov_top_est = vov_from_gm_id(gmID_t)
                        if VSD_top_p < vov_top_est + sat_margin_p:
                            continue

                        vsg_cand = look_up_vgs_vs_gm_id(
                            pch, gmID_t, vds=VSD_top_p, l=Lp
                        )
                        err = abs(vsg_cand - VSG_top_req)
                        if best_vsg_err is None or err < best_vsg_err:
                            best_vsg_err  = err
                            best_gmID_top = gmID_t

                    if best_vsg_err is None or best_vsg_err > 0.03:
                        continue

                    gm_ID_top_p_eff = best_gmID_top

                    p_top_Av = look_up_vs_gm_id(
                        pch, 'GM_GDS', gm_ID_top_p_eff, vds=VSD_top_p, l=Lp
                    )
                    p_top_ft = look_up_vs_gm_id(
                        pch, 'GM_CGG', gm_ID_top_p_eff, vds=VSD_top_p, l=Lp
                    ) / (2 * np.pi)

                elif diode_load:
                    # STANDARD SIMPLE LOAD: top PMOS is also self-biased
                    best_Av_top = None
                    best_gmID_top = None
                    best_ft_top = None

                    for gmID_t in gm_ID_top_p_grid:
                        vov_top_est = vov_from_gm_id(gmID_t)
                        if VSD_top_p < vov_top_est + sat_margin_p:
                            continue

                        Av_top_candidate = look_up_vs_gm_id(
                            pch, 'GM_GDS', gmID_t, vds=VSD_top_p, l=Lp
                        )
                        ft_top_candidate = look_up_vs_gm_id(
                            pch, 'GM_CGG', gmID_t, vds=VSD_top_p, l=Lp
                        ) / (2 * np.pi)

                        if ft_top_candidate <= ft:
                            continue

                        if (best_Av_top is None) or (Av_top_candidate > best_Av_top):
                            best_Av_top = Av_top_candidate
                            best_gmID_top = gmID_t
                            best_ft_top = ft_top_candidate

                    if best_gmID_top is None:
                        continue

                    gm_ID_top_p_eff = best_gmID_top
                    p_top_Av = best_Av_top
                    p_top_ft = best_ft_top

                else:
                    raise ValueError(f"Unknown mode '{mode}'")

                # fT constraint on both PMOS devices
                if p_bot_ft <= ft or p_top_ft <= ft:
                    continue

                # Effective PMOS gds/ID at output:
                p_bot_gds_id_eff = p_bot_gds_id / (1.0 + p_top_Av)

                if p_gds_id_best is None or p_bot_gds_id_eff < p_gds_id_best:
                    p_gds_id_best = p_bot_gds_id_eff
                    best_Lp_idx = idx_lp
                    best_gmID_top_p_here = gm_ID_top_p_eff
                    best_gmID_bot_p_here = gm_ID_bot_p

        if p_gds_id_best is None:
            best_p_l[i] = np.nan
            best_n_gm_id[i, :] = np.nan
            best_Av_norm[i, :] = np.nan
            continue

        p_gds_id = p_gds_id_best
        best_p_l[i] = L_p_range[best_Lp_idx]
        p_top_gm_id_opt[i] = best_gmID_top_p_here
        p_bot_gm_id_opt[i] = best_gmID_bot_p_here

        # =================================================================
        #   NMOS telescopic side for this fixed Vout
        # =================================================================
        for j, L_in in enumerate(L_in_range):
            Vx = vout - casc_headroom_n
            VDS_in = Vx - tail_headroom
            if VDS_in <= 0:
                best_n_gm_id[i, j] = np.nan
                best_Av_norm[i, j] = np.nan
                continue

            vov_in_range_est = vov_from_gm_id(gm_ID_range_in)

            n_gds_id_in = look_up_vs_gm_id(
                nch_2v, 'GDS_ID', gm_ID_range_in, vds=VDS_in, l=L_in
            )
            n_ft_in = look_up_vs_gm_id(
                nch_2v, 'GM_CGG', gm_ID_range_in, vds=VDS_in, l=L_in
            ) / (2 * np.pi)

            best_Av_here = 0.0
            best_gm_id_in_here = np.nan
            best_gm_ID_casc_n_here = np.nan

            for gm_ID_casc_n in gm_ID_casc_n_grid:
                vov_casc_est = vov_from_gm_id(gm_ID_casc_n)
                if casc_headroom_n < vov_casc_est + sat_margin_n:
                    continue

                Av_casc_n = look_up_vs_gm_id(
                    nch, 'GM_GDS', gm_ID_casc_n, vds=casc_headroom_n, l=L_casc_n
                )
                vgs_casc_n = look_up_vgs_vs_gm_id(
                    nch, gm_ID_casc_n, vds=casc_headroom_n, l=L_casc_n
                )
                Vbias_N_casc = Vx + vgs_casc_n

                if Vbias_N_casc > VDD_eff:
                    continue

                K_ft = (n_ft_in > ft)
                K_sat = (VDS_in >= (vov_in_range_est + sat_margin_n))
                K = K_ft & K_sat

                if not np.any(K):
                    continue

                n_gds_id_eff = n_gds_id_in / (1.0 + Av_casc_n)
                gds_id_total = n_gds_id_eff + p_gds_id

                Av_candidates = gm_ID_range_in / gds_id_total
                Av_candidates = np.where(K, Av_candidates, 0.0)

                idx_local = np.argmax(Av_candidates)
                Av_local = Av_candidates[idx_local]

                if Av_local > best_Av_here:
                    best_Av_here = Av_local
                    best_gm_id_in_here = gm_ID_range_in[idx_local]
                    best_gm_ID_casc_n_here = gm_ID_casc_n

            if best_Av_here == 0.0 or np.isnan(best_gm_id_in_here):
                best_n_gm_id[i, j] = np.nan
                best_Av_norm[i, j] = np.nan
                gm_ID_casc_n_opt[i, j] = np.nan
            else:
                best_n_gm_id[i, j] = best_gm_id_in_here
                best_Av_norm[i, j] = best_Av_here
                gm_ID_casc_n_opt[i, j] = best_gm_ID_casc_n_here

    # =====================================================================
    #   PICK GLOBAL OPTIMUM (over L_in) FOR THIS Vout_target
    # =====================================================================
    if np.all(np.isnan(best_Av_norm)):
        print(f"No valid telescopic design found for Vout_target={Vout_target:.3f} V.")
        return None

    vout_idx, n_l_idx = np.unravel_index(
        np.nanargmax(best_Av_norm.flatten()),
        best_Av_norm.shape
    )

    opt_Av_norm = best_Av_norm[vout_idx, n_l_idx]
    opt_n_gm_id = best_n_gm_id[vout_idx, n_l_idx]
    opt_p_l = best_p_l[vout_idx]
    opt_n_l = L_in_range[n_l_idx]
    opt_vout = Vout_range[vout_idx]
    opt_gm_ID_top_p = p_top_gm_id_opt[vout_idx]
    opt_gm_ID_bot_p = p_bot_gm_id_opt[vout_idx]
    opt_gm_ID_casc_n = gm_ID_casc_n_opt[vout_idx, n_l_idx]

    # =====================================================================
    #   NODE VOLTAGES AT OPTIMUM (RECOMPUTE VSD_bot, VZ, VSD_top)
    # =====================================================================
    # Bottom PMOS VSD is tied to gm/ID:
    vov_p_bot_est = vov_from_gm_id(opt_gm_ID_bot_p)
    VSD_bot_po = vov_p_bot_est + sat_margin_p + headroom_p_bot
    VZ_opt = opt_vout + VSD_bot_po
    VSD_top_po = VDD_eff - VZ_opt

    Vx_opt = opt_vout - casc_headroom_n
    VDS_in_o = Vx_opt - tail_headroom

    # NMOS cascode bias VBN
    vgs_casc_n = look_up_vgs_vs_gm_id(
        nch, opt_gm_ID_casc_n, vds=casc_headroom_n, l=L_casc_n
    )
    Vbias_N_casc = Vx_opt + vgs_casc_n

    # =====================================================================
    #   CDD LOOP TO GET gm AND Id FOR M1/M2 AT fu
    # =====================================================================
    n_cdd_w = look_up_vs_gm_id(
        nch_2v, 'CDD_W', opt_n_gm_id, vds=VDS_in_o, l=opt_n_l
    )
    n_jd = look_up_vs_gm_id(
        nch_2v, 'ID_W', opt_n_gm_id, vds=VDS_in_o, l=opt_n_l
    )

    p_bot_cdd_w = look_up_vs_gm_id(
        pch, 'CDD_W', opt_gm_ID_bot_p, vds=VSD_bot_po, l=opt_p_l
    )
    p_bot_jd = look_up_vs_gm_id(
        pch, 'ID_W', opt_gm_ID_bot_p, vds=VSD_bot_po, l=opt_p_l
    )

    cdd = 0.0
    for _ in range(10):
        gm = 2 * np.pi * fu * (Cint + cdd)
        Id = gm / opt_n_gm_id             # branch current (one side)
        wn = Id / n_jd                    # width of input NMOS
        wp_bot = Id / p_bot_jd            # width of bottom PMOS
        cdd = wn * n_cdd_w + wp_bot * p_bot_cdd_w

    # Top PMOS width: same Id
    p_top_jd = look_up_vs_gm_id(
        pch, 'ID_W', opt_gm_ID_top_p, vds=VSD_top_po, l=opt_p_l
    )
    wp_top = Id / p_top_jd

    # Tail device sizing
    tail_Id = 2 * Id
    tail_L = 1.0
    tail_vds = tail_headroom

    vov_tail_target = max(tail_headroom - sat_margin_n, 0.05)
    tail_gm_id = 2.0 / vov_tail_target

    tail_Id_w = look_up_vs_gm_id(
        nch, 'ID_W', tail_gm_id, vds=tail_vds, l=tail_L
    )
    W_tail = tail_Id / tail_Id_w

    vov_tail_est = vov_from_gm_id(tail_gm_id)

    if vov_tail_est > tail_headroom:
        print("Tail headroom likely insufficient (Vdsat > allocated).")

    # NMOS cascode sizing
    casc_Id_w = look_up_vs_gm_id(
        nch, 'ID_W', opt_gm_ID_casc_n, vds=casc_headroom_n, l=L_casc_n
    )
    W_casc = Id / casc_Id_w

    # =====================================================================
    #   INPUT CM + PMOS BIAS VOLTAGES
    # =====================================================================
    vgs_in = look_up_vgs_vs_gm_id(
        nch_2v, opt_n_gm_id, vds=VDS_in_o, l=opt_n_l
    )
    Vcm_in = tail_headroom + vgs_in   # CM at M1/M2 gates

    # Bottom PMOS bias / gate node
    if high_swing:
        vsg_bot = look_up_vgs_vs_gm_id(
            pch, opt_gm_ID_bot_p, vds=VSD_bot_po, l=opt_p_l
        )
        Vbias_P_bot = VZ_opt - vsg_bot      # external bias VBP0
    elif diode_load:
        # Self-biased: gate=drain=Vout
        Vbias_P_bot = opt_vout
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Top PMOS gate bias:
    if high_swing:
        # gate tied to bottom drain on reference side (≈ Vout)
        Vbias_P_top = opt_vout
    elif diode_load:
        # Self-biased: gate=drain=VZ
        Vbias_P_top = VZ_opt
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # =====================================================================
    #   GAIN COMPUTATION (single-ended)
    # =====================================================================
    Av_in = look_up_vs_gm_id(
        nch_2v, 'GM_GDS', opt_n_gm_id, vds=VDS_in_o, l=opt_n_l
    )
    gm_M1 = opt_n_gm_id * Id
    ro_in = Av_in / gm_M1

    Av_casc_n = look_up_vs_gm_id(
        nch, 'GM_GDS', opt_gm_ID_casc_n, vds=casc_headroom_n, l=L_casc_n
    )
    gm_casc_n = opt_gm_ID_casc_n * Id
    ro_casc_n = Av_casc_n / gm_casc_n
    boost_n = 1.0 + gm_casc_n * ro_casc_n
    Rout_down = ro_in * boost_n

    Av_p_bot = look_up_vs_gm_id(
        pch, 'GM_GDS', opt_gm_ID_bot_p, vds=VSD_bot_po, l=opt_p_l
    )
    gm_p_bot = opt_gm_ID_bot_p * Id
    ro_p_bot = Av_p_bot / gm_p_bot

    Av_p_top = look_up_vs_gm_id(
        pch, 'GM_GDS', opt_gm_ID_top_p, vds=VSD_top_po, l=opt_p_l
    )
    gm_p_top = opt_gm_ID_top_p * Id
    ro_p_top = Av_p_top / gm_p_top
    boost_p = 1.0 + gm_p_top * ro_p_top
    Rout_up = ro_p_bot * boost_p

    Rout_total = 1.0 / (1.0 / Rout_down + 1.0 / Rout_up)
    gm_eff = gm_M1
    A1_gain = gm_eff * Rout_total

    # =====================================================================
    #   CALCULATE ACTUAL UNITY GAIN FREQUENCY
    # =====================================================================
    fu_actual = gm_M1 / (2.0 * np.pi * Cint)

    # =====================================================================
    #   HEADROOM / SATURATION SANITY CHECKS (approx, based on gm/ID)
    # =====================================================================
    vov_in_est = vov_from_gm_id(opt_n_gm_id)
    vov_casc_est = vov_from_gm_id(opt_gm_ID_casc_n)
    # vov_p_bot_est already computed above
    vov_p_top_est = vov_from_gm_id(opt_gm_ID_top_p)
    # vov_tail_est already computed above

    violations = []

    if VDS_in_o < vov_in_est + sat_margin_n:
        violations.append("Input NMOS M1/M2 near triode (VDS too small vs Vov).")
    if casc_headroom_n < vov_casc_est + sat_margin_n:
        violations.append("NMOS cascode M3/M4 near triode (VDS too small vs Vov).")
    if VSD_bot_po < vov_p_bot_est + sat_margin_p:
        violations.append("Bottom PMOS near triode (VSD too small vs Vov).")
    if VSD_top_po < vov_p_top_est + sat_margin_p:
        violations.append("Top PMOS near triode (VSD too small vs Vov).")
    if tail_headroom < vov_tail_est + sat_margin_n:
        violations.append("Tail device M9 near triode (VDS too small vs Vov).")

    # =====================================================================
    #   PRINT RESULTS
    # =====================================================================
    print("\n=== TELESCOPIC DESIGN RESULTS ===")
    print(f"Mode:                  {mode}")
    print(f"Target Vout_CM:        {Vout_target:.3f} V")
    print(f"Vout_CM (opt):         {opt_vout:.3f} V")
    print(f"fu actual:             {fu_actual/1e6:.2f} MHz  (from gm1/(2*pi*Cc))")
    print(f"M1/M2 gm/ID (input):   {opt_n_gm_id:.2f} V⁻¹")
    print(f"M3/M4 gm/ID:           {opt_gm_ID_casc_n:.2f} V⁻¹")
    print(f"M5/M6 gm/ID:           {opt_gm_ID_bot_p:.2f} V⁻¹")
    print(f"M7/M8 gm/ID:           {opt_gm_ID_top_p:.2f} V⁻¹")
    print(f"M9 gm/ID (tail):       {tail_gm_id:.2f} V⁻¹")
    print(f"Normalized Av metric:  {opt_Av_norm:.2f}")
    print("")
    print(f"gm(M1):                {gm_M1*1e6:.2f} uS")
    print(f"Rout_down (NMOS side): {Rout_down/1e6:.2f} MΩ")
    print(f"Rout_up   (PMOS side): {Rout_up/1e6:.2f} MΩ")
    print(f"Rout_total (SE):       {Rout_total/1e6:.2f} MΩ")
    print(f"Stage-1 gain A1:       {A1_gain:.1f} V/V  ({20*np.log10(A1_gain):.1f} dB)")
    print("")
    print(f"PMOS L:                {opt_p_l:.2f} um")
    print(f"M7/M8:             W = {wp_top:.2f} um,  VSD = {VSD_top_po:.3f} V, Vov≈{vov_p_top_est:.3f} V")
    print(f"M5/M6:             W = {wp_bot:.2f} um, VSD = {VSD_bot_po:.3f} V, Vov≈{vov_p_bot_est:.3f} V")
    print(f"M3/M4:             L = {L_casc_n:.2f} um, W = {W_casc:.2f} um,  VDS = {casc_headroom_n:.3f} V, Vov≈{vov_casc_est:.3f} V")
    print(f"M1/M2 (2V):        L = {opt_n_l:.2f} um, W = {wn:.2f} um,  VDS = {VDS_in_o:.3f} V, Vov≈{vov_in_est:.3f} V")
    print(f"M9 (Tail NMOS):    L = {tail_L:.2f} um, W = {W_tail:.2f} um, ID_tail = {tail_Id*1e6:.2f} uA, VDS≈{tail_headroom:.3f} V, Vov≈{vov_tail_est:.3f} V")
    print("")
    print(f"Input CM Vcm_in:       {Vcm_in:.3f} V   (this is the CM at M1/M2 gates)")
    print(f"VBN (NMOS bias):       {Vbias_N_casc:.3f} V")
    print(f"VBP0 (PMOS bottom):    {Vbias_P_bot:.3f} V")
    print(f"VBP1 (PMOS top):       {Vbias_P_top:.3f} V")
    print("")
    if violations:
        print("HEADROOM WARNINGS (gm/ID model, approx Vov):")
        for v in violations:
            print("  -", v)
    else:
        print("All headroom checks (approx Vov + margins) passed in gm/ID model.")
    print("=================================\n")

    return {
        "mode":           mode,
        "A1_gain":        A1_gain,
        "Rout_total":     Rout_total,
        "fu_actual":      fu_actual,
        "ft":             ft,
        "Vout":           opt_vout,
        "VZ":             VZ_opt,
        "Vcm_in":         Vcm_in,
        "Vbias_N_casc":   Vbias_N_casc,
        "Vbias_P_bot":    Vbias_P_bot,
        "Vbias_P_top":    Vbias_P_top,
        "Id_branch":      Id,
        "Wn":             wn,
        "Wp_bot":         wp_bot,
        "Wp_top":         wp_top,
        "W_tail":         W_tail,
        "L_in":           opt_n_l,
        "L_p":            opt_p_l,
        "gm_ID_in":       opt_n_gm_id,
        "gm_ID_p_bot":    opt_gm_ID_bot_p,
        "gm_ID_p_top":    opt_gm_ID_top_p,
        "gm_ID_casc_n":   opt_gm_ID_casc_n,
        "VDS_in":         VDS_in_o,
        "VSD_bot":        VSD_bot_po,
        "VSD_top":        VSD_top_po,
        "Vov_in_est":     vov_in_est,
        "Vov_casc_est":   vov_casc_est,
        "Vov_p_bot_est":  vov_p_bot_est,
        "Vov_p_top_est":  vov_p_top_est,
        "Vov_tail_est":   vov_tail_est,
        "VDD_spec":       VDD_spec,
        "VDD_eff":        VDD_eff,
    }


if __name__ == "__main__":
    Vout_target_example = 0.503
    res_hs  = design_telescopic(Vout_target_example, mode="high_swing")
    res_std = design_telescopic(Vout_target_example, mode="standard")
