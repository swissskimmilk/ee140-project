import numpy as np
from look_up import *
import matplotlib.pyplot as plt
import calculate_design_params as params

# Load MOS data
try:
    nch_2v = importdata("nch_2v.mat")  # 2V NMOS for inputs
    nch    = importdata("nch_1v.mat")  # 1V NMOS for cascades, tail, etc.
    pch    = importdata("pch_1v.mat")  # 1V PMOS
    print("MOS data loaded successfully for telescopic design.")
except Exception as e:
    print(f"Error loading MOS data: {e}")
    raise

def design_telescopic(ft_fu_ratio = 5):
    fu   = params.f_u_required
    ft   = fu * ft_fu_ratio
    Cint = params.CC          # internal node cap target (Miller + misc)
    VDD  = 1.1

    # Simple headroom model
    tail_headroom = 0.20      # VDS of tail device (M9)
    casc_headroom_n = 0.18    # VDS of NMOS cascode (M3/M4)
    headroom_p_load = 0.18    # VSD of bottom PMOS load (M5/M6)

    # gm/ID choices for cascoding
    gm_ID_casc_n   = 10.0     # NMOS cascode
    gm_ID_load_p   = 10.0     # PMOS load
    gm_ID_casc_p   = 10.0     # PMOS cascode

    # Length choices
    L_casc_n = nch['L'][0]    # NMOS cascode (shortest 1V NMOS)
    # For PMOS, we'll sweep L_p for both load and cascode, using same L

    # Sweep ranges
    Vout_range  = np.arange(0.3, 0.7, 0.05)
    L_in_range  = nch_2v['L']      # input NMOS (2V) L sweep
    L_p_range   = pch['L']         # PMOS L sweep
    gm_ID_range = np.arange(5, 30, 1)

    # Storage over sweep
    best_p_l     = np.zeros(len(Vout_range))
    best_Av      = np.zeros((len(Vout_range), len(L_in_range)))
    best_n_gm_id = np.zeros((len(Vout_range), len(L_in_range)))

    best_Av[:]      = np.nan
    best_n_gm_id[:] = np.nan
    best_p_l[:]     = np.nan

    # ============================================================
    #   SWEEP OVER Vout, THEN PMOS L, THEN NMOS L / gm_ID
    # ============================================================
    for i, vout in enumerate(Vout_range):
        # ---------------- PMOS stack: load + cascode ----------------
        # Node voltages:
        #   Vout = output node
        #   VY   = node between PMOS load and PMOS cascode
        # Assume bottom PMOS load sees fixed VSD = headroom_p_load
        VY = vout + headroom_p_load
        if VY >= VDD:
            # Not enough headroom for PMOS stack
            continue

        VSD_load_p = headroom_p_load
        VSD_casc_p = VDD - VY

        # For this Vout, sweep PMOS L and find the best one
        p_gds_id_best = None
        best_Lp_idx   = None

        for idx_lp, Lp in enumerate(L_p_range):
            # Bottom PMOS load (mirror device)
            p_load_gds_id = look_up_vs_gm_id(
                pch, 'GDS_ID', gm_ID_load_p, vds=VSD_load_p, l=Lp
            )
            p_load_ft = look_up_vs_gm_id(
                pch, 'GM_CGG', gm_ID_load_p, vds=VSD_load_p, l=Lp
            ) / (2 * np.pi)

            # PMOS cascode device
            p_casc_Av = look_up_vs_gm_id(
                pch, 'GM_GDS', gm_ID_casc_p, vds=VSD_casc_p, l=Lp
            )  # gm/gds = Av_casc_p
            p_casc_ft = look_up_vs_gm_id(
                pch, 'GM_CGG', gm_ID_casc_p, vds=VSD_casc_p, l=Lp
            ) / (2 * np.pi)

            # fT constraint: both load and cascode must have ft > ft_min
            if (p_load_ft <= ft) or (p_casc_ft <= ft):
                continue

            # Effective PMOS gds/Id at output with cascode:
            # ro_load boosted by (1 + gm_casc*ro_casc) ≈ (1 + Av_casc_p)
            p_load_gds_id_eff = p_load_gds_id / (1.0 + p_casc_Av)

            # Keep the PMOS choice that gives the SMALLEST gds_id (largest ro)
            if (p_gds_id_best is None) or (p_load_gds_id_eff < p_gds_id_best):
                p_gds_id_best = p_load_gds_id_eff
                best_Lp_idx   = idx_lp

        if p_gds_id_best is None:
            # No PMOS L meets ft for this Vout
            best_p_l[i]        = np.nan
            best_n_gm_id[i, :] = np.nan
            best_Av[i, :]      = np.nan
            continue

        p_gds_id = p_gds_id_best
        best_p_l[i] = L_p_range[best_Lp_idx]

        # -------- NMOS telescopic side (2V input + 1V cascode) ----------
        for j, L_in in enumerate(L_in_range):
            # Node assumptions:
            #   tail node at ~tail_headroom
            #   NMOS cascode device has fixed VDS = casc_headroom_n
            #   so node X (drain M1 / source M3) is:
            Vx = vout - casc_headroom_n
            VDS_in = Vx - tail_headroom   # VDS of input device

            if VDS_in <= 0:
                best_n_gm_id[i, j] = np.nan
                best_Av[i, j]      = np.nan
                continue

            # Input NMOS (2V) gm/ID sweep at this VDS
            n_gds_id_in = look_up_vs_gm_id(
                nch_2v, 'GDS_ID', gm_ID_range, vds=VDS_in, l=L_in
            )
            n_ft_in = look_up_vs_gm_id(
                nch_2v, 'GM_CGG', gm_ID_range, vds=VDS_in, l=L_in
            ) / (2 * np.pi)

            # NMOS cascode gm/gds (1V device, fixed gm_ID_casc_n, L_casc_n, VDS=casc_headroom_n)
            Av_casc_n = look_up_vs_gm_id(
                nch, 'GM_GDS', gm_ID_casc_n, vds=casc_headroom_n, l=L_casc_n
            )

            # Only keep gm_ID points where input NMOS meets ft target
            K = n_ft_in > ft
            if np.any(K):
                # Effective NMOS gds/ID at output with cascode:
                n_gds_id_eff = n_gds_id_in / (1.0 + Av_casc_n)

                # Total effective gds/ID at output (NMOS || PMOS):
                gds_id_total = n_gds_id_eff + p_gds_id

                # Av ∝ gm / gds_total; gm ∝ gm_ID; Id cancels
                Av_candidates = gm_ID_range / gds_id_total

                # Mask out non-ft-compliant points
                Av_candidates = np.where(K, Av_candidates, 0.0)

                best_idx_n = np.argmax(Av_candidates)
                if Av_candidates[best_idx_n] == 0.0:
                    best_n_gm_id[i, j] = np.nan
                    best_Av[i, j]      = np.nan
                else:
                    best_n_gm_id[i, j] = gm_ID_range[best_idx_n]
                    best_Av[i, j]      = Av_candidates[best_idx_n]
            else:
                best_n_gm_id[i, j] = np.nan
                best_Av[i, j]      = np.nan

    # ============================================================
    #   PICK GLOBAL OPTIMUM
    # ============================================================
    if np.all(np.isnan(best_Av)):
        print("No valid telescopic design found for given constraints.")
        return None

    vout_idx, n_l_idx = np.unravel_index(
        np.nanargmax(best_Av.flatten()),
        best_Av.shape
    )
    opt_Av      = best_Av[vout_idx, n_l_idx]
    opt_n_gm_id = best_n_gm_id[vout_idx, n_l_idx]
    opt_p_l     = best_p_l[vout_idx]
    opt_n_l     = L_in_range[n_l_idx]
    opt_vout    = Vout_range[vout_idx]

    # ============================================================
    #   SIZE DEVICES (WIDTHS) WITH CDD ITERATION
    # ============================================================
    # Node voltages for the optimal point
    Vx_opt      = opt_vout - casc_headroom_n
    VDS_in_o    = Vx_opt - tail_headroom
    VY_opt      = opt_vout + headroom_p_load
    VSD_load_po = headroom_p_load
    VSD_casc_po = VDD - VY_opt
    dv_opt      = VDD - opt_vout  # just VDD - Vout (for reference)

    # --- NMOS cascode bias ---
    vgs_casc_n = look_up_vgs_vs_gm_id(
        nch, gm_ID_casc_n, vds=casc_headroom_n, l=L_casc_n
    )
    Vbias_N_casc = Vx_opt + vgs_casc_n

    # Input NMOS (2V) parasitics per width
    n_cdd_w = look_up_vs_gm_id(
        nch_2v, 'CDD_W', opt_n_gm_id, vds=VDS_in_o, l=opt_n_l
    )
    n_jd = look_up_vs_gm_id(
        nch_2v, 'ID_W', opt_n_gm_id, vds=VDS_in_o, l=opt_n_l
    )

    # PMOS load (bottom device) parasitics per width
    p_load_cdd_w = look_up_vs_gm_id(
        pch, 'CDD_W', gm_ID_load_p, vds=VSD_load_po, l=opt_p_l
    )
    p_load_jd = look_up_vs_gm_id(
        pch, 'ID_W', gm_ID_load_p, vds=VSD_load_po, l=opt_p_l
    )

    # Crude CDD loop: internal node Cint + drain Cdd from input NMOS + PMOS load
    cdd = 0.0
    for _ in range(10):
        gm = 2 * np.pi * fu * (Cint + cdd)
        Id = gm / opt_n_gm_id       # branch current for one side
        wn = Id / n_jd
        wp_load = Id / p_load_jd
        cdd = wn * n_cdd_w + wp_load * p_load_cdd_w

    # PMOS cascode width: same Id flows through, different gm/Id / VDS
    p_casc_jd = look_up_vs_gm_id(
        pch, 'ID_W', gm_ID_casc_p, vds=VSD_casc_po, l=opt_p_l
    )
    wp_casc = Id / p_casc_jd

    # Tail device M9: 1V NMOS, gm/ID fixed
    tail_Id      = 2 * Id                # differential pair tail current
    tail_L       = 1.0                   # you may want to snap this to nch['L'] grid
    tail_gm_id   = 10.0
    tail_vds     = tail_headroom
    tail_Id_w    = look_up_vs_gm_id(
        nch, 'ID_W', tail_gm_id, vds=tail_vds, l=tail_L
    )
    tail_w       = tail_Id / tail_Id_w
    tail_vds_est = 2.0 / tail_gm_id

    if tail_vds_est > tail_headroom:
        print("Tail headroom exceeded!")
    if abs(tail_vds_est - tail_headroom) > 0.01:
        print(f"Warning: tail headroom {tail_headroom:.3f} V far from "
              f"tail_vds_est {tail_vds_est:.3f} V")

    # Input CM (2V NMOS)
    vgs_in = look_up_vgs_vs_gm_id(
        nch_2v, opt_n_gm_id, vds=VDS_in_o, l=opt_n_l
    )
    vcm    = tail_headroom + vgs_in

    # --- PMOS gate biases (for your bias generators) ---
    # PMOS load gate bias: gate relative to VY (source)
    vsg_load = look_up_vgs_vs_gm_id(
        pch, gm_ID_load_p, vds=VSD_load_po, l=opt_p_l
    )
    Vbias_P_load = VY_opt - vsg_load   # gate of M5/M6

    # PMOS cascode gate bias: gate relative to VDD (source)
    vsg_casc = look_up_vgs_vs_gm_id(
        pch, gm_ID_casc_p, vds=VSD_casc_po, l=opt_p_l
    )
    Vbias_P_casc = VDD - vsg_casc      # gate of M7/M8

    # ============================================================
    #   PRINT RESULTS
    # ============================================================
    print("\n=== TELESCOPIC DESIGN RESULTS ===")
    print(f"Vout:                  {opt_vout:.3f} V")
    print(f"Optimal NMOS L_in:     {opt_n_l:.2f} um")
    print(f"Optimal PMOS L:        {opt_p_l:.2f} um")
    print(f"Optimal gm/ID (in):    {opt_n_gm_id:.2f} V⁻¹")
    print(f"Estimated Av:          {opt_Av:.2f} (gm/Id-based)")

    print(f"\nInput NMOS (2V):")
    print(f"  Wn:                  {wn:.2f} um")
    print(f"  VDS_in:              {VDS_in_o:.3f} V")

    print(f"\nPMOS load (M5/M6):")
    print(f"  Wp_load:             {wp_load:.2f} um")
    print(f"  VSD_load:            {VSD_load_po:.3f} V")
    print(f"  Vbias_P_load:        {Vbias_P_load:.3f} V  (gate)")

    print(f"\nPMOS cascode (M7/M8):")
    print(f"  Wp_casc:             {wp_casc:.2f} um")
    print(f"  VSD_casc:            {VSD_casc_po:.3f} V")
    print(f"  Vbias_P_casc:        {Vbias_P_casc:.3f} V  (gate)")

    print(f"\nTail NMOS (M9, 1V):")
    print(f"  L_tail:              {tail_L:.2f} um")
    print(f"  W_tail:              {tail_w:.2f} um")
    print(f"  Tail IDS:            {(tail_Id*1e6):.2f} uA")

    print(f"\nCommon Mode Voltage:   {vcm:.3f} V")
    print(f"NMOS cascode bias VBN: {Vbias_N_casc:.3f} V")
    print("=================================\n")

    return {
        "Vout": opt_vout,
        "Av": opt_Av,
        "opt_n_gm_id": opt_n_gm_id,
        "opt_n_L": opt_n_l,
        "opt_p_L": opt_p_l,
        "wn": wn,
        "wp_load": wp_load,
        "wp_casc": wp_casc,
        "tail_L": tail_L,
        "tail_W": tail_w,
        "Id_branch": Id,
        "Vcm": vcm,
        "Vbias_N_casc": Vbias_N_casc,
        "Vbias_P_load": Vbias_P_load,
        "Vbias_P_casc": Vbias_P_casc,
    }

if __name__ == "__main__":
    design_telescopic()
