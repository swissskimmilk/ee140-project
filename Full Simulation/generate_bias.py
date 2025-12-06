import numpy as np
from look_up import look_up_basic, importdata
import config

try:
    pch = importdata("pch_1v.mat")
    print("Loaded PMOS gm/Id data from 'pch_1v.mat'.")
except Exception as e:
    print(f"[ERROR] Could not load 'pch_1v.mat': {e}")
    raise SystemExit(1)

def design_pmos_bias_diode(
    VDD,
    Vbias_target,
    L=None,
    W_min=0.5,   # um — consistent with lookup tables
    verbose=True,
):
    """
    Design a diode-connected PMOS (source at VDD, gate=drain=Vbias_target)
    for a low-power bias branch.

    Inputs
    ------
    VDD           : high supply voltage (V)
    Vbias_target  : desired gate/drain bias voltage (V)
                    -> node where you want VTH + 2*Vov below VDD
    L             : channel length (m). If None, uses minimum in table.
    W_min         : minimum width you’re willing to use (um, gm/Id tables)
    verbose       : if True, print sizing summary

    Returns
    -------
    A dict with:
        'W_bias'   : PMOS width (m)
        'L_bias'   : PMOS length (m)
        'Id_bias'  : bias current for this device (A)
        'VSG'      : |VSG| at this operating point (V)
        'gm_id'    : gm/Id at this point (1/V)
        'id_w'     : Id/W at this point (A/m)
        'Vov_est'  : estimated overdrive (≈ 2/gm_id, V)
        'Vth_est'  : estimated Vth (≈ VSG − Vov_est, V)
    """

    # Use minimum L from the gm/ID table if not specified
    if L is None:
        # pch['L'] is the vector of available channel lengths from the lookup
        # table. Take the minimum physical length in meters.
        L_values = np.array(pch['L'], dtype=float)
        L = float(np.min(L_values))

    # Node voltages for diode-connected PMOS
    Vs = VDD
    Vg = Vbias_target
    Vd = Vbias_target
    Vb = Vs  # body tied to source

    # The PMOS lookup tables are parameterized vs |VSG| and |VSD|,
    # so we pass positive magnitudes here (not signed VGS).
    VSG = Vs - Vg                         # positive magnitude of VSG
    vgs = VSG
    vds = VSG   # diode-connected: |VSD| = |VSG|
    vsb = 0.0   # body tied to source

    # Look up key quantities at this operating point
    gm_id = float(look_up_basic(pch, 'GM_ID',
                                vgs=vgs, vds=vds, vsb=vsb, l=L))
    id_w = float(look_up_basic(pch, 'ID_W',
                               vgs=vgs, vds=vds, vsb=vsb, l=L))

    # Choose the smallest practical width (in um, consistent with lookup)
    W_bias = W_min

    # Corresponding bias current
    Id_bias = id_w * W_bias

    # Some sanity-check estimates
    Vov_est = 2.0 / gm_id                 # strong-inversion approx
    Vth_est = VSG - Vov_est               # back out an effective Vth

    # Optional quick print for interactive use
    if verbose:
        print("\n=== PMOS Bias Diode Sizing ===")
        print(f"W_bias: {W_bias:.3f} um")
        print(f"L_bias: {L:.3f} um")
        print(f"Id_bias: {Id_bias*1e6:.3f} uA")

    return {
        'W_bias':  W_bias,
        'L_bias':  L,
        'Id_bias': Id_bias,
        'VSG':     VSG,
        'gm_id':   gm_id,
        'id_w':    id_w,
        'Vov_est': Vov_est,
        'Vth_est': Vth_est,
        'VDD':     VDD,
        'Vbias':   Vbias_target,
    }


def design_two_diode_bias_branch(
    VDD,
    Vbias_high_target,
    Vbias_low_target,
    L_top=None,
    L_bot=None,
    W_top_min=0.5,
    W_bot_min=0.5,
):
    """
    Design a *series* pair of diode-connected PMOS devices that generate
    two bias voltages from a single current branch:

        VDD ── M_top (diode) ── Vbias_high ── M_bot (diode) ── Vbias_low

    The same branch current flows through both devices. The top device width
    is adjusted to minimize the branch current while meeting the minimum
    current requirement from config (BIAS_BRANCH_CURRENT_MIN). The resulting
    branch current then sets the required width of the lower device.
    """
    # Use minimum L from the gm/ID table if not specified
    if L_top is None:
        L_values = np.array(pch['L'], dtype=float)
        L_top_eff = float(np.min(L_values))
    else:
        L_top_eff = L_top
    
    # Node voltages for top diode-connected PMOS
    Vs = VDD
    Vg = Vbias_high_target
    Vd = Vbias_high_target
    Vb = Vs  # body tied to source
    
    VSG = Vs - Vg
    vgs = VSG
    vds = VSG
    vsb = 0.0
    
    # Look up key quantities at this operating point
    gm_id = float(look_up_basic(pch, 'GM_ID',
                                vgs=vgs, vds=vds, vsb=vsb, l=L_top_eff))
    id_w = float(look_up_basic(pch, 'ID_W',
                               vgs=vgs, vds=vds, vsb=vsb, l=L_top_eff))
    
    # Determine required width to meet minimum bias branch current
    # Use minimum current from config and minimize to that value
    I_target_min = config.BIAS_BRANCH_CURRENT_MIN
    
    # Calculate width needed to achieve at least the minimum current
    W_for_min_current = I_target_min / id_w if id_w > 0 else W_top_min
    
    # Use the width that gives at least the minimum current, but at least W_top_min
    W_top = max(W_top_min, W_for_min_current)
    
    # Calculate actual branch current
    I_branch = W_top * id_w
    
    # Ensure we meet the minimum (should always be true after above calculation)
    if I_branch < I_target_min:
        W_top = W_for_min_current
        I_branch = I_target_min
    
    # Create top device dict for return
    Vov_est = 2.0 / gm_id
    Vth_est = VSG - Vov_est
    top = {
        'W_bias': W_top,
        'L_bias': L_top_eff,
        'Id_bias': I_branch,
        'VSG': VSG,
        'gm_id': gm_id,
        'id_w': id_w,
        'Vov_est': Vov_est,
        'Vth_est': Vth_est,
        'VDD': VDD,
        'Vbias': Vbias_high_target,
    }

    # Bottom device: Vbias_high_target → Vbias_low_target
    # Use same L grid logic as single-diode helper
    if L_bot is None:
        L_values = np.array(pch['L'], dtype=float)
        L_bot_eff = float(np.min(L_values))
    else:
        L_bot_eff = L_bot

    Vs2 = Vbias_high_target
    Vg2 = Vbias_low_target
    Vd2 = Vbias_low_target
    Vb2 = Vs2

    VSG2 = Vs2 - Vg2
    vgs2 = VSG2
    vds2 = VSG2
    vsb2 = 0.0

    gm_id2 = float(look_up_basic(pch, 'GM_ID',
                                 vgs=vgs2, vds=vds2, vsb=vsb2, l=L_bot_eff))
    id_w2 = float(look_up_basic(pch, 'ID_W',
                                vgs=vgs2, vds=vds2, vsb=vsb2, l=L_bot_eff))

    # Width needed so that Id = I_branch at this operating point
    W_bot_calculated = I_branch / id_w2 if id_w2 > 0 else W_bot_min
    W_bot = max(W_bot_min, W_bot_calculated)

    print("\n=== Two-Diode PMOS Bias Branch ===")
    print(f"VDD:          {VDD:.3f} V")
    print(f"Vbias_high:   {Vbias_high_target:.3f} V")
    print(f"Vbias_low:    {Vbias_low_target:.3f} V")
    print(f"Branch Id:    {I_branch*1e6:.3f} uA")
    print("Device sizing (for Cadence vars):")
    print(f"LB_TOP:       {L_top_eff:.3f}u")
    print(f"WB_TOP:       {W_top:.3f}u")
    print(f"LB_BOT:       {L_bot_eff:.3f}u")
    print(f"WB_BOT:       {W_bot:.3f}u")
    if W_bot_calculated < W_bot_min:
        print(f"  (Note: WB_BOT calculated as {W_bot_calculated:.3f}u, clamped to minimum {W_bot_min:.3f}u)")
    print(f"VSG_TOP:      {top['VSG']:.3f} V")
    print(f"VSG_BOT:      {VSG2:.3f} V")

    return {
        "VDD": VDD,
        "Vbias_high": Vbias_high_target,
        "Vbias_low": Vbias_low_target,
        "Id_branch": I_branch,
        "top": {
            **top,
            "W_top": W_top,
            "L_top": L_top_eff,
        },
        "bottom": {
            "W_bot": W_bot,
            "L_bot": L_bot_eff,
            "VSG": VSG2,
            "gm_id": gm_id2,
            "id_w": id_w2,
        },
    }


if __name__ == "__main__":
    """
    Simple standalone entry point.

    Loads the 1 V PMOS gm/Id table, uses the configured high supply
    and a nominal bias voltage, and runs the bias-diode sizing once.
    """

    VDD = 1.1
    Vbias_high = 0.801
    Vbias_low = 0.282

    res = design_two_diode_bias_branch(
        VDD=VDD,
        Vbias_high_target=Vbias_high,
        Vbias_low_target=Vbias_low,
        IB_target=2e-6
    )
