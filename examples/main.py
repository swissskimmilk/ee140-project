"""
Script used for lab 2. Just here to be used as a reference for the general design and structure that the project should have. 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from look_up import * 
nch = importdata("nch_1v.mat")
pch = importdata("pch_1v.mat")

# print(nch['VDS'])
# print(nch['L'])

def plot_vgs_id():
    L = np.arange(0.1, 1, 0.2)
    plt.semilogy(nch['VGS'], look_up_basic(nch, 'ID', vgs=nch['VGS'], l=L).T)


    plt.legend([f"L = {l:.1f} um" for l in L])
    plt.xlabel("VGS (V)")
    plt.ylabel("ID (A)")
    plt.title("ID–VGS Characteristics for Different Channel Lengths")

    plt.show()

def plot_vds_id():
    VGS = np.arange(0.1, max(nch['VGS']), 0.3)
    plt.plot(nch['VDS'], look_up_basic(nch, 'ID', vgs=VGS, vds=nch['VDS']).T)

    plt.legend([f"VGS = {vgs:.1f} V" for vgs in VGS])
    plt.xlabel("VDS (V)")
    plt.ylabel("ID (A)")
    plt.title("ID–VDS Characteristics for Different VGS")

    plt.show()

def plot_ft_av():
    L = np.array ([0.05, 0.2, 0.5, 1])
    gm_ID = np.arange(4, 30, 0.1)
    fT = look_up_vs_gm_id(nch, 'GM_CGG', gm_ID, l = L) /2/ np.pi
    Av = look_up_vs_gm_id(nch, 'GM_GDS', gm_ID, l = L)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.semilogy(gm_ID, fT.T, '-')
    ax2.plot(gm_ID , Av .T , ':')

    ax1.set_xlabel(r'$g_m/I_D$ (V$^{-1}$)')
    ax1.set_ylabel(r'$f_T$ (Hz)')
    ax2.set_ylabel(r'$A_v = g_m/g_{ds}$ (V/V)')
    ax1.set_title(r'$f_T$ and $A_v$ vs $g_m/I_D$')
    ax1.legend([f'L = {l} um' for l in L])

    plt.show()

def exercise_1():
    gm_ID = 15
    L = 0.1
    fu = 1e9
    CL = 1e-12
    vds = 0.55

    fT = look_up_vs_gm_id(nch, 'GM_CGG', gm_ID, l=L, vds=vds) / (2*np.pi)

    # My code to obtain the intrinsic gain
    A_v0 = look_up_vs_gm_id(nch, 'GM_GDS', gm_ID, l=L, vds=vds)

    # My code to obtain current density 
    ID_norm = look_up_vs_gm_id(nch, 'ID', gm_ID, l=L, vds=vds)
    W_norm = look_up_vs_gm_id(nch, 'W',  gm_ID, l=L, vds=vds)
    J_D = ID_norm / W_norm

    # My code to obtain g_m using the fu equation and load capacitance 
    gm = 2*np.pi * fu * CL

    # Now ID, W, and required VGS
    ID  = gm / gm_ID
    W   = ID / J_D
    VGS = look_up_vgs_vs_gm_id(nch, gm_ID, l=L, vds=vds)

    print("fT (Hz):", fT)
    print("Intrinsic gain A_v0 (V/V):", A_v0)
    print("gm (S):", gm)
    print("ID (A):", ID)
    print("Width W (um):", W)
    print("VGS (V):", VGS)

def exercise_2():
    gm_ID = 5
    L = nch['L']
    fu = 5e8
    CL = 1e-12
    vds = 0.55

    # My code to obtain g_m using the fu equation and load capacitance 
    gm = 2*np.pi * fu * CL

    ID  = gm / gm_ID
    J_D = look_up_vs_gm_id(nch, 'ID_W',  gm_ID, l=L, vds=vds)
    W   = ID / J_D
    print("Ls available:", L)
    selected_L = 0.3
    print("Width at selected L:", W[np.where(L == selected_L)[0][0]])
    print("ID:", ID)
    fT = look_up_vs_gm_id(nch, 'GM_CGG', gm_ID, l=L, vds=vds) / (2*np.pi)
    
    # My code to obtain the intrinsic gain
    Av = look_up_vs_gm_id(nch, 'GM_GDS', gm_ID, l=L, vds=vds)
    
    plt.semilogy(L, fT/1e9, '-k')
    plt.plot(L, Av, ':k')
    plt.xlabel('Channel Length L (um)')
    plt.ylabel('fT [GHz] / Av [V/V]')
    plt.title('fT and Av vs Channel Length')
    plt.legend(['fT', 'Av'])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def exercise_3():
    # python
    L = nch['L']
    fu = 5e8
    CL = 1e-12
    vds = 0.55
    gm_id_range = np.linspace(5, 30, 50)
    gm_id = []
    av = []

    for i, l in enumerate(L):
        ft = look_up_vs_gm_id(nch, 'GM_CGG', gm_id_range, l=l, vds=vds) / (2*np.pi)
        m = ft >= 10 * fu
        if (np.any(m)):
            gm_id.append(gm_id_range[max(np.where(m == 1)[0])])
        else:
            gm_id.append(float('nan'))
        # your code to obtain av[i] = gm/gds
        av_val = look_up_vs_gm_id(nch, 'GM_GDS', gm_id_range, l=l, vds=vds)
        if (np.any(m)):
            av.append(av_val[max(np.where(m == 1)[0])])
        else:
            av.append(float('nan'))

    fig, ax = plt.subplots(1)
    ax.plot(L, av)
    ax.plot(L, gm_id)

    ax.set_xlabel('Channel Length L (um)')
    ax.set_ylabel('Value')
    ax.set_title('Intrinsic Gain (Av) and gm/Id vs Channel Length')
    ax.legend(['A_v = gm/gds', 'gm/Id'])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    print(av[np.where(L == 0.3)[0][0]])
    print(gm_id[np.where(L == 0.3)[0][0]])

    l_opt = 0.3
    gm_id_opt = gm_id[np.where(L == l_opt)[0][0]]

    # your code to find jd = id/w

    jd = look_up_vs_gm_id(nch, 'ID_W',  gm_id_opt, l=l_opt, vds=vds)
    cdd_w = look_up_vs_gm_id(nch, 'CDD_W', gm_id_opt, l=l_opt)
    cdd = 0
    for i in range(1, 10):
        gm_opt = 2 * np.pi * fu * (CL + cdd)
        id  = gm_opt / gm_id_opt
        w = id / jd
        cdd = w * cdd_w

    print("id:", id)
    print("w:", w)
    print("cdd:", cdd)

def exercise_4():
    fu = 1e8
    fT = fu * 10
    CL = 1e-12
    VDD = 1.1
    VDS_range = np.arange(0.1, 1.1, 0.05)
    L_range = np.arange(0.05, 0.15, 0.01)
    gm_ID = np.zeros((len(L_range), len(VDS_range)))
    gds_ID = np.zeros((len(L_range), len(VDS_range)))
    Av = np.zeros((len(L_range), len(VDS_range)))

    for k, vds in enumerate(VDS_range):
        gm_ID[:, k] = look_up_vs_gm_cgg(nch, 'GM_ID', 2 * np.pi * fT, vds=vds, l=L_range)

        for i, l in enumerate(L_range):
            gds_ID[i, k] = look_up_vs_gm_id(nch, 'GDS_ID', gm_ID[i, k], vds=vds, l=l)
            
            Av[i, k] = gm_ID[i, k] / (gds_ID[i, k] + 1/(VDD - vds))

    gain_opt_idx = np.argmax(Av.flatten())
    l_idx, vds_idx = np.unravel_index(gain_opt_idx, Av.shape)
    vds_opt = VDS_range[vds_idx]
    l_opt = L_range[l_idx]

    # Your code to calculate gm_ID_opt
    gm_ID_opt = gm_ID[l_idx, vds_idx]
    # Your code to calculate JD_opt
    JD_opt = look_up_vs_gm_id(nch, 'ID_W', gm_ID_opt, vds=vds_opt, l=l_opt)

    Cdd_W = look_up_vs_gm_id(nch, 'CDD_W', gm_ID_opt, vds=vds_opt, l=l_opt)
    cdd = 0

    for i in range(10):
        gm_opt = 2 * np.pi * fu * (CL + cdd)
        id  = gm_opt / gm_ID_opt
        w = id / JD_opt
        cdd = w * Cdd_W


    # Your code to calculate RD
    RD = (VDD - vds_opt) / id

    # Your code to calculate VGS
    VGS = look_up_vgs_vs_gm_id(nch, gm_ID_opt, l=l_opt, vds=vds_opt)

    print("gm_id:", gm_ID_opt)
    print("id:", id * 1e6)
    print("vds:", vds_opt)
    print("w:", w)
    print("cdd:", cdd)
    print("cdd_w:", Cdd_W)
    print("rd:", RD)
    print("vgs:", VGS)
    print("l:", l_opt)
    print("av:", Av.flatten()[gain_opt_idx])

def exercise_5():
    # Python
    fu = 1e8
    ft = fu*10
    CL = 1e-12
    VDD = 1.1
    mirr_headroom = 0.201
    Vout_range = np.arange(0.3, 0.7, 0.05)
    L_range = nch['L']
    gm_ID_range = np.arange(5, 30, 1)
    # Calculated over the sweep
    best_p_l = np.zeros(len(Vout_range))
    best_Av = np.zeros((len(Vout_range), len(L_range)))
    best_n_gm_id = np.zeros((len(Vout_range), len(L_range)))
    # Sweep over Vout
    for i, vout in enumerate(Vout_range):
        # Find the best PMOS L
        p_gds_id_vs_l = look_up_basic(pch, 'GDS_ID', vgs=(VDD - vout), vds=(VDD - vout), l=L_range)
        p_ft_vs_l = look_up_basic(pch, 'GM_CGG', vgs=(VDD - vout), vds=(VDD - vout), l=L_range) / (2*np.pi)
        M = p_ft_vs_l > ft
        if any(M):
            # We know gds_id will decrease as L increases, so find the highest L
            best_idx = max(np.where(M==1)[0])
            p_gds_id = p_gds_id_vs_l[best_idx]
            best_p_l[i] = L_range[best_idx]
            # Knowing PMOS GDS and L, optimize NMOS L and gm_ID
            for j, l in enumerate(L_range):
                n_gds_id = look_up_vs_gm_id(nch, 'GDS_ID', gm_ID_range, l=l, vds=vout - mirr_headroom)
                n_ft = look_up_vs_gm_id(nch, 'GM_CGG', gm_ID_range, l=l, vds=vout - mirr_headroom) / (2*np.pi)
                av = gm_ID_range / (n_gds_id + p_gds_id)
                K = n_ft > ft
                if any(K):
                    best_idx = np.argmax(np.where(K==1, av, 0))
                    best_n_gm_id[i, j] = gm_ID_range[best_idx]
                    best_Av[i, j] = av[best_idx]
                else:
                    best_n_gm_id[i, j] = float('NaN')
                    best_Av[i, j] = float('NaN')
        else:
            best_p_l[i] = float('NaN')
            best_n_gm_id[i, :] = float('NaN')
            best_Av[i, :] = float('NaN')

    # Find the best Av
    vout_idx, n_l_idx = np.unravel_index(np.nanargmax(best_Av.flatten()), best_Av.shape)
    opt_n_gm_id = best_n_gm_id[vout_idx, n_l_idx]
    opt_p_l = best_p_l[vout_idx]
    opt_n_l = L_range[n_l_idx]
    opt_vout = Vout_range[vout_idx]
    # Find the real widths of the devices
    n_cdd_w = look_up_vs_gm_id(nch, 'CDD_W', opt_n_gm_id, vds=opt_vout - mirr_headroom, l=opt_n_l)
    n_jd = look_up_vs_gm_id(nch, 'ID_W', opt_n_gm_id, vds=opt_vout - mirr_headroom, l=opt_n_l)
    p_cdd_w = look_up_basic(pch, 'CDD_W', vgs=(VDD - opt_vout), vds=(VDD - opt_vout), l=opt_p_l)
    p_jd = look_up_basic(pch, 'ID_W', vgs=(VDD - opt_vout), vds=(VDD - opt_vout), l=opt_p_l)
    cdd = 0
    for i in range(10):
        gm = 2 * np.pi * fu * (CL + cdd)
        id = gm / opt_n_gm_id
        wn = id / n_jd
        wp = id / p_jd
        cdd = wn * n_cdd_w + wp * p_cdd_w

    # Size the NMOS mirror
    mirr_id = 2 * id
    mirr_selected_l = 1
    mirr_gm_id = 10
    mirr_vds = 2 / mirr_gm_id

    if (mirr_vds > mirr_headroom):
        print("Mirror headroom exceeded!")

    if abs(mirr_vds - mirr_headroom) > 0.01:
        print(f"Warning, mirr headroom of {mirr_headroom} far off from mirr_vds of {mirr_vds}!")

    mirr_id_w = look_up_vs_gm_id(nch, 'ID_W', mirr_gm_id, l=mirr_selected_l, vds=mirr_vds)
    mirr_w = mirr_id / mirr_id_w

    vcm = look_up_vgs_vs_gm_id(nch, opt_n_gm_id, l=opt_n_l, vds=opt_vout - mirr_headroom) + mirr_headroom

    print("\n=== DESIGN RESULTS ===")
    print(f"Vout: {opt_vout:.3f} V")
    print(f"Optimal NMOS L: {opt_n_l:.2f} um")
    print(f"NMOS width (Wn): {wn:.2f} um")
    print(f"Optimal PMOS L: {opt_p_l:.2f} um")
    print(f"PMOS width (Wp): {wp:.2f} um")
    print(f"Mirr L: {mirr_selected_l:.2f} um")
    print(f"Mirr width: {mirr_w:.2f} um")
    print(f"Optimal gm/ID: {opt_n_gm_id:.2f} V⁻¹")
    print(f"Av: {np.nanmax(best_Av):.2f}")
    print(f"Mirror IDS: {(id*1e6*2):.2f} uA")
    print(f"Common Mode Voltage: {(vcm):.3f} V")

exercise_5()