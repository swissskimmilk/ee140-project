import numpy as np
import matplotlib.pyplot as plt
from look_up import *
import calculate_design_params as params

# Load MOS data
# Device selection per README constraints:
# - Input devices MUST be 2V (connected to input)
# - Stage 1 uses VDDL = 1.1V, so other devices can be 1V
try:
    nch_2v = importdata("nch_2v.mat")  # For input pair (M1, M2)
    nch = importdata("nch_1v.mat")     # For cascodes and tail (M3, M4, M9)
    pch = importdata("pch_1v.mat")     # For PMOS cascodes and loads (M5-M8)
    print("MOS data loaded successfully.")
except Exception as e:
    print(f"Error loading MOS data: {e}")
    exit()

def design_telescopic_stage1():
    """
    Telescopic Cascode Differential Amplifier Design (9 transistors)
    Single-Ended Output with Current Mirror Load
    
    Topology (9 transistors):
    
    VDD (VDDL = 1.1V)
      |         |
    [M7]═════[M8] ← PMOS current mirror (pch_1v)
      |         |     M7 diode-connected (gate=drain), M8 mirrors
    [M5]═════[M6] ← PMOS cascode (pch_1v)
      |         |     M5 diode-connected (gate=drain), M6 follows M5
    [M3]     [M4] ← NMOS cascode devices (nch_1v), gates tied to VB_casc
      |         |
    [M1]     [M2] ← NMOS input differential pair (nch_2v - REQUIRED)
      |         |     M1 = Vin-, M2 = Vin+
      +----+----+
           |
          [M9] ← NMOS tail current source (nch_1v)
           |
          GND
    
    Output: Taken from M8/M6 drain (single-ended)
    
    Returns:
        dict: Complete design with all device sizes and parameters
    """
    
    print("="*70)
    print("TELESCOPIC CASCODE AMPLIFIER - SINGLE-ENDED OUTPUT")
    print("="*70)
    print("Current Mirror Load: M7/M5 diode-connected, M8/M6 mirrors")
    print("Inputs: M1=Vin-, M2=Vin+ | Output: M8/M6 drain")
    
    # Design Parameters
    A1_target = params.A1_gain_needed
    Gm1_required = params.gm1_required
    f_u_calc = params.f_u_required
    VDD = params.VDDL_MAX
    Rout1_required = A1_target / Gm1_required
    
    print(f"\nTarget Parameters:")
    print(f"  Unity Gain Freq: {f_u_calc/1e6:.2f} MHz")
    print(f"  Gm1: {Gm1_required*1e6:.2f} uS")
    print(f"  Rout1: {Rout1_required/1e6:.2f} MOhm")
    print(f"  Gain A1: {A1_target:.1f} V/V ({20*np.log10(A1_target):.1f} dB)")
    print(f"  VDDL: {VDD} V")
    
    # ========================================================================
    # VOLTAGE-AWARE DESIGN APPROACH
    # ========================================================================
    # Key insight: We must allocate voltage budget FIRST, then size devices
    # to fit within that budget while meeting performance requirements.
    #
    # Voltage allocation strategy:
    # 1. Reserve minimum voltage for each stage
    # 2. Allocate remaining voltage to optimize performance
    # 3. Choose gm/ID values that fit the voltage constraints
    # ========================================================================
    
    # STEP 1: Design Input Pair (M1, M2) with voltage awareness
    print(f"\nSTEP 1: INPUT DIFFERENTIAL PAIR (M1, M2) - nch_2v devices")
    print(f"Designing with voltage-aware approach...")
    
    # Reserve voltages for non-input devices (initial estimates)
    VDS_tail_target = 0.15  # Tail current source
    VDS_casc_n_target = 0.20  # NMOS cascode minimum (increased for margin)
    
    # For PMOS side, reserve voltages
    # Strategy: Be CONSERVATIVE - PMOS typically needs more voltage than VDS_sat suggests
    # Allocate LESS to PMOS, MORE to NMOS to prevent crushing M3/M4
    Vout_target = VDD * 0.6  # Target output higher (was 0.5)
    V_pmos_available = VDD - Vout_target  # Voltage for PMOS stack (now smaller)
    
    # Need to split V_pmos_available between M8 and M6
    # For diode devices, VDS ~ VGS ~ Vth + (2*ID)/(β*(n-1)) where n = 1/gm/ID
    # Rough estimate: VGS ~ 0.3-0.6V depending on gm/ID
    # Allocate conservatively
    V_M8_target = V_pmos_available * 0.5  # ~0.25V
    V_M6_target = V_pmos_available * 0.5  # ~0.25V
    
    # Calculate available voltage for NMOS side
    V_nmos_available = Vout_target  # From GND to Vout
    V_input_available = V_nmos_available - VDS_tail_target - VDS_casc_n_target
    
    print(f"  Voltage budget:")
    print(f"    Vout target: {Vout_target:.3f}V")
    print(f"    NMOS side: {V_nmos_available:.3f}V")
    print(f"    Available for M1/M2: {V_input_available:.3f}V")
    
    L_range = np.array([0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    gm_ID_range = np.linspace(5, 25, 41)
    VDS_input_candidates = np.linspace(0.10, 0.35, 11)
    VDS_input_est = 0.2  # used for ft feasibility check only
    
    best_input_design = None
    min_power = float('inf')
    
    for L in L_range:
        ft_vals = look_up_vs_gm_id(nch_2v, 'GM_CGG', gm_ID_range, l=L, vds=VDS_input_est) / (2*np.pi)
        mask = ft_vals > 10 * f_u_calc
        
        if np.any(mask):
            valid_gm_id = gm_ID_range[mask]
            
            for gm_id in valid_gm_id:
                id_req = Gm1_required / gm_id
                
                for VDS_test in VDS_input_candidates:
                    if VDS_test > V_input_available:
                        continue
                    
                    # Also check total stack feasibility with actual VDS_test
                    stack_min = VDS_tail_target + VDS_test + VDS_casc_n_target + V_pmos_available
                    if stack_min > VDD * 1.01:  # tighter margin since data-driven
                        continue
                    
                    try:
                        JD_tmp = look_up_vs_gm_id(nch_2v, 'ID_W', gm_id, l=L, vds=VDS_test)
                        if JD_tmp <= 0:
                            continue
                        VGS_tmp = look_up_vgs_vs_gm_id(nch_2v, gm_id, l=L, vds=VDS_test)
                        Av_tmp = look_up_vs_gm_id(nch_2v, 'GM_GDS', gm_id, l=L, vds=VDS_test)
                    except:
                        continue
                    
                    power_est = VDD * 2 * id_req
                    
                    if power_est < min_power:
                        min_power = power_est
                        best_input_design = {
                            'L': L,
                            'gm_ID': gm_id,
                            'ID': id_req,
                            'VDS': VDS_test,
                            'VGS': VGS_tmp,
                            'Av': Av_tmp,
                            'JD': JD_tmp
                        }
    
    if not best_input_design:
        print("[X] No valid input pair design found!")
        return None
    
    L_in = best_input_design['L']
    gm_ID_in = best_input_design['gm_ID']
    ID_in = best_input_design['ID']
    VDS_in_actual = best_input_design['VDS']
    VGS_in = best_input_design['VGS']
    Av_in = best_input_design['Av']
    JD_in = best_input_design['JD']
    
    # Calculate width and other parameters
    W_in = ID_in / JD_in
    gds_in = Gm1_required / Av_in
    ro_in = 1 / gds_in
    
    print(f"\n[OK] Input Pair: nch_2v, L={L_in}um, W={W_in:.2f}um")
    print(f"    gm/ID={gm_ID_in:.2f}, ID={ID_in*1e6:.2f}uA, VGS={VGS_in:.3f}V")
    print(f"    VDS={VDS_in_actual:.3f}V (from lookup), budget: {V_input_available:.3f}V")
    print(f"    Gain={Av_in:.1f}, ro={ro_in/1e3:.1f}kOhm")
    
    # STEP 2: Design NMOS Cascode (M3, M4) - Voltage-Aware
    print(f"\nSTEP 2: NMOS CASCODE (M3, M4) - nch_1v devices")
    print(f"    Voltage constraint: {VDS_casc_n_target:.3f}V available")
    
    # Calculate actual available voltage after input stage
    V_tail_actual = VDS_tail_target
    V_used_by_input = VDS_in_actual
    V_remaining_for_casc = V_nmos_available - V_tail_actual - V_used_by_input
    
    print(f"    Actual available: {V_remaining_for_casc:.3f}V")
    
    Target_Ron = 2 * Rout1_required
    L_casc_sweep = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    gm_ID_casc_sweep = [6, 8, 10, 12, 15, 18, 20, 22, 25]
    
    best_ron = 0
    best_L_casc_n = 0.5
    best_gm_ID_casc_n = 8
    
    for L_test in L_casc_sweep:
        for gm_ID_test in gm_ID_casc_sweep:
            try:
                # Voltage constraint: VDS_sat must fit in available voltage
                VDS_sat_casc = 2 / gm_ID_test
                
                if VDS_sat_casc > V_remaining_for_casc * 0.7:  # Use 70% for margin
                    continue
                
                VDS_test = max(VDS_sat_casc * 1.5, 0.15)  # At least 50% above saturation
                
                if VDS_test > V_remaining_for_casc:
                    continue
                
                Av_test = look_up_vs_gm_id(nch, 'GM_GDS', gm_ID_test, l=L_test, vds=VDS_test)
                gm_test = gm_ID_test * ID_in
                ro_test = Av_test / gm_test
                ron_estimate = gm_test * ro_test * (ro_in + 100e3)
                
                if ron_estimate > best_ron:
                    best_ron = ron_estimate
                    best_L_casc_n = L_test
                    best_gm_ID_casc_n = gm_ID_test
            except:
                continue
    
    L_casc_n = best_L_casc_n
    gm_ID_casc_n = best_gm_ID_casc_n
    ID_casc_n = ID_in
    VDS_casc_n = max(2 / gm_ID_casc_n * 1.2, 0.15)  # 20% above saturation
    
    JD_casc_n = look_up_vs_gm_id(nch, 'ID_W', gm_ID_casc_n, l=L_casc_n, vds=VDS_casc_n)
    W_casc_n = ID_casc_n / JD_casc_n
    VGS_casc_n = look_up_vgs_vs_gm_id(nch, gm_ID_casc_n, l=L_casc_n, vds=VDS_casc_n)
    gm_casc_n = gm_ID_casc_n * ID_casc_n
    Av_casc_n = look_up_vs_gm_id(nch, 'GM_GDS', gm_ID_casc_n, l=L_casc_n, vds=VDS_casc_n)
    gds_casc_n = gm_casc_n / Av_casc_n
    ro_casc_n = 1 / gds_casc_n
    
    print(f"[OK] NMOS Cascode: nch_1v, L={L_casc_n}um, W={W_casc_n:.2f}um")
    print(f"    gm/ID={gm_ID_casc_n:.2f}, ID={ID_casc_n*1e6:.2f}uA")
    print(f"    VDS={VDS_casc_n:.3f}V, VDS_sat={2/gm_ID_casc_n:.3f}V")
    print(f"    Boost={gm_casc_n * ro_casc_n:.1f}")
    
    # STEP 2B: Calculate actual Vout based on NMOS side
    print(f"\nSTEP 2B: CALCULATE DC OPERATING POINT")
    V_tail_actual = VDS_tail_target
    Vout_actual = V_tail_actual + VDS_in_actual + VDS_casc_n
    V_pmos_actual = VDD - Vout_actual
    
    print(f"  NMOS stack: {V_tail_actual:.3f} + {VDS_in_actual:.3f} + {VDS_casc_n:.3f} = {Vout_actual:.3f}V")
    print(f"  PMOS available: {V_pmos_actual:.3f}V (for M8 + M6)")
    
    # STEP 3: Design PMOS Side - Voltage-Constrained
    print(f"\nSTEP 3: PMOS DEVICES (M5-M8) - Voltage-Constrained Design")
    print(f"    Total voltage budget: {V_pmos_actual:.3f}V")
    print(f"    Must fit: M8 (mirror) + M6 (cascode)")
    
    # Strategy: Try different gm/ID combinations that FIT in voltage budget
    # For diode-connected devices, VSD ≈ VSG
    # We need: VSD_M8 + VSD_M6 ≤ V_pmos_actual
    
    Target_Rop = 2 * Rout1_required
    L_pmos_sweep = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    # Higher gm/ID = weaker inversion = lower VGS = less voltage consumed!
    gm_ID_pmos_sweep = [15, 18, 20, 22, 25, 28, 30, 32, 35]  # Shifted higher
    
    best_design = None
    best_rop = 0
    
    for L_M8 in L_pmos_sweep:
        for gm_ID_M8 in gm_ID_pmos_sweep:
            for L_M6 in L_pmos_sweep:
                for gm_ID_M6 in gm_ID_pmos_sweep:
                    try:
                        # M8 (mirror): Estimate VSG for diode connection
                        # For diode: VSD ≈ VSG, iterate to find it
                        VSG_M8 = 0.4
                        for _ in range(5):
                            VSD_M8 = VSG_M8
                            VSG_M8_new = look_up_vgs_vs_gm_id(pch, gm_ID_M8, l=L_M8, vds=VSD_M8)
                            VSG_M8 = 0.7 * VSG_M8 + 0.3 * VSG_M8_new
                        VSD_M8 = VSG_M8
                        
                        # M6 (cascode): Also diode on reference side
                        VSG_M6 = 0.4
                        for _ in range(5):
                            VSD_M6 = VSG_M6
                            VSG_M6_new = look_up_vgs_vs_gm_id(pch, gm_ID_M6, l=L_M6, vds=VSD_M6)
                            VSG_M6 = 0.7 * VSG_M6 + 0.3 * VSG_M6_new
                        VSD_M6 = VSG_M6
                        
                        # VOLTAGE CONSTRAINT: Must fit in budget
                        V_total_pmos = VSD_M8 + VSD_M6
                        if V_total_pmos > V_pmos_actual:
                            continue
                        
                        # Check if devices are well above saturation
                        # Need MORE margin than theory suggests!
                        VDS_sat_M8 = 2 / gm_ID_M8
                        VDS_sat_M6 = 2 / gm_ID_M6
                        if VSD_M8 < VDS_sat_M8 * 1.5 or VSD_M6 < VDS_sat_M6 * 1.5:
                            continue
                        
                        # Calculate output resistance
                        gm_M6 = gm_ID_M6 * ID_in
                        Av_M6 = look_up_vs_gm_id(pch, 'GM_GDS', gm_ID_M6, l=L_M6, vds=VSD_M6)
                        ro_M6 = Av_M6 / gm_M6
                        
                        gm_M8 = gm_ID_M8 * ID_in
                        Av_M8 = look_up_vs_gm_id(pch, 'GM_GDS', gm_ID_M8, l=L_M8, vds=VSD_M8)
                        ro_M8 = Av_M8 / gm_M8
                        
                        rop_estimate = gm_M6 * ro_M6 * (ro_M8 + ro_in)
                        
                        # Prefer designs that use more of the voltage budget (better performance)
                        voltage_efficiency = V_total_pmos / V_pmos_actual
                        merit = rop_estimate * voltage_efficiency
                        
                        if merit > best_rop:
                            best_rop = merit
                            best_design = {
                                'L_M8': L_M8,
                                'gm_ID_M8': gm_ID_M8,
                                'VSG_M8': VSG_M8,
                                'VSD_M8': VSD_M8,
                                'L_M6': L_M6,
                                'gm_ID_M6': gm_ID_M6,
                                'VSG_M6': VSG_M6,
                                'VSD_M6': VSD_M6,
                                'V_total': V_total_pmos,
                                'rop': rop_estimate
                            }
                    except:
                        continue
    
    if not best_design:
        print("[X] No valid PMOS design found that fits voltage budget!")
        return None
    
    # Extract PMOS cascode (M5/M6) parameters
    L_casc_p = best_design['L_M6']
    gm_ID_casc_p = best_design['gm_ID_M6']
    VSG_casc_p = best_design['VSG_M6']
    VSD_casc_p = best_design['VSD_M6']
    ID_casc_p = ID_in
    
    JD_casc_p = look_up_vs_gm_id(pch, 'ID_W', gm_ID_casc_p, l=L_casc_p, vds=VSD_casc_p)
    W_casc_p = ID_casc_p / JD_casc_p
    gm_casc_p = gm_ID_casc_p * ID_casc_p
    Av_casc_p = look_up_vs_gm_id(pch, 'GM_GDS', gm_ID_casc_p, l=L_casc_p, vds=VSD_casc_p)
    ro_casc_p = Av_casc_p / gm_casc_p
    
    print(f"[OK] PMOS Cascode (M5/M6): pch_1v, L={L_casc_p}um, W={W_casc_p:.2f}um")
    print(f"    gm/ID={gm_ID_casc_p:.2f}, VSD=VSG={VSG_casc_p:.3f}V (diode)")
    print(f"    VDS_sat={2/gm_ID_casc_p:.3f}V, margin={VSD_casc_p - 2/gm_ID_casc_p:.3f}V")
    
    # STEP 4: PMOS Current Mirror (M7, M8) - Use best design
    print(f"\n[OK] PMOS Current Mirror (M7/M8): pch_1v, L={best_design['L_M8']}um")
    
    L_src_p = best_design['L_M8']
    gm_ID_src_p = best_design['gm_ID_M8']
    VSG_src_p = best_design['VSG_M8']
    VSD_src_p = best_design['VSD_M8']
    ID_src_p = ID_in
    
    JD_src_p = look_up_vs_gm_id(pch, 'ID_W', gm_ID_src_p, l=L_src_p, vds=VSD_src_p)
    W_src_p = ID_src_p / JD_src_p
    Av_src_p = look_up_vs_gm_id(pch, 'GM_GDS', gm_ID_src_p, l=L_src_p, vds=VSD_src_p)
    gm_src_p = gm_ID_src_p * ID_src_p
    ro_src_p = Av_src_p / gm_src_p
    
    print(f"    W={W_src_p:.2f}um, gm/ID={gm_ID_src_p:.2f}")
    print(f"    VSD = VSG = {VSG_src_p:.3f}V (diode-connected)")
    print(f"    VDS_sat={2/gm_ID_src_p:.3f}V, margin={VSD_src_p - 2/gm_ID_src_p:.3f}V")
    
    print(f"\n  PMOS Stack Verification:")
    print(f"    M8: {VSD_src_p:.3f}V + M6: {VSD_casc_p:.3f}V = {VSD_src_p + VSD_casc_p:.3f}V")
    print(f"    Budget: {V_pmos_actual:.3f}V, Used: {best_design['V_total']:.3f}V")
    print(f"    Margin: {V_pmos_actual - best_design['V_total']:.3f}V")
    
    # STEP 5: Design Tail Current Source (M9) - nch_1v
    print(f"\nSTEP 5: TAIL CURRENT SOURCE (M9) - nch_1v device")
    print(f"    Voltage budget: {VDS_tail_target:.3f}V")
    
    L_tail = 0.5
    gm_ID_tail = 15  # Higher gm/ID for lower VGS (voltage-efficient)
    ID_tail = 2 * ID_in
    VDS_tail = VDS_tail_target
    
    JD_tail = look_up_vs_gm_id(nch, 'ID_W', gm_ID_tail, l=L_tail, vds=VDS_tail)
    W_tail = ID_tail / JD_tail
    VGS_tail = look_up_vgs_vs_gm_id(nch, gm_ID_tail, l=L_tail, vds=VDS_tail)
    Av_tail = look_up_vs_gm_id(nch, 'GM_GDS', gm_ID_tail, l=L_tail, vds=VDS_tail)
    gm_tail = gm_ID_tail * ID_tail
    gds_tail = gm_tail / Av_tail
    ro_tail = 1 / gds_tail
    
    print(f"[OK] Tail: nch_1v, L={L_tail}um, W={W_tail:.2f}um, ID={ID_tail*1e6:.2f}uA")
    print(f"    VDS={VDS_tail:.3f}V, VDS_sat={2/gm_ID_tail:.3f}V")
    
    # STEP 6: Voltage Stack Verification
    print(f"\n{'='*70}")
    print("STEP 6: DC OPERATING POINT VERIFICATION")
    print("="*70)
    
    # Use the already-calculated values from voltage-aware design
    Vcm = VGS_in + VDS_tail
    Vout_dc = Vout_actual
    
    # Verify NMOS side
    V_nmos_total = VDS_tail + VDS_in_actual + VDS_casc_n
    
    # Verify PMOS side
    V_pmos_total = VSD_src_p + VSD_casc_p
    
    # Total stack
    V_stack_total = V_nmos_total + V_pmos_total
    headroom_margin = VDD - V_stack_total
    
    print(f"\n  Input Common Mode:")
    print(f"    Vcm = {Vcm:.3f}V (VGS_tail + VDS_tail = {VGS_tail:.3f} + {VDS_tail:.3f})")
    print(f"    Both M1 and M2 gates must be at {Vcm:.3f}V")
    
    print(f"\n  Output DC Level:")
    print(f"    Vout_dc = {Vout_dc:.3f}V")
    
    print(f"\n  NMOS Stack (Ground to Vout):")
    print(f"    M9 (tail):      VDS = {VDS_tail:.3f}V, VDS_sat = {2/gm_ID_tail:.3f}V")
    print(f"    M1/M2 (input):  VDS = {VDS_in_actual:.3f}V (lookup-based)")
    print(f"    M3/M4 (cascode): VDS = {VDS_casc_n:.3f}V, VDS_sat = {2/gm_ID_casc_n:.3f}V")
    print(f"    Total: {V_nmos_total:.3f}V")
    
    print(f"\n  PMOS Stack (VDD to Vout):")
    print(f"    M7/M8 (mirror):  VSD = {VSD_src_p:.3f}V, VSD_sat = {2/gm_ID_src_p:.3f}V")
    print(f"    M5/M6 (cascode): VSD = {VSD_casc_p:.3f}V, VSD_sat = {2/gm_ID_casc_p:.3f}V")
    print(f"    Total: {V_pmos_total:.3f}V")
    
    print(f"\n  Stack Verification:")
    print(f"    NMOS: {V_nmos_total:.3f}V + PMOS: {V_pmos_total:.3f}V = {V_stack_total:.3f}V")
    print(f"    VDD:  {VDD:.3f}V")
    print(f"    Error: {abs(VDD - V_stack_total)*1000:.2f}mV {'[OK]' if abs(VDD - V_stack_total) < 0.01 else '[WARNING]'}")
    
    # Calculate bias voltage for M3/M4 cascode gates
    VB_casc = VDS_tail + VGS_in + VGS_casc_n
    print(f"\n  Bias Voltages:")
    print(f"    VB_casc (M3/M4 gates): {VB_casc:.3f}V")
    
    # STEP 6B: NODE VOLTAGE ANALYSIS FOR DEBUG
    print(f"\n{'='*70}")
    print("EXPECTED NODE VOLTAGES (for Virtuoso verification)")
    print("="*70)
    
    # Calculate all node voltages
    V_tail_node = VDS_tail
    
    # Reference side (M1-M3-M5-M7 stack) - uses diode connections
    V_M1_drain_ref = V_tail_node + VDS_in_actual
    V_M3_drain_ref = V_M1_drain_ref + VDS_casc_n
    V_M5_drain_ref = V_M3_drain_ref + VSD_casc_p  # M5 diode: VSD = VSG
    V_M7_source_ref = V_M5_drain_ref
    V_M7_drain_ref = VDD  # M7 drain = VDD
    
    # Output side (M2-M4-M6-M8 stack) - mirrors reference side
    V_M2_drain_out = V_tail_node + VDS_in_actual
    V_M4_drain_out = V_M2_drain_out + VDS_casc_n  # This is Vout
    V_M6_source_out = V_M4_drain_out  # = Vout
    V_M6_drain_out = V_M6_source_out + VSD_casc_p
    V_M8_source_out = V_M6_drain_out
    V_M8_drain_out = VDD
    
    print(f"\nREFERENCE SIDE (M1-M3-M5-M7 stack):")
    print(f"  GND:           0.000V")
    print(f"  M9 drain:      {V_tail_node:.3f}V (= M1/M2 sources)")
    print(f"  M1 drain:      {V_M1_drain_ref:.3f}V (= M3 source)")
    print(f"  M3 drain:      {V_M3_drain_ref:.3f}V (= M5 source)")
    print(f"  M5 drain:      {V_M5_drain_ref:.3f}V (= M7 source, DIODE)")
    print(f"  M7 source:     {V_M7_source_ref:.3f}V")
    print(f"  M7 drain:      {V_M7_drain_ref:.3f}V (= VDD)")
    
    print(f"\nOUTPUT SIDE (M2-M4-M6-M8 stack):")
    print(f"  GND:           0.000V")
    print(f"  M9 drain:      {V_tail_node:.3f}V (= M1/M2 sources)")
    print(f"  M2 drain:      {V_M2_drain_out:.3f}V (= M4 source)")
    print(f"  M4 drain:      {V_M4_drain_out:.3f}V (= M6 source) *** VOUT ***")
    print(f"  M6 drain:      {V_M6_drain_out:.3f}V (= M8 source)")
    print(f"  M8 drain:      {V_M8_drain_out:.3f}V (= VDD)")
    
    print(f"\nGATE VOLTAGES (Bias/Mirror connections):")
    print(f"  M1/M2 gates (inputs):   {Vcm:.3f}V (both at Vcm)")
    print(f"  M3/M4 gates (VB_casc):  {VB_casc:.3f}V (tied together)")
    print(f"  M5 gate (diode):        {V_M5_drain_ref:.3f}V (= M5 drain)")
    print(f"  M6 gate (mirrors M5):   {V_M5_drain_ref:.3f}V")
    print(f"  M7 gate (diode):        {VDD:.3f}V (= M7 drain = VDD)")
    print(f"  M8 gate (mirrors M7):   {VDD:.3f}V")
    print(f"  M9 gate (bias):         {VGS_tail:.3f}V")
    
    print(f"\nEXPECTED CURRENTS:")
    print(f"  I(M9) = {ID_tail*1e6:.3f}uA (tail current)")
    print(f"  I(M1) = I(M3) = I(M5) = I(M7) = {ID_in*1e6:.3f}uA (reference branch)")
    print(f"  I(M2) = I(M4) = I(M6) = I(M8) = {ID_in*1e6:.3f}uA (output branch)")
    print(f"\n  ** Both branches should have EQUAL currents of {ID_in*1e6:.3f}uA! **")
    print(f"\n{'='*70}\n")
    
    # STEP 7: Performance Summary
    print(f"\nSTEP 7: PERFORMANCE SUMMARY")
    
    # Output resistance at single-ended output node (M8/M6/M4/M2 drain)
    # Looking down: NMOS cascode (M4-M2) resistance
    # Looking up: PMOS cascode (M6-M8) resistance  
    Ron_actual = gm_casc_n * ro_casc_n * (ro_in + ro_tail)
    Rop_actual = gm_casc_p * ro_casc_p * (ro_src_p + ro_in)
    Rout1_actual = 1 / (1/Ron_actual + 1/Rop_actual)
    A1_actual = Gm1_required * Rout1_actual
    I_total = 2 * ID_in
    Power_stage1 = VDD * I_total
    
    print(f"  Gm1={Gm1_required*1e6:.2f}uS, Rout1={Rout1_actual/1e6:.2f}MOhm")
    print(f"  Gain A1={A1_actual:.1f}V/V ({20*np.log10(A1_actual):.1f}dB)")
    print(f"  Current={I_total*1e6:.2f}uA, Power={Power_stage1*1e3:.3f}mW")
    print(f"  Status: {'[PASS]' if A1_actual >= A1_target * 0.9 else '[FAIL]'}")
    
    design = {
        # Input Pair (M1, M2)
        'M1_M2': {
            'device': 'nch_2v',
            'L': L_in,
            'W': W_in,
            'gm_ID': gm_ID_in,
            'ID': ID_in,
            'VGS': VGS_in,
            'multiplicity': 2
        },
        # NMOS Cascode (M3, M4)
        'M3_M4': {
            'device': 'nch_1v',
            'L': L_casc_n,
            'W': W_casc_n,
            'gm_ID': gm_ID_casc_n,
            'ID': ID_casc_n,
            'VGS': VGS_casc_n,
            'multiplicity': 2
        },
        # PMOS Cascode (M5, M6)
        'M5_M6': {
            'device': 'pch_1v',
            'L': L_casc_p,
            'W': W_casc_p,
            'gm_ID': gm_ID_casc_p,
            'ID': ID_casc_p,
            'VSG': VSG_casc_p,
            'multiplicity': 2
        },
        # PMOS Current Sources (M7, M8)
        'M7_M8': {
            'device': 'pch_1v',
            'L': L_src_p,
            'W': W_src_p,
            'gm_ID': gm_ID_src_p,
            'ID': ID_src_p,
            'VSG': VSG_src_p,
            'multiplicity': 2
        },
        # Tail Current Source (M9)
        'M9': {
            'device': 'nch_1v',
            'L': L_tail,
            'W': W_tail,
            'gm_ID': gm_ID_tail,
            'ID': ID_tail,
            'VGS': VGS_tail,
            'multiplicity': 1
        },
        # Performance Metrics
        'performance': {
            'Gm1': Gm1_required,
            'Rout1': Rout1_actual,
            'A1': A1_actual,
            'Vcm': Vcm,
            'Vout_dc': Vout_dc,
            'I_total': I_total,
            'Power': Power_stage1,
            'VB_casc': VB_casc,
            'VDS_tail': VDS_tail,
            'VDS_in': VDS_in_actual,
            'VDS_casc_n': VDS_casc_n,
            'VSD_casc_p': VSD_casc_p,
            'VSD_src_p': VSD_src_p
        }
    }
    
    print(f"\n{'='*70}")
    print("DESIGN COMPLETE - All 9 transistors sized!")
    print("="*70 + "\n")
    
    return design

if __name__ == "__main__":
    result = design_telescopic_stage1()
    
    if result:
        print("\n" + "="*70)
        print("DEVICE SUMMARY TABLE")
        print("="*70)
        print(f"\n{'Device':<10} {'Type':<10} {'L(um)':<9} {'W(um)':<9} {'ID(uA)':<10} {'Count':<5}")
        print("-"*70)
        
        for name, params in result.items():
            if name != 'performance':
                print(f"{name:<10} {params['device']:<10} {params['L']:<9.2f} "
                      f"{params['W']:<9.2f} {params['ID']*1e6:<10.2f} {params['multiplicity']:<5}")
        
        print("-"*70)
        perf = result['performance']
        print(f"\nPower: {perf['Power']*1e3:.3f}mW")
        print(f"Gain: {perf['A1']:.1f}V/V ({20*np.log10(perf['A1']):.1f}dB)")
        print(f"\nBias Voltages:")
        print(f"  VB_casc (M3/M4 gates): {perf['VB_casc']:.3f}V")
        print(f"  Vcm (input CM): {perf['Vcm']:.3f}V")
        print(f"  Vout_dc (output DC): {perf['Vout_dc']:.3f}V")
