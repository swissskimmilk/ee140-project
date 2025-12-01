# EE140/240A LCD Driver Amplifier – README

## 1. Project-Level Specs (from handout)

- **Process:** gpdk045  
- **Supplies:**  
  - VDDL ∈ [0, 1.1] V  
  - VDDH ∈ [0, 1.8] V  
  - GND = 0 V  
- **Closed-loop gain:** 2  
- **Load:** RL ≈ 1 kΩ, CL = 25 pF (settling measured at capacitor node)  
- **Total error:** ≤ 0.2%  
- **Power (EE140):** ≤ 1.25 mW  
- **Phase margin:** ≥ 45°  
- **CMRR:** ≥ 55 dB  
- **PSRR:** ≥ 50 dB  
- **Explicit caps:** ≤ 10 pF  
- **Explicit resistors:** ≤ 200 kΩ  
- **Input devices:** must use nmos2v  
- **Allowed sources:** 1 ideal current source, 2 supply sources (VDDL/VDDH), 1 ground  
- **Output swing:** ≥ 1.4 V

---

## 2. System-Level Timing & Error Budget

### Settling time
- Display: 272×340 = 92,480 px  
- Refresh: 60 Hz → frame = 16.67 ms  
- Pixel update time:
  ```
  T_pixel ≈ 180 ns
  ```
  (upper bound for settling)

---

## 3. Configuration and Design Parameters

### Designer-Adjustable Parameters (`config.py`)
All specification values and design choices that you may want to tune are centralized in `config.py`:

**System Specs:**
- Supply voltages (VDDL_MAX, VDDH_MAX)
- Closed-loop gain (G_CLOSED_LOOP)
- Load components (RL, CL)
- Error budget (TOTAL_ERROR_SPEC)
- Power budget (POWER_MAX)
- Phase margin requirement (PHASE_MARGIN_MIN)
- Output swing requirement (OUTPUT_SWING_MIN)

**Timing:**
- Refresh rate (REFRESH_RATE)
- Pixel time (T_PIXEL)

**Design Choices (tune these for optimization):**
- Error split between static and dynamic (ERROR_SPLIT_STATIC, ERROR_SPLIT_DYNAMIC)
- Compensation capacitor value (CC)

### Calculated Design Parameters (`calculate_design_params.py`)
This script imports values from `config.py` and calculates derived requirements:
- Required DC gain (A0)
- Unity gain frequency (f_u)
- Stage transconductances (gm1, gm2)
- Nulling resistor options (Rz)
- Gain distribution between stages

Run `python calculate_design_params.py` to see all calculated parameters.

**Usage:** To adjust your design, edit values in `config.py`, then re-run the design scripts.

---

## 4. Amplifier Topology

### Stage 2: Class AB Common-Source Output Stage
This design uses a **common-source Class AB** topology (also called "high output swing"):

```
        VDDH
         |
    [PMOS Mp] ← gate: VBias_p
         |
        Vout ──[RL=1kΩ]──[CL=25pF]── GND
         |
    [NMOS Mn] ← gate: VBias_n
         |
        GND
```

**Key characteristics:**
- Both transistor DRAINS connect to the output node
- Gates are biased for Class AB operation (both partially on at quiescent)
- Output resistance: Rout = ro_n || ro_p ≈ 1/(gds_n + gds_p)
- Small-signal gain: A2 = gm × (Rout || RL)
- High voltage swing capability (rail-to-rail possible)

**NOT a source follower** (where sources connect to output with Rout ≈ 1/gm)

---

## 5. File Organization

### Main Design Scripts
- `config.py` - Central configuration (edit this to tune design)
- `calculate_design_params.py` - Derives requirements from specs
- `design_stage1.py` - Complete telescopic amplifier design (9 transistors)
- `design_output_stage.py` - Class AB output stage design
- `analyze_stability.py` - Stability analysis with Bode plots
- `calculate_settling.py` - Settling time simulation
- `design_report.py` - Comprehensive design report

### Utilities
- `look_up.py` - MOSFET lookup table utilities (do not edit)
- `inspect_2v.py` - Quick utility to check 2V device data

### Data Files
- `nch_1v.mat`, `pch_1v.mat` - 1V device characterization data
- `nch_2v.mat`, `pch_2v.mat` - 2V device characterization data

### Examples
- `examples/` - Reference examples and tutorials (not part of main design)
  - `main.py` - Lab 2 gm/ID methodology tutorial exercises
  - See `examples/README_EXAMPLES.md` for details

---

## 5. Design Flow

1. **Configure:** Edit `config.py` to set specs and tune parameters
2. **Calculate:** Run `calculate_design_params.py` to see derived requirements
3. **Design Stages:** 
   - Run `design_stage1.py` for telescopic input stage
   - Run `design_output_stage.py` for Class AB output stage
4. **Analyze:** 
   - Run `analyze_stability.py` for frequency response and stability
   - Run `calculate_settling.py` for transient response
5. **Report:** Run `design_report.py` for complete specification check

---

## 6. Device Selection Rules (IMPORTANT)

**Important notes on V1/V2 mosfets:**
- v2 NMOS must be used for anything that is connected to the input
- For any branch that has VDD > 1.1V, the v2 MOSFETs must be used, even if a specific MOSFET should not have the full VDD across itself 
- Otherwise v1 can be used 