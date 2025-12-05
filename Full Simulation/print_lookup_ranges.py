"""
Print Valid Lookup Table Ranges
================================
Simple script to display all valid parameter values in the device lookup tables.
"""

import sys
import io
# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from look_up import importdata
import numpy as np

def print_device_ranges(device_name, filename):
    """Load device data and print all valid parameter ranges"""
    print("=" * 70)
    print(f"{device_name.upper()}")
    print("=" * 70)
    
    try:
        data = importdata(filename)
        
        # Print discrete parameter ranges
        print(f"\nValid L (Channel Length) values [um]:")
        L_vals = data['L']
        print(f"  {L_vals}")
        print(f"  Min: {np.min(L_vals):.3f} um, Max: {np.max(L_vals):.3f} um")
        print(f"  Count: {len(L_vals)}")
        
        print(f"\nValid VGS (Gate-Source Voltage) values [V]:")
        VGS_vals = data['VGS']
        print(f"  Range: {np.min(VGS_vals):.3f} V to {np.max(VGS_vals):.3f} V")
        print(f"  Count: {len(VGS_vals)}")
        if len(VGS_vals) <= 20:
            print(f"  Values: {VGS_vals}")
        else:
            print(f"  First 5: {VGS_vals[:5]}")
            print(f"  Last 5: {VGS_vals[-5:]}")
        
        print(f"\nValid VDS (Drain-Source Voltage) values [V]:")
        VDS_vals = data['VDS']
        print(f"  Range: {np.min(VDS_vals):.3f} V to {np.max(VDS_vals):.3f} V")
        print(f"  Count: {len(VDS_vals)}")
        if len(VDS_vals) <= 20:
            print(f"  Values: {VDS_vals}")
        else:
            print(f"  First 5: {VDS_vals[:5]}")
            print(f"  Last 5: {VDS_vals[-5:]}")
        
        print(f"\nValid VSB (Source-Bulk Voltage) values [V]:")
        VSB_vals = data['VSB']
        print(f"  Range: {np.min(VSB_vals):.3f} V to {np.max(VSB_vals):.3f} V")
        print(f"  Count: {len(VSB_vals)}")
        if len(VSB_vals) <= 20:
            print(f"  Values: {VSB_vals}")
        else:
            print(f"  First 5: {VSB_vals[:5]}")
            print(f"  Last 5: {VSB_vals[-5:]}")
        
        # Print info variables if available
        print(f"\nDevice Information:")
        if 'TEMP' in data:
            print(f"  Temperature: {data['TEMP']}")
        if 'CORNER' in data:
            print(f"  Corner: {data['CORNER']}")
        if 'NFING' in data:
            print(f"  Number of Fingers: {data['NFING']}")
        if 'INFO' in data:
            print(f"  Info: {data['INFO']}")
        
        # Sample achievable gm/Id range (at a typical operating point)
        print(f"\nSample GM_ID (gm/Id) Range:")
        try:
            # Use middle values for L, VDS, VSB
            L_sample = data['L'][len(data['L'])//2] if len(data['L']) > 0 else data['L'][0]
            VDS_sample = data['VDS'][len(data['VDS'])//2] if len(data['VDS']) > 0 else data['VDS'][0]
            VSB_sample = 0  # Usually 0 for most cases
            
            # Calculate GM_ID across all VGS values
            from look_up import look_up_basic
            gm_id_values = look_up_basic(data, 'GM_ID', vgs=data['VGS'], vds=VDS_sample, vsb=VSB_sample, l=L_sample)
            
            # Filter out invalid values (NaN, inf, negative)
            valid_gm_id = gm_id_values[np.isfinite(gm_id_values) & (gm_id_values > 0)]
            
            if len(valid_gm_id) > 0:
                print(f"  (at L={L_sample:.3f} um, VDS={VDS_sample:.3f} V, VSB={VSB_sample:.3f} V)")
                print(f"  Range: {np.min(valid_gm_id):.2f} to {np.max(valid_gm_id):.2f} [1/V]")
            else:
                print(f"  Could not determine valid range")
        except Exception as e:
            print(f"  Could not calculate GM_ID range: {e}")
        
        print()
        
    except Exception as e:
        print(f"  ERROR loading {filename}: {e}")
        print()


def main():
    print("=" * 70)
    print("LOOKUP TABLE PARAMETER RANGES")
    print("=" * 70)
    print()
    
    # List of device files to check
    devices = [
        ("NMOS 1V", "nch_1v.mat"),
        ("NMOS 2V", "nch_2v.mat"),
        ("PMOS 1V", "pch_1v.mat"),
        ("PMOS 2V", "pch_2v.mat"),
    ]
    
    for device_name, filename in devices:
        print_device_ranges(device_name, filename)


if __name__ == "__main__":
    main()

