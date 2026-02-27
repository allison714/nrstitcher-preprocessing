import glob
import os
import re
import numpy as np

bundle_dir = r'C:\Users\allis\OneDrive\Desktop\run_bundles\my_dataset_local'
shift_files = glob.glob(os.path.join(bundle_dir, 'trace', '*world_to_local_shifts_*.raw'))
if not shift_files:
    shift_files = glob.glob(os.path.join(bundle_dir, '*world_to_local_shifts_*.raw'))
shift_files.sort()

print(f"Found {len(shift_files)} files")

for f in shift_files[:3]:
    match = re.search(r'_(\d+)x(\d+)x(\d+)\.raw$', f)
    if not match:
        print(f"Regex mismatch: {os.path.basename(f)}")
        continue
        
    nx, ny, nz = int(match.group(1)), int(match.group(2)), int(match.group(3))
    expected_size = nx * ny * nz * 3
    
    data = np.fromfile(f, dtype=np.float32)
    print(f"File {os.path.basename(f)}: parsed length {len(data)}, expected {expected_size}")
