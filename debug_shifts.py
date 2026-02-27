import numpy as np
import glob
import re
import os

files = glob.glob('C:/Users/allis/OneDrive/Desktop/run_bundles/my_dataset_local/**/*world_to_local_shifts*.raw', recursive=True)
if not files:
    print("No files found!")
    exit(1)

# try finding one that has non-zero data
for f in files:
    match = re.search(r'_(\d+)x(\d+)x(\d+)\.raw$', f)
    if not match: continue
    nx, ny, nz = int(match.group(1)), int(match.group(2)), int(match.group(3))
    data = np.fromfile(f, dtype=np.float32)
    if not np.all(data == 0):
        print(f"Testing file: {f}")
        print(f"Dimensions: {nx}x{ny}x{nz}")
        break
else:
    print("All files are complete zeros! Using the first one anyway.")
    f = files[0]
    match = re.search(r'_(\d+)x(\d+)x(\d+)\.raw$', f)
    nx, ny, nz = int(match.group(1)), int(match.group(2)), int(match.group(3))
    data = np.fromfile(f, dtype=np.float32)

data_3d = data.reshape((nx, ny, nz, 3))

print("\n--- Raw Data Stats ---")
print(f"X min/med/max: {np.min(data_3d[...,0]):.2f} / {np.median(data_3d[...,0]):.2f} / {np.max(data_3d[...,0]):.2f}")
print(f"Y min/med/max: {np.min(data_3d[...,1]):.2f} / {np.median(data_3d[...,1]):.2f} / {np.max(data_3d[...,1]):.2f}")
print(f"Z min/med/max: {np.min(data_3d[...,2]):.2f} / {np.median(data_3d[...,2]):.2f} / {np.max(data_3d[...,2]):.2f}")

baseline_shift = np.median(data_3d, axis=(0,1,2), keepdims=True)
print(f"\nTile Baseline Rigid Shift (Median): X={baseline_shift[0,0,0,0]:.2f}, Y={baseline_shift[0,0,0,1]:.2f}, Z={baseline_shift[0,0,0,2]:.2f}")

warping_field = data_3d - baseline_shift
mag_warping = np.linalg.norm(warping_field, axis=-1)
print(f"||field - baseline|| non-linear warping magnitude min/med/max: {np.min(mag_warping):.2f} / {np.median(mag_warping):.2f} / {np.max(mag_warping):.2f}")

# Also check how many vectors exceed 10% overlap width (e.g. if overlap is 15% of 323 = 48 px. 10Px is a lot).
print(f"\nSanity Check: % vectors > 5px displacement: {np.mean(mag_warping > 5)*100:.2f}%")
print(f"Sanity Check: % vectors > 10px displacement: {np.mean(mag_warping > 10)*100:.2f}%")
print(f"Sanity Check: % vectors > 50px displacement: {np.mean(mag_warping > 50)*100:.2f}%")


