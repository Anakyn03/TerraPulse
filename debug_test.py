"""
debug_test.py — run this directly (python debug_test.py) to isolate the
pixel-fetching problem, without going through the full Streamlit app.
Much faster to iterate on than clicking through the UI each time.
"""

import ee
from gee_utils import get_composite, fetch_pixel_arrays, SITE_PRESETS

ee.Initialize(project='quantum-star-475304-t3')

site = SITE_PRESETS["Jewar Airport (Noida International Airport), UP"]
before_img, aoi = get_composite(site["bounds"], *site["before"])

print("Fetching pixel arrays for the 'before' image...")
arrays = fetch_pixel_arrays(before_img, aoi)

for band, arr in arrays.items():
    print(f"{band}: shape={arr.shape}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}")
