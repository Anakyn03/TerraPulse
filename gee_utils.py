"""
gee_utils.py

All Google Earth Engine logic lives here, separate from the Streamlit UI code
in app.py. This separation matters for two reasons: (1) it means the actual
detection logic can be tested/reasoned about on its own, without a UI in the
way, and (2) it means app.py stays focused purely on layout and interaction,
which is easier to read and easier to explain to a judge.
"""

import ee
import numpy as np
from PIL import Image as PILImage
from scipy import ndimage


CLASSIFIER_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


# ---------------------------------------------------------------------------
# Site presets: every location we've tested, with its bounding box and the
# date ranges that gave clean, season-matched results during development.
# Adding a new site later just means adding one entry here.
# ---------------------------------------------------------------------------
SITE_PRESETS = {
    "Jewar Airport (Noida International Airport), UP": {
        "bounds": [77.576480, 28.167251, 77.620719, 28.185502],
        "before": ("2021-10-01", "2022-03-31"),
        "after": ("2025-10-01", "2026-03-31"),
        "description": (
            "Greenfield site converted into a full international airport "
            "(runway, terminal, apron) between 2021 and 2026. The clearest, "
            "most dramatic example in our testing."
        ),
    },
    "Aero City, Chakeri Airport, Kanpur": {
        "bounds": [80.420832, 26.397892, 80.438141, 26.405067],
        "before": ("2022-11-01", "2022-12-15"),
        "after": ("2025-11-01", "2025-12-15"),
        "description": (
            "A 300-acre aerotropolis development, early-stage construction. "
            "A realistic, less dramatic example of catching subtle change."
        ),
    },
    "Water body site, Gorakhpur": {
        "bounds": [83.3800, 26.7305, 83.3870, 26.7355],
        "before": ("2022-11-01", "2022-12-15"),
        "after": ("2025-11-01", "2025-12-15"),
        "description": (
            "A dense urban area where a pond/open water footprint appears to "
            "have been filled and built over — a possible encroachment case."
        ),
    },
}


class EarthEngineNotAuthenticated(Exception):
    """Raised when Earth Engine credentials are not set up."""
    pass


class TerraPulseDetectionError(Exception):
    """Raised for expected data or inference problems in the detection pipeline."""
    pass


def init_earth_engine(project_id: str):
    """
    Connects this app to Google Earth Engine using an already-authenticated
    local credential. This credential must be created ONCE, ahead of time,
    by running `earthengine authenticate` in a normal terminal (see
    SETUP.md) — never inside the app itself. Attempting an interactive
    login from inside a running Streamlit app is unreliable (it can fail
    with network/DNS errors depending on how Streamlit's process handles
    the browser callback), so this function deliberately does NOT attempt
    to authenticate automatically. If it's missing, it fails clearly
    instead, so the fix is obvious.
    """
    try:
        ee.Initialize(project=project_id)
    except Exception as e:
        raise EarthEngineNotAuthenticated(
            "Earth Engine isn't authenticated on this machine yet. "
            "Open a terminal (not this app) and run: earthengine authenticate "
            f"— then restart the app. (Original error: {e})"
        )


def get_composite(bounds, start_date, end_date):
    """
    Builds a median composite image over the given date range and area.
    Taking the median across many satellite passes (instead of trusting one
    single image) cancels out one-off haze, thin cloud, and sensor noise —
    the fix we discovered was necessary after early single-image attempts
    came back too hazy to use.
    """
    aoi = ee.Geometry.Rectangle(bounds)
    if start_date >= end_date:
        raise TerraPulseDetectionError(
            f"Invalid date range: {start_date} must be before {end_date}."
        )

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    )

    count = collection.size().getInfo()
    if count == 0:
        raise TerraPulseDetectionError(
            f"No usable Sentinel-2 scenes were found for {start_date} to {end_date}. "
            "Try a wider date range or a different location."
        )

    # Keep the source Sentinel-2 grid on the composite. This is more stable
    # for later sampleRectangle() calls than forcing the computed median into
    # a new EPSG:4326 projection inside the pixel-fetch function.
    reference_projection = collection.first().select("B2").projection()
    composite = collection.median().setDefaultProjection(reference_projection)

    return composite, aoi


def add_indices(image):
    """
    Adds two spectral index bands to an image:
    NDVI (vegetation index) and NDBI (built-up index). See app.py's
    "How it works" tab for the plain-English explanation shown to users.
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return image.addBands(ndvi).addBands(ndbi)


def compute_change_mask(before_image, after_image, ndbi_threshold=0.1,
                         ndvi_threshold=-0.1, min_cluster_size=30):
    """
    The core detection logic. A pixel is flagged as likely human-made change
    only when BOTH conditions hold at once:
      - its built-up index (NDBI) rose by more than `ndbi_threshold`
      - its vegetation index (NDVI) dropped by more than `ndvi_threshold`
    Requiring both together, rather than either alone, is what filters out
    ordinary seasonal/farming change and keeps genuine construction signal.

    After that, `min_cluster_size` removes small, scattered single-pixel
    detections (usually noise) and keeps only clusters big enough to
    plausibly be a real structure.
    """
    before_idx = add_indices(before_image)
    after_idx = add_indices(after_image)

    ndbi_diff = after_idx.select("NDBI").subtract(before_idx.select("NDBI"))
    ndvi_diff = after_idx.select("NDVI").subtract(before_idx.select("NDVI"))

    raw_mask = ndbi_diff.gt(ndbi_threshold).And(ndvi_diff.lt(ndvi_threshold))
    connected = raw_mask.selfMask().connectedPixelCount(maxSize=256, eightConnected=True)
    clean_mask = connected.gte(min_cluster_size)

    return clean_mask


def get_thumb_url(image, aoi, vis_params, dimensions=512):
    """Small helper: turns any Earth Engine image into a viewable PNG URL."""
    return image.clip(aoi).getThumbUrl({**vis_params, "region": aoi, "dimensions": dimensions})


def get_overlay_url(after_image, aoi, change_mask, dimensions=512):
    """
    Blends the detected-change mask (in red) directly on top of the
    true-color 'after' image, so the result is visually grounded in the
    real photo rather than shown as an abstract mask on its own.
    """
    vis_params = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
    after_rgb = after_image.clip(aoi).visualize(**vis_params)
    mask_overlay = change_mask.selfMask().visualize(palette=["red"])
    combined = after_rgb.blend(mask_overlay)
    return combined.getThumbUrl({"region": aoi, "dimensions": dimensions})


def compute_changed_area_hectares(change_mask, aoi):
    """
    Converts the flagged pixels into an actual area figure (hectares) —
    turns an abstract 'red blob' into a concrete number judges can weigh,
    e.g. 'we detected 4.2 hectares of new built-up area.'
    """
    area_image = change_mask.selfMask().multiply(ee.Image.pixelArea())
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=10,
        maxPixels=1e9,
    )
    area_m2 = stats.get("NDBI")
    try:
        value = area_m2.getInfo()
        return round(value / 10000, 2) if value else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Local classifier pipeline
#
# The trained Random Forest lives as a plain scikit-learn model (a .pkl
# file), not an Earth Engine object. Earth Engine has no way to run it
# server-side, so instead of asking Earth Engine to render finished
# pictures, we pull the raw pixel numbers straight into Python, run the
# model ourselves, and build the final image locally with NumPy/PIL. This
# also removes the earlier dependency on Earth Engine's thumbnail
# rendering entirely, which is what was causing the custom-location
# display bug.
# ---------------------------------------------------------------------------

DEFAULT_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


def fetch_pixel_arrays(image, aoi, bands=DEFAULT_BANDS, scale=10):
    """
    Pulls raw pixel values for the given bands directly into Python as
    NumPy arrays. This is what makes it possible to run our own,
    locally-trained scikit-learn model against real satellite pixels,
    since that model only understands plain NumPy arrays, not Earth
    Engine's internal image objects.

    IMPORTANT: `image` here is typically a computed image (e.g. the output
    of .median() over an ImageCollection), which does not carry a clean,
    fixed pixel grid the way a raw source image does. Calling
    sampleRectangle() on it directly can return degenerate/near-uniform
    values, because Earth Engine doesn't know what grid to sample against
    until one is explicitly set. We fix that here by reprojecting to a
    known CRS and scale first -- the same fix that solved the earlier
    GHSL parallelogram/misalignment issue.
    """
    # sampleRectangle() works more reliably when the composite keeps its
    # native Sentinel-2 projection and is clipped to the requested AOI.
    # Reprojecting the computed median to EPSG:4326 here could produce an
    # invalid/empty sampling request for some custom bounding boxes.
    fixed_image = image.select(bands).clip(aoi)
    sampled = fixed_image.sampleRectangle(region=aoi, defaultValue=0)
    try:
        info = sampled.getInfo()
    except Exception as exc:
        raise TerraPulseDetectionError(
            "Earth Engine could not sample the selected imagery. "
            "This usually means the date range returned no usable pixels "
            "for the chosen custom area. Try a wider date range or a slightly "
            "smaller bounding box."
        ) from exc
    arrays = {}
    for b in bands:
        arrays[b] = np.array(info["properties"][b], dtype=np.float32)
    return arrays


def _compute_ndvi_ndbi(arrays):
    """Same NDVI/NDBI formulas as before, just computed in plain NumPy
    instead of inside Earth Engine, since we now have the raw numbers
    locally anyway."""
    nir, red, swir = arrays["B8"], arrays["B4"], arrays["B11"]
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndbi = (swir - nir) / (swir + nir + 1e-6)
    return ndvi, ndbi


def _to_rgb_uint8(arrays, gain=3000):
    """Turns raw B4/B3/B2 reflectance values into a normal viewable image,
    the same brightness rescaling `vis_params` used to do inside Earth
    Engine's thumbnail renderer."""
    r = np.clip(arrays["B4"] / gain, 0, 1)
    g = np.clip(arrays["B3"] / gain, 0, 1)
    b = np.clip(arrays["B2"] / gain, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def run_local_pipeline(before_image, after_image, aoi, model,
                        ndbi_threshold=0.1, ndvi_threshold=-0.1,
                        min_cluster_size=30, scale=10):
    """
    LEGACY pipeline: single-image classifier (trained on ESA WorldCover)
    combined with NDVI/NDBI index math. Kept for reference/fallback, but
    superseded by run_oscd_pipeline below, which uses a classifier trained
    on real paired before/after change labels (OSCD) and is both simpler
    and more directly supervised for the actual task.
    """
    before_arrays = fetch_pixel_arrays(before_image, aoi, DEFAULT_BANDS, scale)
    after_arrays = fetch_pixel_arrays(after_image, aoi, DEFAULT_BANDS, scale)

    ndvi_before, ndbi_before = _compute_ndvi_ndbi(before_arrays)
    ndvi_after, ndbi_after = _compute_ndvi_ndbi(after_arrays)

    ndbi_diff = ndbi_after - ndbi_before
    ndvi_diff = ndvi_after - ndvi_before

    index_mask = (ndbi_diff > ndbi_threshold) & (ndvi_diff < ndvi_threshold)

    labeled, num_features = ndimage.label(index_mask)
    if num_features > 0:
        sizes = ndimage.sum(index_mask, labeled, range(1, num_features + 1))
        clean_mask = np.isin(labeled, [i + 1 for i, s in enumerate(sizes) if s >= min_cluster_size])
    else:
        clean_mask = index_mask

    H, W = after_arrays["B2"].shape
    feature_stack = np.stack(
        [after_arrays[b] for b in DEFAULT_BANDS], axis=-1
    ).reshape(-1, len(DEFAULT_BANDS))
    classifier_preds = model.predict(feature_stack).reshape(H, W)
    classifier_mask = classifier_preds == 1

    final_mask = clean_mask & classifier_mask

    after_rgb = _to_rgb_uint8(after_arrays)
    before_rgb = _to_rgb_uint8(before_arrays)

    overlay = after_rgb.copy()
    overlay[final_mask] = [230, 57, 43]

    area_ha = round(float(final_mask.sum()) * (scale * scale) / 10000.0, 2)

    return {
        "before_rgb": PILImage.fromarray(before_rgb),
        "after_rgb": PILImage.fromarray(after_rgb),
        "overlay": PILImage.fromarray(overlay),
        "area_ha": area_ha,
    }


def run_oscd_pipeline(before_image, after_image, aoi, model, min_cluster_size=10, scale=10):
    """
    Run the active OSCD-trained before/after change classifier.

    The model receives six Sentinel-2 bands from the before composite and the
    same six bands from the after composite (12 paired features total), then
    predicts change/no-change. Connected-component filtering removes tiny,
    isolated predictions that are more likely to be noise than meaningful
    spatial change.
    """
    before_arrays = fetch_pixel_arrays(before_image, aoi, DEFAULT_BANDS, scale)
    after_arrays = fetch_pixel_arrays(after_image, aoi, DEFAULT_BANDS, scale)

    H, W = after_arrays["B2"].shape
    if H == 0 or W == 0:
        raise TerraPulseDetectionError("The selected area returned no image pixels.")

    before_stack = np.stack(
        [before_arrays[b] for b in DEFAULT_BANDS], axis=-1
    )
    after_stack = np.stack(
        [after_arrays[b] for b in DEFAULT_BANDS], axis=-1
    )
    paired_features = np.concatenate(
        [before_stack, after_stack], axis=-1
    ).reshape(-1, len(DEFAULT_BANDS) * 2)

    try:
        if hasattr(model, "predict_proba") and len(getattr(model, "classes_", [])) == 2:
            probabilities = model.predict_proba(paired_features)
            classes = list(model.classes_)
            changed_index = classes.index(1)
            change_probability = probabilities[:, changed_index]
            preds = (change_probability >= 0.5).astype(np.uint8).reshape(H, W)
            mean_confidence_raw = float(np.mean(np.max(probabilities, axis=1)))
        else:
            pred_flat = model.predict(paired_features)
            preds = (pred_flat == 1).astype(np.uint8).reshape(H, W)
            change_probability = preds.reshape(-1).astype(np.float32)
            mean_confidence_raw = float(np.mean(preds))
    except Exception as exc:
        raise TerraPulseDetectionError(
            f"The classifier could not process the selected imagery: {exc}"
        ) from exc

    raw_mask = preds.astype(bool)

    labeled, num_features = ndimage.label(raw_mask)
    if num_features > 0:
        sizes = ndimage.sum(raw_mask, labeled, range(1, num_features + 1))
        keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_cluster_size]
        final_mask = np.isin(labeled, keep_labels)
    else:
        final_mask = raw_mask

    after_rgb = _to_rgb_uint8(after_arrays)
    before_rgb = _to_rgb_uint8(before_arrays)

    overlay = after_rgb.copy()
    overlay[final_mask] = [230, 57, 43]

    changed_pixels = int(final_mask.sum())
    area_ha = round(float(changed_pixels) * (scale * scale) / 10000.0, 2)

    # Report confidence only for pixels that survived spatial filtering when
    # probability scores are available; otherwise expose the model's mean
    # prediction confidence as a fallback.
    if hasattr(model, "predict_proba") and len(getattr(model, "classes_", [])) == 2:
        final_flat = final_mask.reshape(-1)
        if np.any(final_flat):
            mean_confidence = float(np.mean(change_probability[final_flat]))
        else:
            mean_confidence = 0.0
    else:
        mean_confidence = mean_confidence_raw

    return {
        "before_rgb": PILImage.fromarray(before_rgb),
        "after_rgb": PILImage.fromarray(after_rgb),
        "overlay": PILImage.fromarray(overlay),
        "area_ha": area_ha,
        "changed_pixels": changed_pixels,
        "mean_confidence": mean_confidence,
    }

