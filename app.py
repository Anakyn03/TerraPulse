"""
TerraPulse — human-activity change detection from satellite imagery.
Run with: streamlit run app.py

The UI lives here; Earth Engine and detection logic live in gee_utils.py.
"""

from pathlib import Path
from datetime import date
import math

import joblib
import streamlit as st
from streamlit_image_comparison import image_comparison

from gee_utils import (
    SITE_PRESETS,
    EarthEngineNotAuthenticated,
    TerraPulseDetectionError,
    get_composite,
    init_earth_engine,
    run_oscd_pipeline,
)


MODEL_FILENAME = "terrapulse_oscd_classifier.pkl"


@st.cache_resource
def load_classifier():
    """Load the OSCD-trained classifier once per Streamlit session."""
    model_path = Path(__file__).parent / MODEL_FILENAME
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load {MODEL_FILENAME}: {exc}") from exc


def render_html(html: str) -> None:
    """Render HTML reliably across recent Streamlit versions."""
    html_renderer = getattr(st, "html", None)
    if html_renderer is not None:
        html_renderer(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


st.set_page_config(
    page_title="TerraPulse",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(1.5px 1.5px at 20% 30%, rgba(255,255,255,0.25) 1px, transparent 0),
            radial-gradient(1.5px 1.5px at 70% 15%, rgba(255,255,255,0.2) 1px, transparent 0),
            radial-gradient(1px 1px at 40% 70%, rgba(255,255,255,0.15) 1px, transparent 0),
            radial-gradient(1.5px 1.5px at 85% 60%, rgba(255,255,255,0.2) 1px, transparent 0),
            radial-gradient(1px 1px at 10% 85%, rgba(255,255,255,0.15) 1px, transparent 0),
            radial-gradient(1px 1px at 55% 45%, rgba(255,255,255,0.1) 1px, transparent 0),
            linear-gradient(160deg, #060a12 0%, #0c1a1e 35%, #10241c 65%, #16301f 100%);
        background-attachment: fixed;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(30,74,122,0.35), rgba(74,155,110,0.25));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        margin: 0;
        background: linear-gradient(90deg, #eaf3e6, #9fd6b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #a9c2b5;
        margin: 0.3rem 0 0;
        font-size: 1.05rem;
    }

    .metric-card {
        background: rgba(27, 38, 33, 0.65);
        backdrop-filter: blur(6px);
        padding: 1.3rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .card-label { color: #9fb5a8; font-size: 0.85rem; margin: 0 0 2px; }
    .card-value { color: #eaf3e6; font-size: 1.05rem; margin: 0 0 16px; }
    .card-value-big {
        color: #eaf3e6; font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem; font-weight: 700; margin: 0 0 16px;
    }
    .card-caption { color: #a9c2b5; font-size: 0.85rem; line-height: 1.5; margin: 0; }
    .card-hr { border-color: rgba(255,255,255,0.08); margin: 14px 0; }
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(111,207,151,0.10);
        color: #bfe8ce;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .method-grid { margin-top: 0.4rem; }
    .method-step {
        padding: 0.65rem 0.75rem;
        margin: 0.35rem 0;
        border-radius: 9px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .method-step b { color: #eaf3e6; }
    .muted { color: #9fb5a8; }

    section[data-testid="stSidebar"] {
        background: rgba(6, 10, 18, 0.75);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        padding: 0.6rem 1.2rem;
    }
    .stTabs [aria-selected="true"] { color: #6fcf97 !important; }
    .stButton button { border-radius: 10px !important; font-weight: 600; }
    img { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🛰️ TerraPulse")
    st.caption("Sentinel-2 multi-temporal change detection")

    project_id = st.text_input(
        "Earth Engine project ID",
        value="quantum-star-475304-t3",
        help="Google Earth Engine Cloud project used by this deployment.",
    )

    site_name = st.selectbox(
        "Choose a site", list(SITE_PRESETS.keys()) + ["✏️ Custom location"]
    )

    if site_name == "✏️ Custom location":
        # Keep custom-location inputs together so Streamlit reruns do not
        # accidentally clear a previously validated location.
        if "custom_site" not in st.session_state:
            st.session_state.custom_site = None

        with st.form("custom_location_form", clear_on_submit=False):
            st.caption("Enter: min_lon, min_lat, max_lon, max_lat")
            bbox_text = st.text_input(
                "Bounding box",
                value="",
                placeholder="77.576480, 28.167251, 77.620719, 28.185502",
                help="Longitude first, then latitude: min_lon, min_lat, max_lon, max_lat.",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                before_start = st.date_input(
                    "Before: start date",
                    value=date(2022, 11, 1),
                )
                before_end = st.date_input(
                    "Before: end date",
                    value=date(2022, 12, 15),
                )
            with col_b:
                after_start = st.date_input(
                    "After: start date",
                    value=date(2025, 11, 1),
                )
                after_end = st.date_input(
                    "After: end date",
                    value=date(2025, 12, 15),
                )

            apply_custom = st.form_submit_button(
                "Apply custom location",
                type="primary",
                use_container_width=True,
            )

        if apply_custom:
            try:
                parts = [float(x.strip()) for x in bbox_text.split(",")]
                if len(parts) != 4:
                    raise ValueError("Bounding box needs exactly four values.")
                min_lon, min_lat, max_lon, max_lat = parts
                if not (-180 <= min_lon < max_lon <= 180):
                    raise ValueError("Longitude values must satisfy -180 ≤ min_lon < max_lon ≤ 180.")
                if not (-90 <= min_lat < max_lat <= 90):
                    raise ValueError("Latitude values must satisfy -90 ≤ min_lat < max_lat ≤ 90.")
                if before_start >= before_end:
                    raise ValueError("Before end date must be after before start date.")
                if after_start >= after_end:
                    raise ValueError("After end date must be after after start date.")

                center_lat = (min_lat + max_lat) / 2.0
                width_m = (max_lon - min_lon) * 111320 * max(0.2, math.cos(math.radians(center_lat)))
                height_m = (max_lat - min_lat) * 111320
                approx_pixels = max(1, int(width_m / 10)) * max(1, int(height_m / 10))
                if approx_pixels > 250000:
                    raise ValueError(
                        "Custom area is too large for this live prototype. "
                        "Please choose a smaller bounding box (roughly ≤ 250,000 pixels at 10 m)."
                    )

                st.session_state.custom_site = {
                    "bounds": [min_lon, min_lat, max_lon, max_lat],
                    "before": (str(before_start), str(before_end)),
                    "after": (str(after_start), str(after_end)),
                    "description": "Custom location — results depend on imagery quality and site conditions.",
                }
                st.session_state.custom_site_message = "Custom location applied. Click Run detection."
            except ValueError as exc:
                st.session_state.custom_site = None
                st.session_state.custom_site_message = f"Custom location not ready: {exc}"

        site = st.session_state.get("custom_site")
        if site:
            st.success(st.session_state.get("custom_site_message", "Custom location applied."))
        else:
            message = st.session_state.get(
                "custom_site_message",
                "Enter a bounding box, choose dates, then click Apply custom location.",
            )
            st.info(message)
    else:
        site = SITE_PRESETS[site_name]
        st.caption(site["description"])

    with st.expander("Detection settings"):
        min_cluster = st.slider(
            "Minimum connected cluster (pixels)",
            min_value=1,
            max_value=100,
            value=30,
            step=1,
            help="Small isolated predictions are removed as likely noise.",
        )
        st.caption("The active model is an OSCD-trained before/after Random Forest.")

    run_button = st.button(
        "Run detection",
        type="primary",
        use_container_width=True,
        disabled=(site is None),
    )

render_html(
    """
    <div class="hero">
        <h1>🛰️ TerraPulse</h1>
        <p>Detecting meaningful change from multi-temporal Sentinel-2 imagery</p>
    </div>
    """
)

tab_detect, tab_how, tab_impact = st.tabs(["🔍 Live Detection", "📖 How It Works", "🌍 Impact"])

with tab_detect:
    try:
        classifier_model = load_classifier()
    except RuntimeError as exc:
        st.error(str(exc))
        classifier_model = None

    model_path = Path(__file__).parent / MODEL_FILENAME
    if classifier_model is None:
        st.warning(
            f"{MODEL_FILENAME} was not found in the project folder. "
            "Add the trained model file before running detection."
        )
    else:
        st.caption("Model: OSCD-trained Random Forest • Input: paired Sentinel-2 bands • Resolution: 10 m")

    if run_button and classifier_model is not None and site is not None:
        try:
            with st.spinner("Connecting to Earth Engine..."):
                init_earth_engine(project_id)

            with st.spinner("Fetching before/after Sentinel-2 composites..."):
                before_img, aoi = get_composite(site["bounds"], *site["before"])
                after_img, _ = get_composite(site["bounds"], *site["after"])

            with st.spinner("Running OSCD change classifier and spatial filtering..."):
                result = run_oscd_pipeline(
                    before_img, after_img, aoi, classifier_model, min_cluster
                )

            st.success("Detection completed.")

            st.markdown("#### Before / after comparison")
            image_comparison(
                img1=result["before_rgb"],
                img2=result["after_rgb"],
                label1=f"Before ({site['before'][0][:4]})",
                label2=f"After ({site['after'][0][:4]})",
                width=850,
            )

            st.markdown("#### Detected change")
            col_img, col_stats = st.columns([2, 1])
            with col_img:
                st.image(result["overlay"], use_container_width=True)
            with col_stats:
                render_html(
                    f"""
                    <div class="metric-card">
                        <span class="status-pill">OSCD + Sentinel-2</span>
                        <p class="card-label" style="margin-top: 14px;">Estimated changed area</p>
                        <p class="card-value-big">{result['area_ha']} ha</p>
                        <p class="card-label">Site</p>
                        <p class="card-value">{site_name.split(',')[0]}</p>
                        <p class="card-label">Time span</p>
                        <p class="card-value">{site['before'][0][:4]} → {site['after'][0][:4]}</p>
                        <p class="card-label">Detected pixels</p>
                        <p class="card-value">{result['changed_pixels']:,}</p>
                        <p class="card-label">Mean model confidence</p>
                        <p class="card-value">{result['mean_confidence']:.1%}</p>
                        <hr class="card-hr">
                        <p class="card-caption">
                            Red regions are pixels classified as change from paired before/after Sentinel-2 spectral values.
                            Small isolated regions are removed using connected-component filtering.
                        </p>
                    </div>
                    """
                )

            st.caption(
                "Interpretation: the red overlay marks model-detected change clusters; "
                "it is not a survey-grade building footprint or proof of a specific cause."
            )

        except EarthEngineNotAuthenticated as exc:
            st.error(str(exc))
        except TerraPulseDetectionError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
    elif not run_button:
        st.info("Choose a site and click **Run detection** to begin.")

with tab_how:
    render_html(
        """
        <div class="metric-card">
            <h4>How TerraPulse works</h4>
            <div class="method-grid">
                <div class="method-step"><b>1. Sentinel-2 input</b><br><span class="muted">Two time periods are collected for the selected area through Google Earth Engine.</span></div>
                <div class="method-step"><b>2. Median composite</b><br><span class="muted">Multiple observations in each period are combined to reduce one-off haze and cloud effects.</span></div>
                <div class="method-step"><b>3. Paired spectral features</b><br><span class="muted">The model receives six Sentinel-2 bands from the before image and the same six bands from the after image.</span></div>
                <div class="method-step"><b>4. OSCD Random Forest</b><br><span class="muted">A supervised classifier predicts whether the paired pixel represents genuine change.</span></div>
                <div class="method-step"><b>5. Spatial filtering</b><br><span class="muted">Connected-component filtering removes small isolated predictions and keeps meaningful clusters.</span></div>
                <div class="method-step"><b>6. Output</b><br><span class="muted">TerraPulse shows the before/after images, a red change overlay and an estimated changed area.</span></div>
            </div>
        </div>
        """
    )

with tab_impact:
    render_html(
        """
        <div class="metric-card">
            <h4>Potential use</h4>
            <p class="muted">TerraPulse is designed as a lightweight monitoring prototype for detecting spatially meaningful change over selected areas.</p>
            <div class="method-grid">
                <div class="method-step"><b>Construction monitoring</b><br><span class="muted">Identify areas showing significant land-surface change over time.</span></div>
                <div class="method-step"><b>Urban monitoring</b><br><span class="muted">Support periodic review of growth and infrastructure development.</span></div>
                <div class="method-step"><b>Screening, not final enforcement</b><br><span class="muted">Detected regions can be used to prioritize locations for human inspection.</span></div>
            </div>
            <hr class="card-hr">
            <p class="card-caption"><b>Current scope:</b> TerraPulse operates on Sentinel-2 10 m imagery and identifies change at pixel/cluster scale. It is not an individual vehicle or aircraft detector and is not a substitute for high-resolution survey imagery.</p>
        </div>
        """
    )
