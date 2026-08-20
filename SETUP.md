# Running TerraPulse locally (VS Code)

## 1. Open the project

Open the `TerraPulse_app` folder in VS Code and open **Terminal → New Terminal**.

## 2. Create a virtual environment

```text
python -m venv venv
```

Activate it:

**Windows**
```text
venv\Scripts\activate
```

**Mac/Linux**
```text
source venv/bin/activate
```

## 3. Install dependencies

```text
pip install -r requirements.txt
```

## 4. Add the trained classifier

Place the trained OSCD Random Forest file in the project root with this exact name:

```text
terrapulse_oscd_classifier.pkl
```

The app will show a clear warning and will not run detection until this file exists.

## 5. Authenticate Earth Engine once

In a normal VS Code terminal:

```text
earthengine authenticate
```

Complete the browser sign-in, then return to the terminal.

## 6. Run TerraPulse

```text
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Recommended demo flow

1. Select **Jewar Airport (Noida International Airport), UP**.
2. Keep the default date ranges.
3. Keep the default cluster size unless you have already tested another value.
4. Click **Run detection**.
5. Show the before/after comparison.
6. Show the red change overlay and estimated changed area.
7. Explain that the red regions are model-detected change clusters at Sentinel-2 scale, not survey-grade building footprints.

## Before presenting

Run the app once on the exact laptop and network you will use for the presentation. Confirm:

- Earth Engine authentication works.
- `terrapulse_oscd_classifier.pkl` is present.
- The Jewar preset completes without an error.
- The before/after comparison renders.
- The change overlay renders.
- The statistics card shows area, changed pixels and model confidence.
- A custom bounding box is only used after the preset demo is confirmed.


## Custom location mode

1. Select **✏️ Custom location**.
2. Enter `min_lon, min_lat, max_lon, max_lat`.
3. Choose the before and after date ranges.
4. Click **Apply custom location**.
5. Wait for the green confirmation message.
6. Click **Run detection**.

For this presentation prototype, keep the custom area reasonably small (the app rejects very large areas to avoid slow local pixel processing). Use a date range wide enough to contain usable Sentinel-2 observations.
