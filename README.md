
# Rebel–Alliance Deterrence Model — v3 (Streamlit)

Adds **2D Explore** heatmaps and fixes to reliability/effectiveness overrides.

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy
Push to GitHub and point Streamlit Cloud at `streamlit_app.py`.

## New
- Heatmaps for π(d), π*, and rebellion when varying **two** characteristics at once (e.g., Reliability × Institutionalization).
- Reliability and λ (via l0) now properly override in Explore and Scenario Lab.
