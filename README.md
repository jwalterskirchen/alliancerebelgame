
# Rebel–Alliance Deterrence Model — v3.1 (Streamlit)

**Fix:** Scenario Lab no longer assumes you selected exactly three axes. It now uses `itertools.product` so 1–3 axes work without indexing errors.

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy
Push to GitHub and point Streamlit Cloud at `streamlit_app.py`.
