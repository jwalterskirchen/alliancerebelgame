
# Rebel–Alliance Deterrence Model (Streamlit)

A simple, step‑by‑step Streamlit app that demonstrates a three‑player game between a **government**, **potential rebels**, and an **ally/alliance**. The app focuses on how **alliance design** (legalization, institutionalization, provisions) and **partner characteristics** shape:

1. the **probability** the ally intervenes (resolve / obligations), and  
2. the **effectiveness** of that intervention and constraints on the government.

## Live demo (deploy yourself)
1. Create a new GitHub repository and add these files: `streamlit_app.py` and `requirements.txt`.
2. On **Streamlit Community Cloud**, create a new app, select your repo, and set **Main file path** to `streamlit_app.py`.
3. Deploy.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Model (high level)
- **Contest success:** \( p_R(\tau)=\frac{\rho(d)}{\rho(d)+\tau},\;\rho(d)=\sqrt{\tfrac{v_R c_G(d)}{v_G c_R}} \)
- **Rebel contest payoff:** \( W_R(\tau;d)=v_R\,\frac{\rho(d)\left(\rho(d)+\tfrac12\tau\right)}{(\rho(d)+\tau)^2} \)
- **Ally’s net advantage to intervene:** \( \Delta_A(\theta,d)=W_A[p_R(1)-p_R(\lambda(\theta,d))]+F(d) \)
- **Intervention rule:** \( I^*(\theta,d)=\mathbbm{1}\{\Delta_A(\theta,d)\ge K_A(\theta)\} \), so \( \pi(d)=\mu I^*(H,d)+(1-\mu)I^*(L,d) \)
- **Rebels’ decision:** rebel iff \( \mathbb E[U_R\mid m{=}1,d]\ge S \), with
  \( \mathbb E[U_R\mid m{=}1,d]=(1-\pi)W_R(1;d)+\pi\,\overline W_R^{\,I}(d)-m_0(d)+g(d) \).  
  When \(W_R(1;d)>\overline W_R^{\,I}(d)\), this yields the deterrence threshold
  \( \pi(d)\le\pi^*(d)=\frac{W_R(1;d)-S-m_0(d)+g(d)}{W_R(1;d)-\overline W_R^{\,I}(d)}\).

## File list
- `streamlit_app.py` — the app (description + simulation + plots)
- `requirements.txt` — Python deps

## License
MIT — see header in `streamlit_app.py`.
