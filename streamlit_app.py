
# streamlit_app.py
# Rebel–Alliance Deterrence Model: a simple step-by-step app
# Author: (Your name or institution)
# License: MIT
#
# This app implements a three-player game (Government, Rebels, Alliance/Ally)
# and simulates the rebels' mobilization decision as a function of alliance design.
#
# It follows the model laid out in the accompanying README and in-app overview.
#
# Run locally:
#   pip install -r requirements.txt
#   streamlit run streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import json
import math
import matplotlib.pyplot as plt

# ---------------------- Streamlit setup ----------------------
st.set_page_config(
    page_title="Rebel–Alliance Deterrence Model",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Utility & math ----------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def rho(v_R, v_G, c_R, c_G):
    # relative "offense–defense" index
    # rho(d) = sqrt((v_R * c_G(d)) / (v_G * c_R))
    return math.sqrt(max(1e-12, (v_R * c_G) / (v_G * c_R)))

def pR(tau, rho_val):
    # rebel victory probability at the war-stage equilibrium
    # p_R(τ) = ρ / (ρ + τ)
    return rho_val / (rho_val + tau)

def WR(tau, rho_val, v_R):
    # rebels' equilibrium contest payoff (pre status-quo terms)
    # W_R(τ;d) = v_R * [ ρ (ρ + τ/2) ] / (ρ + τ)^2
    num = rho_val * (rho_val + 0.5 * tau)
    den = (rho_val + tau) ** 2
    return v_R * num / den

def lambda_base(L, C, l0, lL, lC):
    # intervene effectiveness base (must be > 1)
    val = 1.0 + l0 + lL * L + lC * C
    return max(1.01, val)

def lambda_type(lam_base, kappa):
    # type-specific multiplier (ensure > 1.01)
    return max(1.01, lam_base * kappa)

def cG(cG0, P, L, alphaP, betaL):
    # government marginal coercion cost under provisions & institutionalization
    # c_G(d) = cG0 * (1 + α_P P - β_L L), bounded below
    mult = 1.0 + alphaP * P - betaL * L
    return max(1e-6, cG0 * mult)

def F_design(F0, L, P, D, fL, fP, fD):
    # reneging/obligation penalty as a function of design & partner regime
    return max(0.0, F0 + fL * L + fP * P + fD * D)

def K_cost(baseline, L, C, D, kL, kC, kD):
    # ally's cost of intervention
    # decreases with L and C, increases with D (audience/casualty aversion)
    return max(0.0, baseline - kL * L - kC * C + kD * D)

def m0_cost(m0_base, P, mP):
    # fixed mobilization cost (falls with liberalizing provisions)
    return max(0.0, m0_base * (1.0 - mP * P))

def compute_all(
    # design axes
    L, P, C, D,
    # context & baselines
    v_R, v_G, c_R, cG0,
    mu, W_A, S, m0_base, g_shift,
    # mapping coefficients
    F0, fL, fP, fD,
    l0, lL, lC, kappa_H, kappa_L,
    alphaP, betaL,
    K_H0, K_L0, kL, kC, kD,
    mP
):
    # Derived design-dependent primitives
    lam_base = lambda_base(L, C, l0, lL, lC)
    lam_H = lambda_type(lam_base, kappa_H)
    lam_L = lambda_type(lam_base, kappa_L)

    cG_val = cG(cG0, P, L, alphaP, betaL)
    rho_val = rho(v_R, v_G, c_R, cG_val)

    F_val = F_design(F0, L, P, D, fL, fP, fD)
    K_H = K_cost(K_H0, L, C, D, kL, kC, kD)
    K_L = K_cost(K_L0, L, C, D, kL, kC, kD)

    # War-stage values
    W1 = WR(1.0, rho_val, v_R)
    WH = WR(lam_H, rho_val, v_R)
    WL = WR(lam_L, rho_val, v_R)

    # Ally intervention net advantage by type
    pR_noI = pR(1.0, rho_val)
    pR_lamH = pR(lam_H, rho_val)
    pR_lamL = pR(lam_L, rho_val)

    Delta_H = W_A * (pR_noI - pR_lamH) + F_val
    Delta_L = W_A * (pR_noI - pR_lamL) + F_val

    I_H = 1 if Delta_H >= K_H else 0
    I_L = 1 if Delta_L >= K_L else 0

    pi = mu * I_H + (1 - mu) * I_L

    # Average rebel payoff if intervention occurs (conditional on intervening types if any)
    if I_H + I_L > 0:
        weight = 0.0
        acc = 0.0
        if I_H:
            acc += mu * WH
            weight += mu
        if I_L:
            acc += (1 - mu) * WL
            weight += (1 - mu)
        WR_Ibar = acc / max(1e-9, weight)
    else:
        # Counterfactual mix (if no type would intervene at this design)
        WR_Ibar = mu * WH + (1 - mu) * WL

    # Rebels' expected utility if they mobilize (given design d)
    m0 = m0_cost(m0_base, P, mP)
    EU_m1 = (1 - pi) * W1 + pi * WR_Ibar - m0 + g_shift

    # Threshold logic
    denom = W1 - WR_Ibar
    numer = W1 - S - m0 + g_shift

    threshold = None
    threshold_note = ""
    if abs(denom) < 1e-9:
        threshold = None
        threshold_note = "Intervention does not change rebels' contest payoff (denominator ≈ 0); no π-threshold is defined—rebellion depends only on status-quo vs. mobilization cost."
    elif denom > 0:
        threshold = numer / denom
        threshold_note = "Intervention hurts rebels on average (denominator > 0). They rebel iff π ≤ π*."
    else:
        threshold = numer / denom  # negative denominator
        threshold_note = "Intervention helps rebels on average (denominator < 0). They rebel iff π ≥ π† (note inequality flips)."

    # Decision: rebel if EU_m1 ≥ S
    rebel = EU_m1 >= S

    # Pack everything
    out = {
        "design": {"L": L, "P": P, "C": C, "D": D},
        "context": {
            "v_R": v_R, "v_G": v_G, "c_R": c_R, "c_G0": cG0,
            "mu": mu, "W_A": W_A, "S": S, "m0_base": m0_base, "g_shift": g_shift
        },
        "mapping": {
            "F0": F0, "fL": fL, "fP": fP, "fD": fD,
            "l0": l0, "lL": lL, "lC": lC, "kappa_H": kappa_H, "kappa_L": kappa_L,
            "alphaP": alphaP, "betaL": betaL,
            "K_H0": K_H0, "K_L0": K_L0, "kL": kL, "kC": kC, "kD": kD,
            "mP": mP
        },
        "derived": {
            "lambda_base": lam_base, "lambda_H": lam_H, "lambda_L": lam_L,
            "c_G(d)": cG_val, "rho(d)": rho_val,
            "F(d)": F_val, "K_H": K_H, "K_L": K_L,
            "pR_noI": pR_noI, "pR_lambda_H": pR_lamH, "pR_lambda_L": pR_lamL,
            "W_R_noI": W1, "W_R_lambda_H": WH, "W_R_lambda_L": WL,
            "Delta_H": Delta_H, "Delta_L": Delta_L,
            "I_H": I_H, "I_L": I_L, "pi(d)": pi,
            "WR_Ibar": WR_Ibar, "EU_m1": EU_m1, "denom": denom, "numer": numer
        },
        "threshold": {"pi_star": threshold, "note": threshold_note},
        "decision": {"rebel": bool(rebel)}
    }
    return out

def plot_pi_vs_threshold(xgrid, xname, compute_fn):
    # Build series: π(d), π*(d) (or flipped threshold), rebellion indicator
    pis = []
    pistars = []
    rebel_flags = []
    flip_flags = []
    for x in xgrid:
        res = compute_fn(x)
        pi = res["derived"]["pi(d)"]
        pi_star = res["threshold"]["pi_star"]
        note = res["threshold"]["note"]
        rebel = res["decision"]["rebel"]
        pis.append(pi)
        pistars.append(pi_star if pi_star is not None else np.nan)
        rebel_flags.append(1 if rebel else 0)
        flip_flags.append(1 if "flips" in note or "denominator < 0" in note else 0)

    df = pd.DataFrame({xname: xgrid, "pi(d)": pis, "pi_threshold": pistars, "rebel": rebel_flags, "flip": flip_flags})

    # Plot π and π*
    fig1, ax1 = plt.subplots()
    ax1.plot(df[xname], df["pi(d)"], label="π(d) (ally shows up)")
    ax1.plot(df[xname], df["pi_threshold"], label="π* (deterrence threshold)")
    ax1.set_xlabel(xname)
    ax1.set_ylabel("Probability / Threshold")
    ax1.set_title(f"π(d) vs threshold across {xname}")
    ax1.legend()

    # Plot rebellion indicator
    fig2, ax2 = plt.subplots()
    ax2.plot(df[xname], df["rebel"], label="Rebellion occurs (1=yes)")
    ax2.set_xlabel(xname)
    ax2.set_ylabel("0/1")
    ax2.set_title(f"Predicted rebellion across {xname}")
    ax2.legend()

    return df, (fig1, fig2)

# ---------------------- UI ----------------------
st.title("🛡️ Rebel–Alliance Deterrence Model")
st.caption("A compact, deployable Streamlit app that walks through the three‑player game and lets you simulate how alliance design affects rebels' mobilization.")

tabs = st.tabs([
    "1) Overview",
    "2) Model & Assumptions",
    "3) Choose Alliance & Context",
    "4) Results",
    "5) Explore",
    "6) Export / Save"
])

# ---------------------- Tab 1: Overview ----------------------
with tabs[0]:
    st.subheader("Players, sequence, and intuition")
    st.markdown(
        """
        **Players.** Government (G), potential rebels (R), and an ally/alliance (A).  
        **Sequence.** The ally chooses an observable design **d** (obligations, institutionalization, provisions, partner traits). Rebels observe **d** and decide whether to mobilize. If they rebel, the ally decides whether to **intervene**; then G and R fight a contest.

        The alliance design affects three levers:
        1. **Resolve / obligations (F(d))** → how costly it is for the ally not to help;
        2. **Effectiveness (λ(d))** → how much allied intervention tilts the battlefield;
        3. **Constraints on G (c_G(d))** → provisions/conditionality that raise the state's cost of coercion.

        Rebels compare their expected payoff from mobilizing to their status‑quo payoff **S**. A high perceived intervention probability **π(d)** and a highly effective ally (**λ**) can deter rebellion; constraints that raise the government's costs can cut the other way.
        """
    )
    st.markdown("---")
    st.markdown("Use the tabs to **read the model**, **set parameters**, **see results**, and **explore scenarios**.")

# ---------------------- Tab 2: Model ----------------------
with tabs[1]:
    st.subheader("Key equations (peace, war, and deterrence)")
    st.latex(r"p_R(\tau)=\frac{\rho(d)}{\rho(d)+\tau},\quad \rho(d)=\sqrt{\frac{v_R\,c_G(d)}{v_G\,c_R}}")
    st.latex(r"W_R(\tau;d)=v_R\,\frac{\rho(d)\left(\rho(d)+\tfrac12\tau\right)}{(\rho(d)+\tau)^2}")
    st.latex(r"\Delta_A(\theta,d)=W_A\,[\,p_R(1)-p_R(\lambda(\theta,d))\,]+F(d)")
    st.latex(r"I^*(\theta,d)=\mathbbm{1}\{\Delta_A(\theta,d)\ge K_A(\theta)\},\quad \pi(d)=\mu I^*(H,d)+(1-\mu)I^*(L,d)")
    st.latex(r"\mathbb E[U_R\mid m{=}1,d]=(1-\pi)W_R(1;d)+\pi\,\overline W_R^{\,I}(d)-m_0(d)+g(d)")
    st.latex(r"\text{Rebels mobilize iff } \mathbb E[U_R\mid m{=}1,d]\ge S.")
    st.markdown(
        """
        When intervention **hurts** rebels on average (\\(W_R(1;d)>\overline W_R^{\,I}(d)\\)), the deterrence
        condition is a simple threshold:
        """
    )
    st.latex(r"\pi(d)\le \pi^*(d)\equiv\frac{W_R(1;d)-S-m_0(d)+g(d)}{W_R(1;d)-\overline W_R^{\,I}(d)}")
    st.markdown(
        """
        If intervention **helps** rebels on average (rare but possible under strong constraints), the inequality flips:
        rebels mobilize when \\( \pi(d)\ge \pi^\dagger(d) \\). The app detects and reports which case you're in.
        """
    )
    with st.expander("How design features map into primitives", expanded=False):
        st.markdown(
            """
            - **Legalization & institutionalization (L)** → raises **F(d)** and **λ(d)**; may lower \\(c_G(d)\\).
            - **Provisions/conditionality (P)** → raise \\(c_G(d)\\) (harder repression), can raise **F(d)**.
            - **Partner capability/proximity (C)** → raises **λ(d)**, lowers the ally's **K_A(θ)**.
            - **Partner democracy (D)** → raises **F(d)** (audience/legal costs) and can raise **K_A(θ)** (casualty aversion).
            """
        )
    st.info("All mapping coefficients are editable in the next tab (Advanced).")

# ---------------------- Tab 3: Inputs ----------------------
with tabs[2]:
    st.subheader("Step 1 — Choose alliance design and partner characteristics")
    col1, col2 = st.columns(2)
    with col1:
        L = st.slider("Legalization & institutionalization (L)", 0.0, 1.0, 0.6, 0.01)
        P = st.slider("Provisions / conditionality (P)", 0.0, 1.0, 0.3, 0.01)
        C = st.slider("Partner capability/proximity (C)", 0.0, 1.0, 0.7, 0.01)
        D = st.slider("Partner democracy (D)", 0.0, 1.0, 0.7, 0.01)

    with col2:
        st.markdown("#### Context & payoffs")
        mu = st.slider("Prior that the ally is high capability/resolve (μ)", 0.0, 1.0, 0.5, 0.01)
        W_A = st.slider("Ally's value alignment W_A", 0.0, 3.0, 1.0, 0.05)
        S = st.slider("Rebels' status‑quo payoff S", -1.0, 2.0, 0.2, 0.05)
        m0_base = st.slider("Baseline mobilization cost m₀ (before provisions)", 0.0, 2.0, 0.2, 0.01)
        g_shift = st.slider("Grievance shift g(d) (e.g., unpopular patron)", -1.0, 1.0, 0.0, 0.05)

    st.divider()
    st.subheader("Step 2 — Advanced (optional)")
    with st.expander("Game primitives", expanded=False):
        colA, colB, colC = st.columns(3)
        with colA:
            v_R = st.number_input("v_R (rebels' value from victory)", value=1.0, min_value=0.01)
            v_G = st.number_input("v_G (government's value from victory)", value=1.0, min_value=0.01)
        with colB:
            c_R = st.number_input("c_R (rebels' marginal effort cost)", value=1.0, min_value=0.001)
            cG0 = st.number_input("c_G0 (government baseline marginal effort cost)", value=1.0, min_value=0.001)
        with colC:
            st.markdown("—")

    with st.expander("Design → primitives mapping coefficients", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            F0 = st.number_input("F0 base reneging penalty", value=0.0)
            fL = st.number_input("fL (L → F)", value=1.0)
            fP = st.number_input("fP (P → F)", value=0.5)
            fD = st.number_input("fD (D → F)", value=1.0)
        with col2:
            l0 = st.number_input("l0 (base λ offset)", value=0.2, help="Ensures λ>1 even at minimal design")
            lL = st.number_input("lL (L → λ)", value=0.8)
            lC = st.number_input("lC (C → λ)", value=1.0)
            kappa_H = st.number_input("κ_H (ally H type multiplier)", value=1.2, min_value=1.0)
            kappa_L = st.number_input("κ_L (ally L type multiplier)", value=1.0, min_value=1.0)
        with col3:
            alphaP = st.number_input("α_P (P → c_G)", value=1.0)
            betaL = st.number_input("β_L (L → c_G)", value=0.2)
            mP = st.number_input("mP (P lowers m₀ by factor)", value=0.5, min_value=0.0, max_value=1.0)
        with col4:
            K_H0 = st.number_input("K_H0 (ally H baseline cost)", value=0.8, min_value=0.0)
            K_L0 = st.number_input("K_L0 (ally L baseline cost)", value=1.5, min_value=0.0)
            kL = st.number_input("kL (L lowers K_A)", value=0.6, min_value=0.0)
            kC = st.number_input("kC (C lowers K_A)", value=0.8, min_value=0.0)
            kD = st.number_input("kD (D raises K_A)", value=0.5, min_value=0.0)

    # Store in session state for downstream tabs
    st.session_state.inputs = dict(
        L=L, P=P, C=C, D=D, v_R=v_R, v_G=v_G, c_R=c_R, cG0=cG0,
        mu=mu, W_A=W_A, S=S, m0_base=m0_base, g_shift=g_shift,
        F0=F0, fL=fL, fP=fP, fD=fD,
        l0=l0, lL=lL, lC=lC, kappa_H=kappa_H, kappa_L=kappa_L,
        alphaP=alphaP, betaL=betaL, K_H0=K_H0, K_L0=K_L0, kL=kL, kC=kC, kD=kD,
        mP=mP
    )

# ---------------------- Tab 4: Results ----------------------
with tabs[3]:
    st.subheader("Results at your chosen design d")
    if "inputs" not in st.session_state:
        st.warning("Set parameters in the previous tab first.")
    else:
        args = st.session_state.inputs
        res = compute_all(**args)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("π(d) — intervention probability", f"{res['derived']['pi(d)']:.3f}")
            st.metric("λ_H", f"{res['derived']['lambda_H']:.3f}")
            st.metric("λ_L", f"{res['derived']['lambda_L']:.3f}")
            st.metric("F(d)", f"{res['derived']['F(d)']:.3f}")
        with col2:
            st.metric("ρ(d)", f"{res['derived']['rho(d)']:.3f}")
            st.metric("c_G(d)", f"{res['derived']['c_G(d)']:.3f}")
            st.metric("p_R (no intervention)", f"{res['derived']['pR_noI']:.3f}")
            st.metric("p_R (if intervene, H)", f"{res['derived']['pR_lambda_H']:.3f}")
        with col3:
            st.metric("Δ_A(H)", f"{res['derived']['Delta_H']:.3f}")
            st.metric("Δ_A(L)", f"{res['derived']['Delta_L']:.3f}")
            st.metric("I_H (1/0)", f"{res['derived']['I_H']}")
            st.metric("I_L (1/0)", f"{res['derived']['I_L']}")

        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            pi_star = res["threshold"]["pi_star"]
            if pi_star is None:
                st.info("π-threshold: undefined (intervention leaves W_R unchanged).")
            else:
                st.metric("π-threshold", f"{pi_star:.3f}")
            st.caption(res["threshold"]["note"])
        with colB:
            st.metric("EU if rebels mobilize", f"{res['derived']['EU_m1']:.3f}")
            st.metric("Status quo S", f"{args['S']:.3f}")
            st.success("Prediction: **REBEL**") if res["decision"]["rebel"] else st.info("Prediction: **NO REBELLION**")

        with st.expander("Show all computed values (JSON)"):
            st.code(json.dumps(res, indent=2))

# ---------------------- Tab 5: Explore ----------------------
with tabs[4]:
    st.subheader("Explore how design axes shift deterrence")
    if "inputs" not in st.session_state:
        st.warning("Set parameters in the inputs tab first.")
    else:
        args = st.session_state.inputs.copy()
        axis = st.selectbox("Sweep which design variable?", ["L (legalization)", "P (provisions)", "C (capability/proximity)", "D (democracy)"])
        npts = st.slider("Number of grid points", 10, 200, 60, 10)
        xgrid = np.linspace(0, 1, npts)
        if axis.startswith("L"):
            xname = "L"
        elif axis.startswith("P"):
            xname = "P"
        elif axis.startswith("C"):
            xname = "C"
        else:
            xname = "D"

        def compute_at_x(x):
            kwargs = args.copy()
            kwargs["L"] = args["L"] if xname != "L" else float(x)
            kwargs["P"] = args["P"] if xname != "P" else float(x)
            kwargs["C"] = args["C"] if xname != "C" else float(x)
            kwargs["D"] = args["D"] if xname != "D" else float(x)
            return compute_all(**kwargs)

        df, figs = plot_pi_vs_threshold(xgrid, xname, compute_at_x)
        st.dataframe(df.head(10))
        st.pyplot(figs[0])
        st.pyplot(figs[1])

# ---------------------- Tab 6: Export ----------------------
with tabs[5]:
    st.subheader("Export your current scenario")
    if "inputs" not in st.session_state:
        st.warning("Set parameters first.")
    else:
        args = st.session_state.inputs
        res = compute_all(**args)

        payload = {
            "inputs": args,
            "results": res
        }
        blob = json.dumps(payload, indent=2).encode("utf-8")
        st.download_button(
            "Download scenario JSON",
            data=blob,
            file_name="rebel_alliance_scenario.json",
            mime="application/json"
        )

    st.markdown("---")
    st.markdown(
        """
        **How to deploy**  
        1. Push `streamlit_app.py` and `requirements.txt` to a GitHub repo.  
        2. On [Streamlit Community Cloud](https://streamlit.io/cloud), create an app, point it to your repo and `streamlit_app.py`.  
        3. Click **Deploy**.
        """
    )
