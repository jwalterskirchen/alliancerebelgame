
# streamlit_app.py
# Rebel–Alliance Deterrence Model: v3.1 (fixes Scenario Lab indexing via product)
# License: MIT

import streamlit as st
import numpy as np
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
from itertools import product

st.set_page_config(
    page_title="Rebel–Alliance Deterrence Model",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Utility & math ----------------------
def rho(v_R, v_G, c_R, c_G):
    return math.sqrt(max(1e-12, (v_R * c_G) / (v_G * c_R)))

def pR(tau, rho_val):
    return rho_val / (rho_val + tau)

def WR(tau, rho_val, v_R):
    num = rho_val * (rho_val + 0.5 * tau)
    den = (rho_val + tau) ** 2
    return v_R * num / den

def lambda_base(L, C, l0, lL, lC):
    return max(1.01, 1.0 + l0 + lL * L + lC * C)

def lambda_type(lam_base, kappa):
    return max(1.01, lam_base * kappa)

def cG(cG0, P, L, alphaP, betaL):
    mult = 1.0 + alphaP * P - betaL * L
    return max(1e-6, cG0 * mult)

def F_design(F0, L, P, D, fL, fP, fD):
    return max(0.0, F0 + fL * L + fP * P + fD * D)

def K_cost(baseline, L, C, D, kL, kC, kD):
    return max(0.0, baseline - kL * L - kC * C + kD * D)

def m0_cost(m0_base, P, mP):
    return max(0.0, m0_base * (1.0 - mP * P))

def compute_all(
    # design axes
    L, P, C, D, secret,
    # context & baselines
    v_R, v_G, c_R, cG0,
    mu, W_A, S, m0_base, g_shift,
    # mapping coefficients
    F0, fL, fP, fD,
    l0, lL, lC, kappa_H, kappa_L,
    alphaP, betaL,
    K_H0, K_L0, kL, kC, kD,
    mP,
    secrecy_factor_F=0.25,   # how much secrecy reduces observable/operative F
    secrecy_blur=0.0         # how much secrecy dampens perceived π towards μ (0=no blur, 1=full blur)
):
    lam_base = lambda_base(L, C, l0, lL, lC)
    lam_H = lambda_type(lam_base, kappa_H)
    lam_L = lambda_type(lam_base, kappa_L)

    cG_val = cG(cG0, P, L, alphaP, betaL)
    rho_val = rho(v_R, v_G, c_R, cG_val)

    F_val_raw = F_design(F0, L, P, D, fL, fP, fD)
    F_val = F_val_raw * (secrecy_factor_F if secret else 1.0)

    K_H = K_cost(K_H0, L, C, D, kL, kC, kD)
    K_L = K_cost(K_L0, L, C, D, kL, kC, kD)

    # War-stage pieces
    W1 = WR(1.0, rho_val, v_R)
    WH = WR(lam_H, rho_val, v_R)
    WL = WR(lam_L, rho_val, v_R)

    pR_noI = pR(1.0, rho_val)
    pR_lamH = pR(lam_H, rho_val)
    pR_lamL = pR(lam_L, rho_val)

    Delta_H = W_A * (pR_noI - pR_lamH) + F_val
    Delta_L = W_A * (pR_noI - pR_lamL) + F_val

    I_H = 1 if Delta_H >= K_H else 0
    I_L = 1 if Delta_L >= K_L else 0

    pi_true = mu * I_H + (1 - mu) * I_L
    pi = (1 - secrecy_blur) * pi_true + secrecy_blur * mu  # secrecy makes beliefs revert toward μ

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
        WR_Ibar = mu * WH + (1 - mu) * WL

    m0 = m0_cost(m0_base, P, 0.5)
    EU_m1 = (1 - pi) * W1 + pi * WR_Ibar - m0 + g_shift

    denom = W1 - WR_Ibar
    numer = W1 - S - m0 + g_shift

    if abs(denom) < 1e-9:
        pi_star = None
        threshold_note = "Denominator ≈ 0; intervention leaves W_R unchanged."
        inequality = "—"
    elif denom > 0:
        pi_star = numer / denom
        threshold_note = "Intervention hurts rebels on average; rebel iff π ≤ π*."
        inequality = "≤"
    else:
        pi_star = numer / denom
        threshold_note = "Intervention helps rebels on average; rebel iff π ≥ π†."
        inequality = "≥"

    rebel = EU_m1 >= S

    return {
        "design": {"L": L, "P": P, "C": C, "D": D, "secret": bool(secret)},
        "derived": {
            "lambda_base": lam_base, "lambda_H": lam_H, "lambda_L": lam_L,
            "c_G(d)": cG_val, "rho(d)": rho_val,
            "F_raw": F_val_raw, "F_effective": F_val,
            "K_H": K_H, "K_L": K_L,
            "pR_noI": pR_noI, "pR_lambda_H": pR_lamH, "pR_lambda_L": pR_lamL,
            "Delta_H": Delta_H, "Delta_L": Delta_L,
            "I_H": I_H, "I_L": I_L,
            "pi_true": pi_true, "pi(d)": pi,
            "WR_noI": W1, "WR_lambda_H": WH, "WR_lambda_L": WL,
            "WR_Ibar": WR_Ibar,
            "EU_m1": EU_m1, "denom": denom, "numer": numer
        },
        "threshold": {"pi_star": pi_star, "note": threshold_note, "ineq": inequality},
        "context": {"mu": mu, "W_A": W_A, "S": S, "m0_base": m0_base, "g_shift": g_shift},
        "decision": {"rebel": bool(rebel)}
    }

# Simple line plot helper
def line_plot(x, ys, labels, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    for y, lab in zip(ys, labels):
        ax.plot(x, y, label=lab)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig

# Heatmap helper
def heatmap(Z, xvals, yvals, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    im = ax.imshow(Z, origin="lower", extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()], aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    return fig

# ---------------------- UI ----------------------
st.title("🛡️ Rebel–Alliance Deterrence Model (v3.1)")
st.caption("Adds 2D Explore and fixes Scenario Lab indexing (no more list index errors).")

tabs = st.tabs([
    "1) Overview",
    "2) Model",
    "3) Inputs",
    "4) Results",
    "5) Explore (1D)",
    "6) Explore (2D)",
    "7) Scenario Lab (multi)",
    "8) Export"
])

# ---------------------- Tab 1: Overview ----------------------
with tabs[0]:
    st.subheader("What’s new in v3.1")
    st.markdown(
        """
        - **Scenario Lab** fixed: uses `itertools.product` so 1–3 chosen axes work without indexing errors.  
        - 2D Explore still provides heatmaps for π(d), π*, and rebellion.
        """
    )

# ---------------------- Tab 2: Model ----------------------
with tabs[1]:
    st.latex(r"p_R(\tau)=\frac{\rho(d)}{\rho(d)+\tau},\quad \rho(d)=\sqrt{\frac{v_R\,c_G(d)}{v_G\,c_R}}")
    st.latex(r"W_R(\tau;d)=v_R\,\frac{\rho(d)\left(\rho(d)+\tfrac12\tau\right)}{(\rho(d)+\tau)^2}")
    st.latex(r"\Delta_A(\theta,d)=W_A\,[\,p_R(1)-p_R(\lambda(\theta,d))\,]+F(d)")
    st.latex(r"I^*(\theta,d)=\mathbbm{1}\{\Delta_A(\theta,d)\ge K_A(\theta)\},\quad \pi(d)=\mu I^*(H,d)+(1-\mu)I^*(L,d)")
    st.latex(r"\mathbb E[U_R\mid m{=}1,d]=(1-\pi)W_R(1;d)+\pi\,\overline W_R^{\,I}(d)-m_0(d)+g(d)")

# ---------------------- Tab 3: Inputs ----------------------
with tabs[2]:
    st.subheader("Alliance design & partner traits")
    c1, c2, c3 = st.columns(3)
    with c1:
        L = st.slider("Institutionalization / legalization (L)", 0.0, 1.0, 0.6, 0.01)
        P = st.slider("Provisions / conditionality (P)", 0.0, 1.0, 0.3, 0.01)
        C = st.slider("Power: capability / proximity (C)", 0.0, 1.0, 0.7, 0.01)
    with c2:
        D = st.slider("Democracy of partner (D)", 0.0, 1.0, 0.7, 0.01)
        reliability = st.slider("Reliability (affects F and K_A)", 0.0, 1.0, 0.7, 0.01)
        secret = st.checkbox("Secret / non-public alliance features", value=False)
    with c3:
        mu = st.slider("Prior: High-type probability μ", 0.0, 1.0, 0.5, 0.01)
        W_A = st.slider("Ally alignment W_A", 0.0, 3.0, 1.0, 0.05)
        S = st.slider("Rebels' status-quo S", -1.0, 2.0, 0.2, 0.05)
        m0_base = st.slider("Baseline mobilization cost m₀", 0.0, 2.0, 0.2, 0.01)
        g_shift = st.slider("Grievance shift g(d)", -1.0, 1.0, 0.0, 0.05)

    st.markdown("### Advanced coefficients")
    with st.expander("Mapping & costs", expanded=False):
        cA, cB, cC, cD = st.columns(4)
        with cA:
            v_R = st.number_input("v_R", value=1.0, min_value=0.01)
            v_G = st.number_input("v_G", value=1.0, min_value=0.01)
            c_R = st.number_input("c_R", value=1.0, min_value=0.001)
            cG0 = st.number_input("c_G0", value=1.0, min_value=0.001)
        with cB:
            F0 = st.number_input("F0", value=0.0)
            fL = st.number_input("fL (L→F)", value=1.0)
            fP = st.number_input("fP (P→F)", value=0.5)
            fD = st.number_input("fD (D→F)", value=1.0)
        with cC:
            l0 = st.number_input("l0 (base λ offset)", value=0.2)
            lL = st.number_input("lL (L→λ)", value=0.8)
            lC = st.number_input("lC (C→λ)", value=1.0)
            kappa_H = st.number_input("κ_H", value=1.2, min_value=1.0)
            kappa_L = st.number_input("κ_L", value=1.0, min_value=1.0)
        with cD:
            alphaP = st.number_input("α_P (P→c_G)", value=1.0)
            betaL = st.number_input("β_L (L→c_G)", value=0.2)
            K_H0 = st.number_input("K_H0", value=0.8, min_value=0.0)
            K_L0 = st.number_input("K_L0", value=1.5, min_value=0.0)
            kL = st.number_input("kL (L lowers K_A)", value=0.6, min_value=0.0)
            kC = st.number_input("kC (C lowers K_A)", value=0.8, min_value=0.0)
            kD = st.number_input("kD (D raises K_A)", value=0.5, min_value=0.0)

    with st.expander("Reliability & secrecy mechanics", expanded=False):
        secrecy_factor_F = st.slider("Secrecy factor on F (0=erase,1=no change)", 0.0, 1.0, 0.25, 0.05)
        secrecy_blur = st.slider("Secrecy belief blur toward μ", 0.0, 1.0, 0.3, 0.05)
        rel_F_boost = st.slider("Reliability boost to F", 0.0, 2.0, 1.0, 0.05)
        rel_K_cut = st.slider("Reliability cut to K_A", 0.0, 2.0, 0.5, 0.05)

    def compute_current(
        secret_flag=None,
        reliability_override=None,
        L_=None, P_=None, C_=None, D_=None,
        l0_override=None, lL_override=None, lC_override=None
    ):
        Lx = L if L_ is None else L_
        Px = P if P_ is None else P_
        Cx = C if C_ is None else C_
        Dx = D if D_ is None else D_
        sec = secret if secret_flag is None else secret_flag
        rel = reliability if reliability_override is None else reliability_override

        F0_eff = F0
        fL_eff = fL * (1.0 + rel * rel_F_boost)
        fP_eff = fP * (1.0 + rel * rel_F_boost)
        fD_eff = fD * (1.0 + rel * rel_F_boost)

        K_H0_eff = max(0.0, K_H0 * (1.0 - rel * rel_K_cut))
        K_L0_eff = max(0.0, K_L0 * (1.0 - rel * rel_K_cut))

        l0_use = l0 if l0_override is None else l0_override
        lL_use = lL if lL_override is None else lL_override
        lC_use = lC if lC_override is None else lC_override

        return compute_all(
            L=Lx, P=Px, C=Cx, D=Dx, secret=sec,
            v_R=v_R, v_G=v_G, c_R=c_R, cG0=cG0,
            mu=mu, W_A=W_A, S=S, m0_base=m0_base, g_shift=g_shift,
            F0=F0_eff, fL=fL_eff, fP=fP_eff, fD=fD_eff,
            l0=l0_use, lL=lL_use, lC=lC_use, kappa_H=kappa_H, kappa_L=kappa_L,
            alphaP=alphaP, betaL=betaL, K_H0=K_H0_eff, K_L0=K_L0_eff, kL=kL, kC=kC, kD=kD,
            mP=0.5,
            secrecy_factor_F=secrecy_factor_F, secrecy_blur=secrecy_blur
        )

    st.session_state.inputs = dict(
        L=L, P=P, C=C, D=D, secret=secret, reliability=reliability,
        mu=mu, W_A=W_A, S=S, m0_base=m0_base, g_shift=g_shift,
        v_R=v_R, v_G=v_G, c_R=c_R, cG0=cG0,
        F0=F0, fL=fL, fP=fP, fD=fD,
        l0=l0, lL=lL, lC=lC, kappa_H=kappa_H, kappa_L=kappa_L,
        alphaP=alphaP, betaL=betaL, K_H0=K_H0, K_L0=K_L0, kL=kL, kC=kC, kD=kD,
        secrecy_factor_F=secrecy_factor_F, secrecy_blur=secrecy_blur,
        rel_F_boost=rel_F_boost, rel_K_cut=rel_K_cut
    )
    st.session_state.compute_current = compute_current

# ---------------------- Tab 4: Results ----------------------
with tabs[3]:
    st.subheader("Point prediction at your baseline design")
    if "compute_current" not in st.session_state:
        st.warning("Set parameters in the Inputs tab first.")
    else:
        res = st.session_state.compute_current()
        d = res["derived"]; t = res["threshold"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("π(d) (beliefs)", f"{d['pi(d)']:.3f}")
            st.metric("π_true (objective)", f"{d['pi_true']:.3f}")
            st.metric("λ_H", f"{d['lambda_H']:.3f}")
            st.metric("λ_L", f"{d['lambda_L']:.3f}")
        with c2:
            st.metric("F_raw", f"{d['F_raw']:.3f}")
            st.metric("F_effective", f"{d['F_effective']:.3f}")
            st.metric("Δ_A(H)", f"{d['Delta_H']:.3f}")
            st.metric("Δ_A(L)", f"{d['Delta_L']:.3f}")
        with c3:
            st.metric("ρ(d)", f"{d['rho(d)']:.3f}")
            st.metric("p_R (no I)", f"{d['pR_noI']:.3f}")
            st.metric("p_R (H intervenes)", f"{d['pR_lambda_H']:.3f}")
            st.metric("p_R (L intervenes)", f"{d['pR_lambda_L']:.3f}")
        st.markdown("---")
        st.metric("π-threshold", f"{t['pi_star']:.3f}" if t['pi_star'] is not None else "—")
        st.caption(t["note"] + f" (Inequality: {t['ineq']})")
        st.success("Prediction: **REBEL**") if res["decision"]["rebel"] else st.info("Prediction: **NO REBELLION**")
        with st.expander("Raw JSON"):
            st.code(json.dumps(res, indent=2))

# ---------------------- Tab 5: Explore (1D) ----------------------
with tabs[4]:
    st.subheader("Sweep a single characteristic")
    if "compute_current" not in st.session_state:
        st.warning("Set parameters in the Inputs tab first.")
    else:
        compute_current = st.session_state.compute_current
        axis = st.selectbox("Sweep which?", ["Reliability", "Effectiveness (λ via l0)", "Power (C)", "Democracy (D)", "Institutionalization (L)", "Provisions (P)", "Secrecy toggle"])
        npts = st.slider("Grid points", 10, 200, 60, 10)
        xs = np.linspace(0, 1, npts)

        pis, pistars, rebels = [], [], []
        for v in xs:
            if axis == "Reliability":
                res = compute_current(reliability_override=float(v))
            elif axis == "Effectiveness (λ via l0)":
                res = compute_current(l0_override=st.session_state.inputs["l0"] + (float(v)-0.5)*0.8)
            elif axis == "Power (C)":
                res = compute_current(C_=float(v))
            elif axis == "Democracy (D)":
                res = compute_current(D_=float(v))
            elif axis == "Institutionalization (L)":
                res = compute_current(L_=float(v))
            elif axis == "Provisions (P)":
                res = compute_current(P_=float(v))
            else:
                res = compute_current(secret_flag=(v>=0.5))

            pis.append(res["derived"]["pi(d)"])
            pistars.append(np.nan if res["threshold"]["pi_star"] is None else res["threshold"]["pi_star"])
            rebels.append(1 if res["decision"]["rebel"] else 0)

        fig1 = line_plot(xs, [pis, pistars], ["π(d)", "π*"], f"{axis}: π vs threshold", axis, "Probability / threshold")
        fig2 = line_plot(xs, [rebels], ["Rebellion (1=yes)"], f"{axis}: rebellion prediction", axis, "0/1")
        st.pyplot(fig1)
        st.pyplot(fig2)

# ---------------------- Tab 6: Explore (2D) ----------------------
with tabs[5]:
    st.subheader("Heatmaps: vary two characteristics at once")
    if "compute_current" not in st.session_state:
        st.warning("Set parameters in the Inputs tab first.")
    else:
        compute_current = st.session_state.compute_current
        axes = ["Reliability", "Power (C)", "Democracy (D)", "Institutionalization (L)", "Provisions (P)", "Secrecy"]
        ax_x = st.selectbox("X-axis", axes, index=0)
        ax_y = st.selectbox("Y-axis", axes, index=3)
        if ax_x == ax_y:
            st.warning("Pick two different axes.")
        else:
            nx = st.slider("X resolution", 10, 100, 40, 5)
            ny = st.slider("Y resolution", 10, 100, 40, 5)
            xs = np.linspace(0, 1, nx)
            ys = np.linspace(0, 1, ny)

            Z_pi = np.zeros((ny, nx))
            Z_pistar = np.zeros((ny, nx))
            Z_rebel = np.zeros((ny, nx))

            def apply_axis(val, axis, kwargs):
                if axis == "Reliability":
                    kwargs["reliability_override"] = float(val)
                elif axis == "Power (C)":
                    kwargs["C_"] = float(val)
                elif axis == "Democracy (D)":
                    kwargs["D_"] = float(val)
                elif axis == "Institutionalization (L)":
                    kwargs["L_"] = float(val)
                elif axis == "Provisions (P)":
                    kwargs["P_"] = float(val)
                elif axis == "Secrecy":
                    kwargs["secret_flag"] = (val >= 0.5)
                return kwargs

            for iy, vy in enumerate(ys):
                for ix, vx in enumerate(xs):
                    kwargs = {}
                    kwargs = apply_axis(vx, ax_x, kwargs)
                    kwargs = apply_axis(vy, ax_y, kwargs)
                    res = compute_current(**kwargs)
                    Z_pi[iy, ix] = res["derived"]["pi(d)"]
                    Z_pistar[iy, ix] = np.nan if res["threshold"]["pi_star"] is None else res["threshold"]["pi_star"]
                    Z_rebel[iy, ix] = 1 if res["decision"]["rebel"] else 0

            fig_pi = heatmap(Z_pi, xs, ys, f"π(d) across {ax_x} (X) and {ax_y} (Y)", ax_x, ax_y)
            fig_pistar = heatmap(Z_pistar, xs, ys, f"π* across {ax_x} (X) and {ax_y} (Y)", ax_x, ax_y)
            fig_rebel = heatmap(Z_rebel, xs, ys, f"Rebellion (1=yes) across {ax_x} (X) and {ax_y} (Y)", ax_x, ax_y)
            st.pyplot(fig_pi); st.pyplot(fig_pistar); st.pyplot(fig_rebel)

# ---------------------- Tab 7: Scenario Lab (multi) ----------------------
with tabs[6]:
    st.subheader("Compare multiple characteristics at once (1–3 axes)")
    if "compute_current" not in st.session_state:
        st.warning("Set parameters in the Inputs tab first.")
    else:
        compute_current = st.session_state.compute_current
        st.markdown("Pick **one to three** axes to vary; others held at baseline.")

        axes_all = ["Reliability", "Power (C)", "Democracy (D)", "Institutionalization (L)", "Provisions (P)", "Secrecy"]
        chosen = st.multiselect("Axes", axes_all, default=["Reliability", "Institutionalization (L)"])
        chosen = chosen[:3]
        if len(chosen) == 0:
            st.info("Select at least one axis.")
        else:
            npts = st.slider("Grid points per chosen axis", 5, 30, 10, 1)
            grids = {ax: np.linspace(0, 1, npts) for ax in chosen}

            rows = []
            for combo in product(*[grids[ax] for ax in chosen]):
                kwargs = {}
                # fill kwargs per chosen axis
                for ax, val in zip(chosen, combo):
                    if ax == "Reliability":
                        kwargs["reliability_override"] = float(val)
                    elif ax == "Power (C)":
                        kwargs["C_"] = float(val)
                    elif ax == "Democracy (D)":
                        kwargs["D_"] = float(val)
                    elif ax == "Institutionalization (L)":
                        kwargs["L_"] = float(val)
                    elif ax == "Provisions (P)":
                        kwargs["P_"] = float(val)
                    elif ax == "Secrecy":
                        kwargs["secret_flag"] = (val >= 0.5)
                res = compute_current(**kwargs)

                # build a clean row with NaNs for unselected axes
                rec = {"Reliability": np.nan, "C": np.nan, "D": np.nan, "L": np.nan, "P": np.nan, "Secrecy": np.nan}
                for ax, val in zip(chosen, combo):
                    key = {"Reliability":"Reliability","Power (C)":"C","Democracy (D)":"D","Institutionalization (L)":"L","Provisions (P)":"P","Secrecy":"Secrecy"}[ax]
                    rec[key] = val
                rec.update({
                    "pi(d)": res["derived"]["pi(d)"],
                    "pi_true": res["derived"]["pi_true"],
                    "pi*": res["threshold"]["pi_star"] if res["threshold"]["pi_star"] is not None else np.nan,
                    "rebel": 1 if res["decision"]["rebel"] else 0
                })
                rows.append(rec)

            df = pd.DataFrame(rows)
            st.dataframe(df.head(20))

            # Summaries by chosen axes
            for metric in ["pi(d)", "pi*", "rebel"]:
                for ax in chosen:
                    colname = {"Reliability":"Reliability","Power (C)":"C","Democracy (D)":"D","Institutionalization (L)":"L","Provisions (P)":"P","Secrecy":"Secrecy"}[ax]
                    sub = df[[colname, metric]].dropna()
                    if len(sub) == 0:
                        continue
                    gb = sub.groupby(colname)[metric].mean().reset_index().sort_values(colname)
                    fig = line_plot(gb[colname].values, [gb[metric].values], [metric], f"{metric} vs {ax} (avg over others)", ax, metric)
                    st.pyplot(fig)

# ---------------------- Tab 8: Export ----------------------
with tabs[7]:
    st.subheader("Export baseline and settings")
    if "compute_current" not in st.session_state:
        st.warning("Set parameters first.")
    else:
        baseline = st.session_state.compute_current()
        payload = {"inputs": st.session_state.inputs, "baseline": baseline}
        blob = json.dumps(payload, indent=2).encode("utf-8")
        st.download_button("Download baseline JSON", data=blob, file_name="baseline_scenario.json", mime="application/json")
    st.markdown("---")
    st.markdown("To deploy, push this file and `requirements.txt` to GitHub and use Streamlit Community Cloud.")
