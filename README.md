# Alliance–Rebel Deterrence Simulator (Streamlit)

A small app with sliders to explore how alliance *credibility*, *effectiveness*,
*capacity-building*, and *conditionality*, together with a government's
**repression** and **concessions**, shape the rebel attack decision.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser. Adjust sliders, select alliance types,
and click **Run simulation**.

## Files

- `app.py` – the Streamlit UI
- `model.py` – the core model and simulation functions
- `requirements.txt` – Python dependencies
- `README.md` – this file

# The Game

# Government–Rebel–Alliance Model

## 1) Players, actions, timeline

**Players:** Government $G$, potential Rebels $R$, Alliance $A$.

**Alliance types:** $t \in T$ (e.g., none, consultative/entente, defense pact, defense pact + forward basing, etc.).  
Each type $t$ maps to primitives defined below.

**Timeline (perfect Bayesian equilibrium):**
1. **Government chooses alliance type.**  
   $G$ chooses $t \in T$ and pays a (sovereignty, budget, diplomatic) cost $k_G(t)$.  
   Assume the alliance partner accepts any $t$ that $G$ would feasibly sign (you can endogenize this later).
2. **Observation.**  
   $R$ observes $t$.
3. **Rebel move.**  
   $R$ chooses $a \in \{\text{Attack}, \text{Quiet}\}$.
4. **Alliance move if attacked.**  
   If $a=\text{Attack}$, $A$ chooses $i \in \{\text{Intervene}, \text{Stay Out}\}$ given $t$ and its private cost draw $c$.
5. **Resolution.**  
   If conflict occurs, $R$ wins with probability $q_i(t)$ and loses with probability $1-q_i(t)$, where $i \in \{I,U\}$ (intervene, stay out).

---

## 2) What “alliance type” does (four channels)

Alliance type $t$ is summarized by four monotone objects:

- **Credibility:** $\pi(t) \equiv \Pr(i=I \mid \text{attack}, t)$, the *ex ante* probability $A$ will intervene if fighting starts.
- **Intervention effectiveness:** $d(t) \equiv q_U(t) - q_I(t) \in [0,1]$, the reduction in rebels’ win probability if $A$ actually intervenes.
- **Capacity building (without intervention):** $q_U(t)$, rebels’ win probability *if* $A$ stays out (training/equipment to $G$ can reduce it).
- **Conditionality / constraints:** $s(t)$, the rebels’ *status‑quo* payoff if they remain quiet (conditional alliances can raise it; moral‑hazard alliances might lower it).

**Types as bundles:**
- None: $\pi \approx 0,\ d\approx 0,\ q_U=q_0,\ s=s_0$.
- Consultative/entente: small $\pi,d$; minimal $q_U$ shift; small $+s$.
- Defense pact (no deployment): medium $\pi,d$; small $q_U$ shift; $s$ neutral.
- Conditional defense (reform‑linkage): medium/high $\pi,d$; $q_U$ down; **$s$ up**.

---

## 3) Payoffs

Normalize $R$’s payoffs:

- If **Quiet**: $u_R(\text{Quiet}\mid t) = s(t)$.
- If **Attack** and **win**: $v_R$.
- If **Attack** and **lose**: $\ell_R$.
- Fixed cost of fighting: $k_R>0$.

Let $\Delta_R \equiv v_R - \ell_R > 0$. Given win probability $q$, the war payoff is
$$
u_R(\text{war at } q) \;=\; \ell_R - k_R + q\,\Delta_R.
$$

With alliance type $t$:
- If $A$ stays out: win prob $q_U(t)$.
- If $A$ intervenes: win prob $q_I(t) = q_U(t) - d(t)$.

Rebels’ **ex‑ante** win probability (integrating over $A$’s intervention) is
$$
q^*(t) \;\equiv\; \mathbb{E}[q \mid t]
\;=\; (1-\pi(t))\,q_U(t) + \pi(t)\,q_I(t)
\;=\; q_U(t) - \pi(t)\,d(t).
$$

---

## 4) Rebel decision and central threshold

$R$ attacks iff its expected war payoff exceeds the status quo:
$$
\ell_R - k_R + q^*(t)\,\Delta_R \;\ge\; s(t).
$$

Define the “required” win probability threshold
$$
\hat{q}(t) \;\equiv\; \frac{s(t) - \ell_R + k_R}{\Delta_R}.
$$

Then **attack** occurs iff
$$
q^*(t) \;=\; q_U(t) - \pi(t)\,d(t) \;\ge\; \hat{q}(t).
$$

**Deterrence condition (equivalent):**
$$
\boxed{\;\pi(t) \;\ge\; \frac{q_U(t) - \hat{q}(t)}{d(t)}\;}
$$

**Interpretation.** Alliances deter via: (1) raising $\pi(t)$; (2) raising $d(t)$; (3) lowering $q_U(t)$; (4) raising $s(t)$ (which raises $\hat{q}$).

---

## 5) Microfounding $\pi(t)$ (credibility as costly commitment)

Let the alliance’s variable intervention cost $c$ be private, with CDF $F$ on $[0,\bar c]$.  
Signing type $t$ creates a reputational/legal cost $r(t)$ for **not** intervening (increasing in treaty strength) and a strategic benefit $b$ to preventing a rebel victory.

Upon attack, $A$ intervenes iff $c \le r(t)+b$. Thus
$$
\pi(t) \;=\; F\!\big(r(t)+b\big), \qquad r'(t) > 0.
$$
Stronger treaties raise $r(t)$ and thus increase $\pi(t)$ *endogenously*, delivering credible commitments.

---

## 6) Government’s choice of alliance type

Let $L_G^I$ and $L_G^U$ be $G$’s expected losses if attacked with/without intervention (these embed $q_I,q_U$). The government’s expected loss under $t$ is
$$
\mathrm{EL}_G(t)
= k_G(t) \;+\; \Pr(\text{attack}\mid t)\,\big[\,\pi(t)L_G^I + (1-\pi(t))L_G^U\,\big].
$$

The optimal $t$ trades off **treaty costs** $k_G(t)$ against lowering $\Pr(\text{attack}\mid t)$ via the deterrence condition.  
If $T$ is totally ordered so that $t$ increases $\pi$ and $d$ and (weakly) lowers $q_U$ or raises $s$, then a **cutoff type** $t^\star$ exists: all $t \ge t^\star$ deter; $t < t^\star$ do not.

---

## 7) Propositions (sketches)

**Proposition 1 (Deterrence).** If $\pi(t)\,d(t) \ge q_U(t)-\hat{q}(t)$, then in any PBE rebels do not attack upon observing $t$.  
*Sketch.* Plug $q^*(t)=q_U-\pi d$ into the attack condition and reverse the inequality.

**Proposition 2 (Monotone comparative statics).** Holding other channels fixed, increasing any of $\pi(t),d(t),s(t)$ or decreasing $q_U(t)$ (pointwise) weakly lowers the set of beliefs/payoffs under which rebels attack.

**Proposition 3 (Signaling by treaty strength).** If reneging costs $r(t)$ are high enough for strong types that low‑resolve alliances expect negative payoffs from signing them, only high‑resolve alliances sign strong $t$. Then $t$ is a credible signal (higher $r \Rightarrow$ higher $\pi$), reducing attacks in equilibrium.

---

## 8) Mapping to alliance “types”

| Type $t$ | $\pi(t)$ | $d(t)$ | $q_U(t)$ | $s(t)$ | Typical intuition |
|---|---:|---:|---:|---:|---|
| None | $\sim 0$ | $0$ | $q_0$ | $s_0$ | No promise, no help |
| Entente/consultative | Low | Low | $q_0$ | $s_0{+}$ | Talk, small tripwire |
| Defense pact | Med | Med | $q_0{-}$ | $s_0$ | Commitments, some aid |
| Pact + forward basing | High | High | $q_0{-}{-}$ | $s_0$ | Rapid lift \& C2 |
| Conditional defense | Med/High | Med/High | $q_0{-}$ | $s_0{+}{+}$ | Reforms/aid constraints |

- **Capacity channel:** $q_U(t)\downarrow$ even if $A$ stays out (training/equipment).  
- **Intervention channel:** $d(t)\uparrow$ with deployment/integration.  
- **Credibility channel:** $\pi(t)\uparrow$ via legal/reneging costs $r(t)$.  
- **Constraint channel:** $s(t)\uparrow$ if alliance conditions improve governance.

---

## 9) Simple numeric illustration (calibration guide)

Pick $(v_R,\ell_R,k_R)=(1,0,0.1) \Rightarrow \Delta_R=1,\ \hat{q}(t)=s(t)+0.1$. Let $q_0=0.45$.

- **None:** $q_U=0.45,\ \pi=0,\ d=0,\ s=0 \Rightarrow q^*=0.45>\hat{q}=0.1$ → Attack.
- **Defense pact:** $q_U=0.35,\ \pi=0.6,\ d=0.2,\ s=0 \Rightarrow q^*=0.35-0.6\cdot0.2=0.23>\hat{q}=0.1$ → Attack (weaker deterrence).
- **Forward basing:** $q_U=0.30,\ \pi=0.8,\ d=0.25,\ s=0 \Rightarrow q^*=0.30-0.8\cdot0.25=0.10=\hat{q}$ → Indifferent (knife‑edge deterrence).
- **Conditional defense:** same as basing but $s=0.05 \Rightarrow \hat{q}=0.15$; now $q^*=0.10<0.15$ → **No attack**.

---

## 10) Useful extensions

1. **Endogenous repression or concessions by $G$.**  
   $G$ chooses $x$ (repression) or $y$ (concessions) after picking $t$, shifting $q_U(t)$ and $s(t)$. Strong alliances may create moral hazard (lower $s$, raise attack risk) unless constrained—letting you test “alliance type × constraint” interactions.
2. **Alliance’s objective and selection.**  
   Let $A$ pick $t$ first (offer set), anticipating $G$’s acceptance and $R$’s response; optimize $-\mathbb{E}[\text{intervention cost}] + \omega \cdot \mathbb{E}[\text{stability}]$.
3. **Private information about rebel strength.**  
   Nature draws $q_0$. Alliance type $t$ becomes a screening device; you can get pooling/separating in $R$’s attack decision by strength.
4. **Third‑party bias or conditional intervention.**  
   Let $A$’s $b$ depend on regime type or abuses; then $\pi(t)$ becomes state‑contingent, capturing alliances that punish government behavior.

---

## Endogenous domestic policy: repression & concessions

Let
$$
q_U(t,r,y)=q_U^0(t)-\phi_r(t)\,r-\phi_y(t)\,y,
\qquad
s(t,r,y)=s^0(t)-\psi_r(t)\,r+\psi_y(t)\,y,
$$
with costs $C_G(r,y;t)=\tfrac12 c_r(t)r^2+\tfrac12 c_y(t)y^2$.

Define $A_r\equiv \phi_r-\psi_r/\Delta_R$, $A_y\equiv \phi_y+\psi_y/\Delta_R$, and
$$
D(r,y;t)
= \big(q_U^0(t)-\pi(t)d(t)\big)
- \frac{s^0(t)-\ell_R+k_R}{\Delta_R}
- A_r r - A_y y.
$$

- If $D_0\equiv D(0,0;t)\le 0$: no attack at $(r,y)=(0,0)$.
- If $D_0>0$ and at least one of $A_r,A_y>0$, the **minimal‑cost deterring policy** solves
$$
\min_{r,y\ge 0}\ \tfrac12(c_r r^2+c_y y^2)
\quad\text{s.t.}\quad
A_r r + A_y y \ge D_0,
$$
with solution
$$
r^*=\lambda \frac{A_r}{c_r},\quad
y^*=\lambda \frac{A_y}{c_y},\quad
\lambda=\frac{D_0}{A_r^2/c_r + A_y^2/c_y},\quad
C^*=\tfrac12\,\frac{D_0^2}{A_r^2/c_r + A_y^2/c_y}.
$$

$G$ implements $(r^*,y^*)$ iff $C^*\le L(t)\equiv \pi(t)L_G^I+(1-\pi(t))L_G^U$.
