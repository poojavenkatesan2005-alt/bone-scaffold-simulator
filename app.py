import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import scaffold_model as sm

st.set_page_config(page_title="Bone Scaffold Simulator", layout="wide")
st.title("Bone Scaffold Design Simulator")
st.caption("Gibson-Ashby mechanics + Kozeny-Carman permeability (analytical model)")

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Design parameters")
material = st.sidebar.selectbox("Material", list(sm.MATERIALS.keys()), index=1)
porosity = st.sidebar.slider("Porosity (%)", 40, 95, 60, 1) / 100
pore_um = st.sidebar.slider("Pore size (µm)", 100, 500, 300, 10)

E, S, k = sm.scaffold_properties(material, porosity, pore_um)
suitable = sm.is_suitable(E, S, k, pore_um)

# ---------------- Live results ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Elastic modulus", f"{E:.3f} GPa")
c2.metric("Compressive strength", f"{S:.2f} MPa")
c3.metric("Permeability", f"{k:.2e} m²")
c4.metric("Bone suitability", "Suitable" if suitable else "Not suitable")

if suitable:
    st.success("This design falls inside the cancellous bone window "
               "(0.1-2 GPa, ≥ 2 MPa, 100-500 µm pores, adequate permeability).")
else:
    reasons = []
    if not (sm.BONE_E_RANGE[0] <= E <= sm.BONE_E_RANGE[1]):
        reasons.append("modulus outside 0.1-2 GPa")
    if S < sm.BONE_S_MIN:
        reasons.append("strength below 2 MPa")
    if k < sm.K_MIN:
        reasons.append("permeability too low")
    st.warning("Not suitable: " + ", ".join(reasons))

# ---------------- Curves ----------------
phi = np.linspace(0.40, 0.95, 100)
colors = {"PCL": "tab:green", "PLA": "tab:blue", "Ti-6Al-4V": "tab:red"}

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in sm.MATERIALS:
        ax.plot(phi * 100, [sm.scaffold_properties(m, p, pore_um)[1] for p in phi],
                label=m, color=colors[m], lw=2 if m == material else 1)
    ax.axhspan(2, 12, color="gray", alpha=0.2, label="Cancellous bone")
    ax.scatter([porosity * 100], [S], color="k", zorder=5, label="Your design")
    ax.set_yscale("log")
    ax.set_xlabel("Porosity (%)")
    ax.set_ylabel("Strength (MPa)")
    ax.set_title("Strength vs porosity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    pores = np.linspace(100, 500, 50)
    for p in (0.5, 0.7, 0.9):
        ax.plot(pores, [sm.scaffold_properties(material, p, d)[2] for d in pores],
                label=f"porosity {int(p * 100)}%")
    ax.scatter([pore_um], [k], color="k", zorder=5, label="Your design")
    ax.axhline(sm.K_MIN, color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Pore size (µm)")
    ax.set_ylabel("Permeability (m²)")
    ax.set_title("Permeability vs pore size")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ---------------- Full study ----------------
st.header("Full design study")
df = sm.build_grid()
best = sm.best_designs(df)

st.write(f"Designs simulated: **{len(df)}**  |  "
         f"Suitable for bone: **{int(df.suitable.sum())}**")

top = best.groupby("material").head(1).copy()
top["perm_m2"] = top.perm_m2.map(lambda v: f"{v:.2e}")
st.subheader("Best design per material")
st.dataframe(top.drop(columns=["suitable", "log_k"]).round(4), width='stretch')

st.subheader("Strength vs permeability trade-off")
fig, ax = plt.subplots(figsize=(7, 4.5))
for m in sm.MATERIALS:
    sub = df[df.material == m]
    ax.scatter(sub.perm_m2, sub.strength_MPa, s=14, color=colors[m], alpha=0.35, label=m)
ok = df[df.suitable]
ax.scatter(ok.perm_m2, ok.strength_MPa, s=40, facecolors="none", edgecolors="k",
           label="Suitable for bone")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Permeability (m²)")
ax.set_ylabel("Strength (MPa)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
st.pyplot(fig)

st.download_button("Download all results (CSV)", df.to_csv(index=False),
                   file_name="scaffold_results_all.csv", mime="text/csv")

st.caption("Material values are typical literature values; this is a simplified "
           "analytical model, not a finite element simulation.")
