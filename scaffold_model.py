"""
Simplified bone-scaffold simulator
----------------------------------
Mechanics : Gibson-Ashby open-cell foam scaling laws
    E*/Es     = C1 * (rho_rel)^2      (C1 = 1.0)
    sigma*/ss = C2 * (rho_rel)^1.5    (C2 = 0.3)
    rho_rel   = 1 - porosity

Fluid flow: Kozeny-Carman permeability (approximation for porous media)
    k = d^2 * phi^3 / (K * (1 - phi)^2)     (K = 180)
    d = pore size (m), phi = porosity

NOTE: material values below are typical literature values. Verify and cite
them from your own references before putting them in the report.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Material properties (typical literature values) ----------------
MATERIALS = {
    #            Young's modulus Es (GPa), yield/compressive strength ss (MPa)
    "PCL":       {"E": 0.40,  "S": 20.0},
    "PLA":       {"E": 3.50,  "S": 60.0},
    "Ti-6Al-4V": {"E": 110.0, "S": 880.0},
}

# ---------------- Target window: cancellous (trabecular) bone ----------------
BONE_E_RANGE = (0.1, 2.0)      # GPa
BONE_S_MIN = 2.0               # MPa (lower end of cancellous compressive strength)
PORE_RANGE_UM = (100, 500)     # um, range usually reported for bone ingrowth
K_MIN = 1e-11                  # m^2, rough lower limit for adequate nutrient flow

C1, C2, KK = 1.0, 0.3, 180.0


def scaffold_properties(material, porosity, pore_um):
    m = MATERIALS[material]
    rho = 1.0 - porosity
    E = m["E"] * C1 * rho ** 2                      # GPa
    S = m["S"] * C2 * rho ** 1.5                    # MPa
    d = pore_um * 1e-6
    k = d ** 2 * porosity ** 3 / (KK * (1 - porosity) ** 2)   # m^2
    return E, S, k


def is_suitable(E, S, k, pore_um):
    return (BONE_E_RANGE[0] <= E <= BONE_E_RANGE[1]
            and S >= BONE_S_MIN
            and k >= K_MIN
            and PORE_RANGE_UM[0] <= pore_um <= PORE_RANGE_UM[1])


def build_grid():
    rows = []
    for mat in MATERIALS:
        for phi in np.arange(0.40, 0.951, 0.05):
            for pore in (100, 200, 300, 400, 500):
                E, S, k = scaffold_properties(mat, phi, pore)
                rows.append({"material": mat, "porosity_%": round(phi * 100),
                             "pore_um": pore, "E_GPa": E, "strength_MPa": S,
                             "perm_m2": k, "suitable": is_suitable(E, S, k, pore)})
    return pd.DataFrame(rows)


def best_designs(df):
    """Rank suitable designs WITHIN each material.
    score = 0.5 * norm(strength) + 0.5 * norm(log10 permeability)
    (min-max normalised inside that material's suitable set; log used because
    permeability spans several orders of magnitude)."""
    ok = df[df.suitable].copy()
    if ok.empty:
        return ok
    ok["log_k"] = np.log10(ok.perm_m2)
    def norm(x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0 + 1.0
    ok["score"] = ok.groupby("material")["strength_MPa"].transform(norm) * 0.5 + \
                  ok.groupby("material")["log_k"].transform(norm) * 0.5
    return ok.sort_values(["material", "score"], ascending=[True, False])


def make_figures(df, outdir="."):
    colors = {"PCL": "tab:green", "PLA": "tab:blue", "Ti-6Al-4V": "tab:red"}

    # 1. Strength vs porosity
    fig, ax = plt.subplots(figsize=(6, 4))
    phi = np.linspace(0.40, 0.95, 100)
    for mat in MATERIALS:
        S = [scaffold_properties(mat, p, 300)[1] for p in phi]
        ax.plot(phi * 100, S, label=mat, color=colors[mat])
    ax.axhspan(2, 12, color="gray", alpha=0.2, label="Cancellous bone (2-12 MPa)")
    ax.set_yscale("log"); ax.set_xlabel("Porosity (%)"); ax.set_ylabel("Compressive strength (MPa)")
    ax.set_title("Scaffold strength vs porosity"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig1_strength_vs_porosity.png", dpi=200); plt.close(fig)

    # 2. Modulus vs porosity
    fig, ax = plt.subplots(figsize=(6, 4))
    for mat in MATERIALS:
        E = [scaffold_properties(mat, p, 300)[0] for p in phi]
        ax.plot(phi * 100, E, label=mat, color=colors[mat])
    ax.axhspan(*BONE_E_RANGE, color="gray", alpha=0.2, label="Cancellous bone (0.1-2 GPa)")
    ax.set_yscale("log"); ax.set_xlabel("Porosity (%)"); ax.set_ylabel("Elastic modulus (GPa)")
    ax.set_title("Scaffold stiffness vs porosity"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig2_modulus_vs_porosity.png", dpi=200); plt.close(fig)

    # 3. Permeability vs pore size
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in (0.5, 0.7, 0.9):
        pores = np.linspace(100, 500, 50)
        k = [scaffold_properties("PLA", p, d)[2] for d in pores]
        ax.plot(pores, k, label=f"porosity {int(p*100)}%")
    ax.axhline(K_MIN, color="k", ls="--", lw=1, label="Min. permeability used")
    ax.set_yscale("log"); ax.set_xlabel("Pore size (um)"); ax.set_ylabel("Permeability (m$^2$)")
    ax.set_title("Permeability vs pore size"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig3_permeability_vs_pore.png", dpi=200); plt.close(fig)

    # 4. Trade-off: strength vs permeability for all designs, suitable highlighted
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for mat in MATERIALS:
        sub = df[df.material == mat]
        ax.scatter(sub.perm_m2, sub.strength_MPa, s=14, color=colors[mat], alpha=0.35, label=mat)
    ok = df[df.suitable]
    ax.scatter(ok.perm_m2, ok.strength_MPa, s=40, facecolors="none", edgecolors="k", label="Suitable for bone")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Permeability (m$^2$)"); ax.set_ylabel("Compressive strength (MPa)")
    ax.set_title("Strength vs permeability trade-off"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig4_tradeoff.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    import os
    out = os.path.dirname(os.path.abspath(__file__))
    df = build_grid()
    df.to_csv(f"{out}/scaffold_results_all.csv", index=False)
    best = best_designs(df)
    best.to_csv(f"{out}/scaffold_results_suitable_ranked.csv", index=False)
    make_figures(df, out)

    pd.set_option("display.width", 160)
    print("Total designs simulated:", len(df))
    print("Designs suitable for cancellous bone:", int(df.suitable.sum()))
    print("\nSuitable designs per material:")
    print(df[df.suitable].groupby("material").size())
    print("\nBest design per material (trade-off score):")
    top = best.groupby("material").head(1)
    top = top.assign(perm_m2=top.perm_m2.map(lambda v: f"{v:.2e}"))
    print(top.drop(columns=["suitable", "log_k"]).round(4).to_string(index=False))
    print("\nTop 5 per material:")
    for m, g in best.groupby("material"):
        g = g.head(5).assign(perm_m2=g.head(5).perm_m2.map(lambda v: f"{v:.2e}"))
        print("\n", m); print(g.drop(columns=["suitable", "log_k", "material"]).round(4).to_string(index=False))
