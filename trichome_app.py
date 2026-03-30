import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Trichome patterning model")

st.markdown(
    """
Diese App verwendet das 7-Gleichungs-Modell aus `help.pdf` mit den Variablen  
**TTG1, GL1, GL3, TRY, CPC, AC1, AC2**.  
Die Heatmaps zeigen Konzentrationen im Zellraster; die Trichomkarte ist ein Readout auf Basis hoher **AC2**-Werte.
"""
)

# ============================================================
# FESTE SIMULATIONSEINSTELLUNGEN
# ============================================================

GRID_SIZE = 50
N_STEPS = 400
DT = 0.001
NOISE = 0.01

# ============================================================
# PARAMETER AUS DEM PDF
# ============================================================

default_params = {
    "k1": 0.5982,   # TTG1 basal production
    "k2": 0.1405,   # TTG1 degradation
    "k3": 2.1971,   # TTG1-GL3 binding
    "k4": 1.1245,   # TTG1 diffusion
    "k5": 0.2916,   # GL1 basal production
    "k6": 2.3028,   # GL1 activation by AC2
    "k7": 0.3466,   # GL1 degradation
    "k8": 1.7822,   # GL1-GL3 binding
    "k9": 0.3976,   # GL3 basal production
    "k10": 9.9829,  # GL3 activation by AC1
    "k11": 1.2590,  # GL3 activation by AC2
    "k12": 2.6202,  # GL3 degradation
    "k13": 1.5731,  # TRY-GL3 binding
    "k14": 5.2625,  # CPC-GL3 binding
    "k15": 4.8758,  # TRY activation by AC1
    "k16": 0.3196,  # TRY degradation
    "k17": 0.1465,  # TRY diffusion
    "k18": 2.1453,  # CPC activation by AC2
    "k19": 0.5396,  # CPC degradation
    "k20": 56.0520, # CPC diffusion
    "k21": 0.5131,  # AC1 degradation
    "k22": 0.8396,  # AC2 degradation
    "k23": 7.8041,  # saturation GL3 activation by AC2
    "k24": 1.3647   # saturation GL3 activation by AC1
}

# ============================================================
# SIDEBAR / REGLER
# ============================================================

st.sidebar.header("Komponenten-Regler")

ttg1_factor = st.sidebar.slider("TTG1", 0.0, 3.0, 1.0, 0.1)
gl1_factor  = st.sidebar.slider("GL1",  0.0, 3.0, 1.0, 0.1)
gl3_factor  = st.sidebar.slider("GL3",  0.0, 3.0, 1.0, 0.1)
try_factor  = st.sidebar.slider("TRY",  0.0, 3.0, 1.0, 0.1)
cpc_factor  = st.sidebar.slider("CPC",  0.0, 3.0, 1.0, 0.1)

st.sidebar.header("Komplexbildung / Bindung")
k3_factor = st.sidebar.slider("AC1-Bildung (k3: TTG1-GL3)", 0.0, 3.0, 1.0, 0.1)
k8_factor = st.sidebar.slider("AC2-Bildung (k8: GL1-GL3)", 0.0, 3.0, 1.0, 0.1)

st.sidebar.header("Darstellung")
activator_view = st.sidebar.selectbox(
    "Aktivator-Heatmap",
    ["AC2", "AC1", "GL3", "GL1", "TTG1"]
)

inhibitor_view = st.sidebar.selectbox(
    "Inhibitor-Heatmap",
    ["TRY", "CPC", "TRY + CPC"]
)

threshold_percentile = st.sidebar.slider(
    "Trichom-Schwelle (AC2-Perzentil)",
    80, 99, 95, 1
)

st.sidebar.header("Presets")
preset = st.sidebar.selectbox(
    "Optionales Preset",
    ["Wildtyp", "ttg1-9"]
)

# ============================================================
# PARAMETER ANPASSEN
# ============================================================

params = default_params.copy()

# Presets
if preset == "ttg1-9":
    params["k3"] = 0.5  # laut PDF für ttg1-9  [oai_citation:3‡help.pdf](sediment://file_000000004d487243bd631047aa44afc8)

# Faktoren auf Produktionen / Bindungen anwenden
params["k1"] *= ttg1_factor
params["k5"] *= gl1_factor
params["k9"] *= gl3_factor
params["k15"] *= try_factor
params["k18"] *= cpc_factor

params["k3"] *= k3_factor
params["k8"] *= k8_factor

# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def laplace(X):
    return (
        -4 * X
        + np.roll(X, 1, axis=0)
        + np.roll(X, -1, axis=0)
        + np.roll(X, 1, axis=1)
        + np.roll(X, -1, axis=1)
    )

def pick_field(name, fields):
    if name == "TTG1":
        return fields["TTG1"]
    if name == "GL1":
        return fields["GL1"]
    if name == "GL3":
        return fields["GL3"]
    if name == "TRY":
        return fields["TRY"]
    if name == "CPC":
        return fields["CPC"]
    if name == "AC1":
        return fields["AC1"]
    if name == "AC2":
        return fields["AC2"]
    if name == "TRY + CPC":
        return fields["TRY"] + fields["CPC"]
    raise ValueError(f"Unknown field: {name}")

def radial_profile(field, center):
    """
    Mittelwert der Konzentration in Abhängigkeit vom Abstand zum Zentrum.
    """
    y, x = np.indices(field.shape)
    cy, cx = center
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_int = r.astype(int)

    max_r = r_int.max()
    profile = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)

    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            rr = r_int[i, j]
            profile[rr] += field[i, j]
            counts[rr] += 1

    counts[counts == 0] = 1
    profile /= counts
    return np.arange(max_r + 1), profile

# ============================================================
# INITIALISIERUNG
# ============================================================

rng = np.random.default_rng(42)

TTG1 = 0.1 + rng.random((GRID_SIZE, GRID_SIZE)) * NOISE
GL1  = 0.1 + rng.random((GRID_SIZE, GRID_SIZE)) * NOISE
GL3  = 0.1 + rng.random((GRID_SIZE, GRID_SIZE)) * NOISE
TRY  = 0.1 + rng.random((GRID_SIZE, GRID_SIZE)) * NOISE
CPC  = 0.1 + rng.random((GRID_SIZE, GRID_SIZE)) * NOISE
AC1  = 0.01 + rng.random((GRID_SIZE, GRID_SIZE)) * NOISE
AC2  = 0.01 + rng.random((GRID_SIZE, GRID_SIZE)) * NOISE

# kleine zentrale Störung zum Symmetriebruch
center_y, center_x = GRID_SIZE // 2, GRID_SIZE // 2
TTG1[center_y, center_x] += 0.05
GL1[center_y, center_x]  += 0.05
GL3[center_y, center_x]  += 0.05
AC1[center_y, center_x]  += 0.02
AC2[center_y, center_x]  += 0.02

# ============================================================
# SIMULATION: ALLE 7 GLEICHUNGEN
# ============================================================

for _ in range(N_STEPS):
    ac1sq = AC1**2
    ac2sq = AC2**2

    dTTG1 = (
        params["k1"]
        - TTG1 * (params["k2"] + params["k3"] * GL3)
        + params["k2"] * params["k4"] * laplace(TTG1)
    )

    dGL1 = (
        params["k5"]
        + params["k6"] * AC2
        - GL1 * (params["k7"] + params["k8"] * GL3)
    )

    dGL3 = (
        params["k9"]
        + (params["k24"] * params["k10"] * ac1sq) / (params["k24"] + ac1sq + 1e-12)
        + (params["k23"] * params["k11"] * ac2sq) / (params["k23"] + ac2sq + 1e-12)
        - GL3 * (
            params["k12"]
            + params["k3"] * TTG1
            + params["k8"] * GL1
            + params["k13"] * TRY
            + params["k14"] * CPC
        )
    )

    dTRY = (
        params["k15"] * ac1sq
        - TRY * (params["k16"] + params["k13"] * GL3)
        + params["k16"] * params["k17"] * laplace(TRY)
    )

    dCPC = (
        params["k18"] * ac2sq
        - CPC * (params["k19"] + params["k14"] * GL3)
        + params["k19"] * params["k20"] * laplace(CPC)
    )

    dAC1 = (
        params["k3"] * GL3 * TTG1
        - params["k21"] * AC1
    )

    dAC2 = (
        params["k8"] * GL3 * GL1
        - params["k22"] * AC2
    )

    TTG1 = np.maximum(0, TTG1 + DT * dTTG1)
    GL1  = np.maximum(0, GL1  + DT * dGL1)
    GL3  = np.maximum(0, GL3  + DT * dGL3)
    TRY  = np.maximum(0, TRY  + DT * dTRY)
    CPC  = np.maximum(0, CPC  + DT * dCPC)
    AC1  = np.maximum(0, AC1  + DT * dAC1)
    AC2  = np.maximum(0, AC2  + DT * dAC2)

fields = {
    "TTG1": TTG1,
    "GL1": GL1,
    "GL3": GL3,
    "TRY": TRY,
    "CPC": CPC,
    "AC1": AC1,
    "AC2": AC2
}

# ============================================================
# READOUTS
# ============================================================

act_field = pick_field(activator_view, fields)
inh_field = pick_field(inhibitor_view, fields)

# "Trichome" als hohe AC2-Werte
threshold = np.percentile(AC2, threshold_percentile)
trichome_map = AC2 >= threshold

# Zentrum = stärkstes AC2-Signal
peak_index = np.unravel_index(np.argmax(AC2), AC2.shape)
dist_act_x, dist_act_y = radial_profile(act_field, peak_index)
dist_inh_x, dist_inh_y = radial_profile(inh_field, peak_index)

# ============================================================
# AUSGABE
# ============================================================

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    im1 = ax1.imshow(act_field, cmap="inferno")
    ax1.set_title(f"Aktivator-Heatmap: {activator_view}")
    ax1.set_xticks([])
    ax1.set_yticks([])
    cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("relative Konzentration")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    im2 = ax2.imshow(inh_field, cmap="viridis")
    ax2.set_title(f"Inhibitor-Heatmap: {inhibitor_view}")
    ax2.set_xticks([])
    ax2.set_yticks([])
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("relative Konzentration")
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.imshow(trichome_map, cmap="gray_r")
    ax3.set_title("Trichom-Readout (hohe AC2-Werte)")
    ax3.set_xticks([])
    ax3.set_yticks([])
    st.pyplot(fig3)

with col4:
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ax4.plot(dist_act_x, dist_act_y, label=f"Aktivator: {activator_view}", linewidth=2)
    ax4.plot(dist_inh_x, dist_inh_y, label=f"Inhibitor: {inhibitor_view}", linewidth=2)
    ax4.set_title("Konzentration in Abhängigkeit von der Entfernung zum Trichom")
    ax4.set_xlabel("Entfernung vom AC2-Maximum")
    ax4.set_ylabel("relative Konzentration")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

# ============================================================
# ZUSATZINFO
# ============================================================

with st.expander("Aktuelle Parameter anzeigen"):
    st.json(params)

with st.expander("Biologische / modellhafte Einordnung"):
    st.markdown(
        """
- **AC1 = TTG1-GL3**
- **AC2 = GL1-GL3**
- **TRY** und **CPC** sind inhibitorische Komponenten
- Die Trichomkarte ist ein **Readout**, kein eigenes zusätzliches Modellgen
- Das Trichomzentrum wird hier als Position des höchsten **AC2**-Signals definiert
"""
    )
