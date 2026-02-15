# lda_design_gui.py

# ============================================================
# TÜBİTAK UME – NanoLDA Optical Design & Simulation
#
# Kısa Açıklama:
# Bu Streamlit uygulaması, NanoLDA sisteminin optik hattını hızlı tasarlamak ve
# doğrulamak için geliştirilmiştir. Gaussian beam odak metrikleri (d_f, δ, N_f),
# güç bütçesi (fiber kuplaj → BS → AOM → fokus lens) ve hizalama hatalarının
# (dx, dy, beam miss) kuplaj/kontrast üzerindeki etkilerini gerçek zamanlı olarak
# simüle eder. “Detected (proxy)” değeri trend amaçlıdır; gerçek Rx toplama/saçılma
# modeli eklendiğinde fiziksel dedektör gücüne evrilecektir.
# ============================================================


import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from optical_components import (
    LaserSource, Fiber, Lens, BeamState,
    q_from_waist, w_from_q, propagate_free_space, thin_lens,
    fiber_coupling_eta, coupling_loss_db,
    fringe_spacing, focus_waist_from_beam_diameter, fringe_count,
    contrast_from_miss_overlap, PowerStage
)
import thorlabs_catalog as cat



# ------------------------------------------------------------
# ✅ Streamlit-safe callback (widget key'leri için tek doğru yol)
# ------------------------------------------------------------
def apply_autotune(theta_mrad: float, D_mm: float):
    st.session_state["crossing_angle_mrad"] = float(theta_mrad)
    st.session_state["D_e2_mm"] = float(D_mm)

# ------------------------------------------------------------
# PAGE
# ------------------------------------------------------------
st.set_page_config(page_title="NanoLDA Optics Design", layout="wide")
st.title("TÜBİTAK UME NanoLDA Optical Design & Simulation (Ön Tasarım Çalışması)")

# ------------------------------------------------------------
# 1) SIDEBAR INPUTS (ÖNCE)
# ------------------------------------------------------------
st.sidebar.header("Source")
lam_nm = st.sidebar.number_input("Wavelength (nm)", 375.0, 2000.0, 532.0, 1.0, key="lam_nm")
p_mw = st.sidebar.number_input("Laser Power (mW)", 0.1, 5000.0, 50.0, 1.0, key="p_mw")
m2 = st.sidebar.slider("M²", 1.0, 3.0, 1.1, 0.05, key="m2")

st.sidebar.header("Fiber / Collimator")
fiber_choice = st.sidebar.selectbox("Fiber", [f["name"] for f in cat.FIBERS], key="fiber_choice")
fib = next(f for f in cat.FIBERS if f["name"] == fiber_choice)
eta0 = st.sidebar.slider("Baseline coupling η0", 0.3, 0.95, float(fib["eta0"]), 0.01, key="eta0")

dx_um = st.sidebar.slider("Fiber X-offset (µm)", -20.0, 20.0, 0.0, 0.5, key="dx_um")
dy_um = st.sidebar.slider("Fiber Y-offset (µm)", -20.0, 20.0, 0.0, 0.5, key="dy_um")

st.sidebar.header("Transmission chain")
bs_choice = st.sidebar.selectbox("Beam Splitter", [b["name"] for b in cat.BEAMSPLITTERS], key="bs_choice")
bs = next(b for b in cat.BEAMSPLITTERS if b["name"] == bs_choice)

aom_choice = st.sidebar.selectbox("AOM", [a["name"] for a in cat.AOMS], key="aom_choice")
aom = next(a for a in cat.AOMS if a["name"] == aom_choice)

st.sidebar.header("Probe / Focusing")
lens_choice = st.sidebar.selectbox("Focusing Lens", [L["name"] for L in cat.LENSES], key="lens_choice")
L = next(L for L in cat.LENSES if L["name"] == lens_choice)

beam_sep_mm = st.sidebar.number_input("Beam separation at lens (mm)", 1.0, 200.0, 40.0, 1.0, key="beam_sep_mm")

# ✅ key'ler burada kritik: Auto-Tune butonu bunları değiştirecek
crossing_angle_mrad = st.sidebar.number_input(
    "Crossing angle θ (mrad)", 0.1, 200.0, 20.0, 0.5, key="crossing_angle_mrad"
)
D_e2_mm = st.sidebar.number_input(
    "Beam diameter at lens (e^-2) (mm)", 0.1, 20.0, 2.0, 0.1, key="D_e2_mm"
)

st.sidebar.header("Misalignment in probe")
miss_um = st.sidebar.slider("Beam miss at crossing (µm)", 0.0, 200.0, 0.0, 1.0, key="miss_um")

# ------------------------------------------------------------
# 2) BUILD OBJECTS + PHYSICS (SONRA)
# ------------------------------------------------------------
lam = lam_nm * 1e-9
P0 = p_mw * 1e-3

source = LaserSource(wavelength_m=lam, power_w=P0, m2=m2)
fiber = Fiber(na=float(fib["na"]), core_diam_m=float(fib["core_um"]) * 1e-6, eta0=eta0)
focus_lens = Lens(f_m=float(L["f_mm"]) * 1e-3, transmission=float(L["T"]), name=L["name"])

# Probe metrics
D_e2 = D_e2_mm * 1e-3
w_focus = focus_waist_from_beam_diameter(focus_lens.f_m, lam, D_e2)  # waist radius
d_focus = 2.0 * w_focus

theta = crossing_angle_mrad * 1e-3  # rad
delta = fringe_spacing(lam, theta)
Nf = fringe_count(d_focus, delta)

# Power budget chain
stages = []

# Fiber coupling: mode radius guess (MVP)
w_mode = max(0.5 * fiber.core_diam_m, 2.0e-6)
eta_coup = fiber_coupling_eta(fiber, w_mode, dx_um * 1e-6, dy_um * 1e-6)

P_after_coup = source.power_w * eta_coup
stages.append(PowerStage("After fiber coupling", P_after_coup, f"η_coup={eta_coup:.3f}"))

# Beam splitter
P_tx = P_after_coup * (1.0 - bs["abs"]) * bs["split"]
P_ref = P_after_coup * (1.0 - bs["abs"]) * (1.0 - bs["split"])
stages.append(PowerStage("After BS (arm A)", P_tx, f"split={bs['split']:.2f}, abs={bs['abs']:.2f}"))
stages.append(PowerStage("After BS (arm B)", P_ref, f"split={1 - bs['split']:.2f}, abs={bs['abs']:.2f}"))

# AOM in one arm
P_aom = P_tx * aom["eta_1st"]
stages.append(PowerStage("After AOM 1st order (arm A)", P_aom, f"η_AOM={aom['eta_1st']:.2f}"))

# Focus lens transmission
P_aom_f = P_aom * focus_lens.transmission
P_ref_f = P_ref * focus_lens.transmission
stages.append(PowerStage("After focus lens (arm A)", P_aom_f, f"T_lens={focus_lens.transmission:.3f}"))
stages.append(PowerStage("After focus lens (arm B)", P_ref_f, f"T_lens={focus_lens.transmission:.3f}"))

P_probe_total = P_aom_f + P_ref_f

# Contrast from beam miss
contrast = contrast_from_miss_overlap(miss_um * 1e-6, w_focus)

# Detected proxy
P_det_proxy = P_probe_total * 0.001 * contrast
stages.append(PowerStage("Detected (proxy)", P_det_proxy, f"contrast={contrast:.3f} (proxy model)"))

# ------------------------------------------------------------
# 3) EXPLAINERS
# ------------------------------------------------------------
with st.expander("🧠 Sistem nasıl çalışıyor? (Model + Varsayımlar)", expanded=False):
    st.markdown(r"""
### Bu uygulama neyi simüle ediyor?
Bu Streamlit aracı, bir LDA optik hattının **çekirdek tasarım metriklerini** ve **güç bütçesini** hızlıca görmeniz için hazırlanmış bir Ön Tasarım Çalışmasıdır.

Şu an 3 şeyi yapar:
1) **Probe metrikleri:** odak çapı, fringe aralığı, fringe sayısı  
2) **Power budget:** fiber kuplaj → BS → AOM → lens → (detected proxy)  
3) **Misalignment etkisi:** fiber ofseti gücü düşürür; probe beam miss kontrastı düşürür.

> “Detected (proxy)” **gerçek dedektör gücü değildir**. Trend göstergesidir.
""")

with st.expander("📌 Sonuçları nasıl yorumlayacağız?", expanded=False):
    st.markdown(r"""
- **d_f küçük** → ölçüm hacmi küçük (çözünürlük ↑) ama tolerans zor.
- **δ küçük** → fringe sık; doppler frekansı aynı hızda ↑.
- **After fiber coupling** düşükse ana sorun hizadır.
- **Contrast** düşükse “güç var ama sinyal yok” olabilir.
""")

# ------------------------------------------------------------
# 4) CHECK-LIST + AUTO-TUNE
# ------------------------------------------------------------
st.subheader("✅ Tasarım Hedefleri Check-list (Auto PASS/FAIL)")

with st.expander("Hedef eşikleri (istersen değiştir)", expanded=False):
    c1, c2, c3 = st.columns(3)

    with c1:
        df_min_um = st.number_input("d_f min (µm)", 1.0, 1000.0, 50.0, 1.0, key="df_min_um")
        df_max_um = st.number_input("d_f max (µm)", 1.0, 2000.0, 100.0, 1.0, key="df_max_um")

    with c2:
        contrast_min = st.number_input("Contrast min", 0.0, 1.0, 0.70, 0.01, key="contrast_min")
        coup_min = st.number_input("Coupling η min", 0.0, 1.0, 0.60, 0.01, key="coup_min")

    with c3:
        delta_min_um = st.number_input("δ min (µm)", 0.01, 1000.0, 2.0, 0.1, key="delta_min_um")
        delta_max_um = st.number_input("δ max (µm)", 0.01, 2000.0, 5.0, 0.1, key="delta_max_um")

df_um = d_focus * 1e6
delta_um = delta * 1e6
P_probe_mw = P_probe_total * 1e3

checks = []

def add_check(label: str, passed: bool, value_str: str, target_str: str, hint_pass: str = "", hint_fail: str = ""):
    checks.append({
        "Kriter": label,
        "Durum": "✅ PASS" if passed else "❌ FAIL",
        "Değer": value_str,
        "Hedef": target_str,
        "Aksiyon": hint_pass if passed else hint_fail
    })

add_check(
    "Odak çapı d_f",
    (df_min_um <= df_um <= df_max_um),
    f"{df_um:.1f} µm",
    f"{df_min_um:.1f}–{df_max_um:.1f} µm",
    hint_pass="Odak boyutu hedef aralıkta.",
    hint_fail="d_f küçükse: D ↓ veya f ↑. d_f büyükse: D ↑ veya f ↓."
)

add_check(
    "Fringe spacing δ",
    (delta_min_um <= delta_um <= delta_max_um),
    f"{delta_um:.2f} µm",
    f"{delta_min_um:.2f}–{delta_max_um:.2f} µm",
    hint_pass="Fringe aralığı hedef aralıkta.",
    hint_fail="δ büyükse: θ ↑. δ küçükse: θ ↓."
)

add_check(
    "Fringe contrast",
    (contrast >= contrast_min),
    f"{contrast:.3f}",
    f">= {contrast_min:.2f}",
    hint_pass="Overlap iyi; modulation güçlü.",
    hint_fail="Contrast düşük: beam miss ↓, mekanik hizayı iyileştir."
)

add_check(
    "Fiber coupling η",
    (eta_coup >= coup_min),
    f"{eta_coup:.3f}",
    f">= {coup_min:.2f}",
    hint_pass="Kuplaj yeterli.",
    hint_fail="Kuplaj düşük: dx/dy ↓, hizayı iyileştir."
)

probe_min_mw = st.number_input("Probe toplam gücü min (mW) (opsiyonel)", 0.0, 1e6, 5.0, 0.5, key="probe_min_mw")
add_check(
    "Probe toplam gücü (A+B)",
    (P_probe_mw >= probe_min_mw),
    f"{P_probe_mw:.2f} mW",
    f">= {probe_min_mw:.2f} mW",
    hint_pass="Probe gücü yeterli (zincir açısından).",
    hint_fail="Probe gücü düşük: coupling ↑, BS/AOM/lens kayıplarını azalt."
)

pass_count = sum(1 for x in checks if x["Durum"].startswith("✅"))
total = len(checks)

if pass_count == total:
    st.success(f"GENEL DURUM: ✅ HEDEFLER SAĞLANDI ({pass_count}/{total})")
elif pass_count >= total - 1:
    st.warning(f"GENEL DURUM: ⚠️ SINIRDA ({pass_count}/{total}) — 1 kriter problemli.")
else:
    st.error(f"GENEL DURUM: ❌ HEDEFLER SAĞLANMADI ({pass_count}/{total}) — tasarım revizyonu gerekli.")

st.dataframe(checks, use_container_width=True)

# -----------------------------
# AUTO-TUNE (FAIL → öner + apply)  ✅ FIXED
# -----------------------------
any_fail = any(x["Durum"].startswith("❌") for x in checks)

if any_fail:
    with st.expander("🔧 Auto-Tune Önerileri (FAIL → önerilen θ ve D)", expanded=True):
        # hedef merkezleri
        df_target_um = 0.5 * (df_min_um + df_max_um)
        delta_target_um = 0.5 * (delta_min_um + delta_max_um)

        # önerilen theta: delta = λ/(2 sin(theta/2))
        delta_target_m = delta_target_um * 1e-6
        arg = lam / (2.0 * delta_target_m)
        arg = float(np.clip(arg, 1e-12, 0.999999999))
        theta_rec_rad = 2.0 * np.arcsin(arg)
        theta_rec_mrad = theta_rec_rad * 1e3

        # önerilen D: d_f = 4 f λ/(π D)
        df_target_m = df_target_um * 1e-6
        D_rec_m = (4.0 * focus_lens.f_m * lam) / (np.pi * df_target_m)
        D_rec_mm = D_rec_m * 1e3

        # input limitlerine clamp
        theta_rec_mrad_clamped = float(np.clip(theta_rec_mrad, 0.1, 200.0))
        D_rec_mm_clamped = float(np.clip(D_rec_mm, 0.1, 20.0))

        cA, cB = st.columns(2)
        with cA:
            st.markdown("### Önerilen Crossing Angle θ")
            st.write(f"Hedef δ: **{delta_target_um:.2f} µm**")
            st.write(f"Önerilen θ: **{theta_rec_mrad_clamped:.2f} mrad**")
        with cB:
            st.markdown("### Önerilen Beam Diameter D (e⁻²)")
            st.write(f"Hedef d_f: **{df_target_um:.1f} µm**")
            st.write(f"Önerilen D: **{D_rec_mm_clamped:.2f} mm**")

        st.info("Butona basınca θ ve D, widget state'i callback ile güncellenir (Streamlit-safe).")

        # ✅ Streamlit-safe button: session_state set ONLY in callback
        st.button(
            "✅ Önerileri Uygula (θ ve D)",
            type="primary",
            key="apply_autotune",
            on_click=apply_autotune,
            kwargs={"theta_mrad": theta_rec_mrad_clamped, "D_mm": D_rec_mm_clamped},
        )
else:
    st.caption("Auto-Tune: Tüm kriterler PASS, öneri üretmeye gerek yok.")

# ------------------------------------------------------------
# 5) LAYOUT / PLOTS
# ------------------------------------------------------------
colL, colR = st.columns([1.2, 1.0])

with colL:
    st.subheader("Beam / Fringe Metrics")
    st.write(f"- Focus lens: **{focus_lens.name}**, f = **{focus_lens.f_m*1e3:.1f} mm**")
    st.write(f"- Beam diameter at lens (e^-2): **{D_e2_mm:.2f} mm**")
    st.write(f"- Focus diameter d_f: **{d_focus*1e6:.1f} µm**")
    st.write(f"- Fringe spacing δ: **{delta*1e6:.2f} µm**")
    st.write(f"- Number of fringes N_f: **{Nf:.1f}**")
    st.write(f"- Contrast (misalignment): **{contrast:.3f}**")

    st.subheader("Beam envelope vs z (conceptual)")
    z = np.linspace(-5e-3, 5e-3, 400)
    zR = np.pi * (w_focus**2) / (lam * m2)
    w_z = w_focus * np.sqrt(1.0 + (z / zR) ** 2)

    fig, ax = plt.subplots()
    ax.plot(z * 1e3, 2.0 * w_z * 1e6)
    ax.set_xlabel("z around focus (mm)")
    ax.set_ylabel("Beam diameter 2w (µm)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with colR:
    st.subheader("Power Budget")
    names = [s.name for s in stages]
    powers_mw = [s.power_w * 1e3 for s in stages]

    fig2, ax2 = plt.subplots()
    ax2.barh(names, powers_mw)
    ax2.set_xlabel("Power (mW)")
    ax2.grid(True, axis="x", alpha=0.3)
    st.pyplot(fig2)

    st.subheader("Key sensitivity check")
    r = np.sqrt((dx_um * 1e-6) ** 2 + (dy_um * 1e-6) ** 2)
    loss_db = coupling_loss_db(r, w_mode)
    st.write(f"- Fiber lateral offset r = **{r*1e6:.1f} µm**")
    st.write(f"- Small-offset coupling loss ≈ **{loss_db:.2f} dB** (your formula)")
    st.write("Not: Bu dB modeli küçük offset için iyi; büyük offsette exp-overlap daha doğru davranır.")

st.divider()
st.caption("Ön Tasarım Çalışması: Beam splitter + AOM + focusing + simple misalignment→contrast + simple power proxy. Sonraki adım: Rx fiber collection + Mie/Geometrik toplama + gerçek Thorlabs BOM.")
st.divider()
st.caption("© Prof. Dr. Arif Demir – TÜBİTAK UME NanoLDA Yazılımı. Tüm hakları saklıdır. Bu yazılımın kaynak kodu, arayüzü ve hesaplama çıktıları izin alınmadan kopyalanamaz, çoğaltılamaz, dağıtılamaz veya ticari amaçla kullanılamaz.")
