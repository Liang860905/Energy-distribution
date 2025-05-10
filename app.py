import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as k_B, atomic_mass
from scipy.special import erf
import streamlit as st

# 設定 Streamlit 頁面配置以適應手機顯示
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# 預設座標軸範圍
X_MIN, X_MAX = 0, 6e-20
Y_MIN, Y_MAX = 0, 15e+19

def mb_kinetic_energy_dist(E, T):
    # Maxwell-Boltzmann kinetic energy distribution for 3D
    return (2/np.sqrt(np.pi)) * (1/(k_B*T)**1.5) * np.sqrt(E) * np.exp(-E/(k_B*T))

def calculate_theoretical_exceed_ratio(E_threshold, T):
    # 計算理論上的 exceed ratio，使用解析解
    if E_threshold <= 0:
        return 100.0  # 如果閾值為 0 或負數，所有粒子都超過閾值
    x = np.sqrt(E_threshold / (k_B * T))
    exceed_ratio = 1 - erf(x) + (2 / np.sqrt(np.pi)) * x * np.exp(-x**2)
    return exceed_ratio * 100  # 轉換為百分比

# 設定 Streamlit 頁面標題
st.title("Maxwell-Boltzmann 動能分布")
st.markdown("比較不同溫度下動能分布的差異，建議手機橫屏查看。")

# 使用 st.expander 將滑桿摺疊起來，節省空間
with st.expander("分布控制", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        temp_kelvin_1 = st.slider("溫度 (K) - 左側", min_value=100, max_value=1000, step=50, value=300, key="temp1")
        energy_threshold_1 = st.slider("閾值 (×1e⁻²¹ J) - 左側", min_value=0, max_value=50, step=2, value=30, key="threshold1")
    with col2:
        temp_kelvin_2 = st.slider("溫度 (K) - 右側", min_value=100, max_value=1000, step=50, value=300, key="temp2")
        energy_threshold_2 = st.slider("閾值 (×1e⁻²¹ J) - 右側", min_value=0, max_value=50, step=2, value=30, key="threshold2")

# 計算動能閾值（單位：焦耳）
energy_threshold_j_1 = energy_threshold_1 * 1e-21
energy_threshold_j_2 = energy_threshold_2 * 1e-21

# 數據生成 - 左側
mass_kg = 39.95 * atomic_mass
n_particles = 100000
sigma_1 = np.sqrt(k_B * temp_kelvin_1 / mass_kg)
velocities_1 = np.random.normal(0, sigma_1, (n_particles, 3))
speeds_1 = np.linalg.norm(velocities_1, axis=1)
energies_1 = 0.5 * mass_kg * speeds_1**2

# 數據生成 - 右側
sigma_2 = np.sqrt(k_B * temp_kelvin_2 / mass_kg)
velocities_2 = np.random.normal(0, sigma_2, (n_particles, 3))
speeds_2 = np.linalg.norm(velocities_2, axis=1)
energies_2 = 0.5 * mass_kg * speeds_2**2

# 計算模擬的 exceed ratio
exceed_ratio_sim_1 = (energies_1 > energy_threshold_j_1).mean() * 100
exceed_ratio_sim_2 = (energies_2 > energy_threshold_j_2).mean() * 100

# 計算理論的 exceed ratio（精確解析解）
exceed_ratio_theory_1 = calculate_theoretical_exceed_ratio(energy_threshold_j_1, temp_kelvin_1)
exceed_ratio_theory_2 = calculate_theoretical_exceed_ratio(energy_threshold_j_2, temp_kelvin_2)

# 建立左右兩張圖表，調整尺寸以適應手機橫屏顯示
fig1, ax1 = plt.subplots(figsize=(5, 3.5))
fig2, ax2 = plt.subplots(figsize=(5, 3.5))

# 左側圖表設置
ax1.set_xlim(X_MIN, X_MAX)
ax1.set_ylim(Y_MIN, Y_MAX)
bins = np.linspace(X_MIN, X_MAX, 361)
ax1.hist(energies_1, bins=bins, density=True, alpha=0.6, color='dodgerblue', label='Simulation')
E_plot = np.linspace(X_MIN, X_MAX, 1000)
theory_1 = mb_kinetic_energy_dist(E_plot, temp_kelvin_1)
ax1.plot(E_plot, theory_1, 'r-', lw=2, label='MB Theory')
ax1.axvline(energy_threshold_j_1, color='limegreen', ls='--', lw=2)
ax1.text(0.55, 0.85, f'exceed (sim): {exceed_ratio_sim_1:.2f}%', 
         transform=ax1.transAxes, fontsize=8,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
ax1.text(0.55, 0.75, f'exceed (theory): {exceed_ratio_theory_1:.2f}%', 
         transform=ax1.transAxes, fontsize=8,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
ax1.set_xlabel('Kinetic energy (J)', fontsize=8)
ax1.set_ylabel('prob. density', fontsize=8)
ax1.set_title(f'temp. {temp_kelvin_1} K', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=7)

# 右側圖表設置
ax2.set_xlim(X_MIN, X_MAX)
ax2.set_ylim(Y_MIN, Y_MAX)
ax2.hist(energies_2, bins=bins, density=True, alpha=0.6, color='dodgerblue', label='Simulation')
theory_2 = mb_kinetic_energy_dist(E_plot, temp_kelvin_2)
ax2.plot(E_plot, theory_2, 'r-', lw=2, label='MB Theory')
ax2.axvline(energy_threshold_j_2, color='limegreen', ls='--', lw=2)
ax2.text(0.55, 0.85, f'exceed (sim): {exceed_ratio_sim_2:.2f}%', 
         transform=ax2.transAxes, fontsize=8,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
ax2.text(0.55, 0.75, f'exceed (theory): {exceed_ratio_theory_2:.2f}%', 
         transform=ax2.transAxes, fontsize=8,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
ax2.set_xlabel('Kinetic energy (J)', fontsize=8)
ax2.set_ylabel('prob. density', fontsize=8)
ax2.set_title(f'temp. {temp_kelvin_2} K', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=7)

# 使用 Streamlit 的 columns 顯示左右圖表
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1, use_container_width=True)
with col2:
    st.pyplot(fig2, use_container_width=True)
