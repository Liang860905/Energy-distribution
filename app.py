import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as k_B, atomic_mass
import streamlit as st

# 設定 Streamlit 頁面配置以適應手機顯示
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# 預設座標軸範圍
X_MIN, X_MAX = 0, 6e-20
Y_MIN, Y_MAX = 0, 15e+19

def mb_kinetic_energy_dist(E, T):
    # Maxwell-Boltzmann kinetic energy distribution for 3D
    return (2/np.sqrt(np.pi)) * (1/(k_B*T)**1.5) * np.sqrt(E) * np.exp(-E/(k_B*T))

# 設定 Streamlit 頁面標題
st.title("Maxwell-Boltzmann 動能分布模擬 - 上下對照")
st.markdown("此工具允許您比較不同溫度下動能分布的差異，適合手機與桌面顯示。")

# 第一組滑桿（上側）
st.subheader("上側分布")
temp_kelvin_1 = st.slider("溫度 (K) - 上側", min_value=100, max_value=1000, step=50, value=300, key="temp1")
energy_threshold_1 = st.slider("動能閾值 (×1e⁻²¹ J) - 上側", min_value=0, max_value=50, step=2, value=20, key="threshold1")

# 第二組滑桿（下側）
st.subheader("下側分布")
temp_kelvin_2 = st.slider("溫度 (K) - 下側", min_value=100, max_value=1000, step=50, value=500, key="temp2")
energy_threshold_2 = st.slider("動能閾值 (×1e⁻²¹ J) - 下側", min_value=0, max_value=50, step=2, value=30, key="threshold2")

# 計算動能閾值（單位：焦耳）
energy_threshold_j_1 = energy_threshold_1 * 1e-21
energy_threshold_j_2 = energy_threshold_2 * 1e-21

# 數據生成 - 上側
mass_kg = 39.95 * atomic_mass
n_particles = 100000
sigma_1 = np.sqrt(k_B * temp_kelvin_1 / mass_kg)
velocities_1 = np.random.normal(0, sigma_1, (n_particles, 3))
speeds_1 = np.linalg.norm(velocities_1, axis=1)
energies_1 = 0.5 * mass_kg * speeds_1**2

# 數據生成 - 下側
sigma_2 = np.sqrt(k_B * temp_kelvin_2 / mass_kg)
velocities_2 = np.random.normal(0, sigma_2, (n_particles, 3))
speeds_2 = np.linalg.norm(velocities_2, axis=1)
energies_2 = 0.5 * mass_kg * speeds_2**2

# 建立上下兩張圖表，調整尺寸以適應手機顯示
fig1, ax1 = plt.subplots(figsize=(8, 5))
fig2, ax2 = plt.subplots(figsize=(8, 5))

# 上側圖表設置
ax1.set_xlim(X_MIN, X_MAX)
ax1.set_ylim(Y_MIN, Y_MAX)
bins = np.linspace(X_MIN, X_MAX, 361)
ax1.hist(energies_1, bins=bins, density=True, alpha=0.6, color='dodgerblue', label='Simulation')
E_plot = np.linspace(X_MIN, X_MAX, 1000)
theory_1 = mb_kinetic_energy_dist(E_plot, temp_kelvin_1)
ax1.plot(E_plot, theory_1, 'r-', lw=2, label='Maxwell-Boltzmann Theory')
ax1.axvline(energy_threshold_j_1, color='limegreen', ls='--', lw=2)
exceed_ratio_1 = (energies_1 > energy_threshold_j_1).mean() * 100
ax1.text(0.55, 0.85, f'exceed_ratio: {exceed_ratio_1:.2f}%',
         transform=ax1.transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))
ax1.set_xlabel('Kinetic energy (J)', fontsize=10)
ax1.set_ylabel('prob. density', fontsize=10)
ax1.set_title(f'temp. {temp_kelvin_1} K', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

# 下側圖表設置
ax2.set_xlim(X_MIN, X_MAX)
ax2.set_ylim(Y_MIN, Y_MAX)
ax2.hist(energies_2, bins=bins, density=True, alpha=0.6, color='dodgerblue', label='Simulation')
theory_2 = mb_kinetic_energy_dist(E_plot, temp_kelvin_2)
ax2.plot(E_plot, theory_2, 'r-', lw=2, label='Maxwell-Boltzmann Theory')
ax2.axvline(energy_threshold_j_2, color='limegreen', ls='--', lw=2)
exceed_ratio_2 = (energies_2 > energy_threshold_j_2).mean() * 100
ax2.text(0.55, 0.85, f'exceed_ratio: {exceed_ratio_2:.2f}%',
         transform=ax2.transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))
ax2.set_xlabel('Kinetic energy (J)', fontsize=10)
ax2.set_ylabel('prob. density', fontsize=10)
ax2.set_title(f'temp. {temp_kelvin_2} K', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

# 顯示上下圖表
st.pyplot(fig1, use_container_width=True)
st.pyplot(fig2, use_container_width=True)
