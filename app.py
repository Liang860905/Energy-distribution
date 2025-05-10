import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as k_B, atomic_mass
import streamlit as st

# 預設座標軸範圍
X_MIN, X_MAX = 0, 6e-20
Y_MIN, Y_MAX = 0, 15e+19

def mb_kinetic_energy_dist(E, T):
    # Maxwell-Boltzmann kinetic energy distribution for 3D
    return (2/np.sqrt(np.pi)) * (1/(k_B*T)**1.5) * np.sqrt(E) * np.exp(-E/(k_B*T))

# 設定 Streamlit 頁面標題
st.title("Maxwell-Boltzmann 動能分布模擬")

# 使用 Streamlit 滑桿來讓使用者調整溫度與動能閾值
temp_kelvin = st.slider("溫度 (K)", min_value=100, max_value=1000, step=50, value=300)
energy_threshold = st.slider("動能閾值 (×1e⁻²¹ J)", min_value=0, max_value=50, step=2, value=20)

energy_threshold_j = energy_threshold * 1e-21

# 數據生成
mass_kg = 39.95 * atomic_mass
n_particles = 100000
sigma = np.sqrt(k_B * temp_kelvin / mass_kg)
velocities = np.random.normal(0, sigma, (n_particles, 3))
speeds = np.linalg.norm(velocities, axis=1)
energies = 0.5 * mass_kg * speeds**2

# 繪圖
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)

# 直方圖
bins = np.linspace(X_MIN, X_MAX, 361)
ax.hist(energies, bins=bins, density=True, alpha=0.6, color='dodgerblue', label='Simulation')

# 理論分布曲線
E_plot = np.linspace(X_MIN, X_MAX, 1000)
theory = mb_kinetic_energy_dist(E_plot, temp_kelvin)
ax.plot(E_plot, theory, 'r-', lw=2, label='Maxwell-Boltzmann Theory')

# 低限能
ax.axvline(energy_threshold_j, color='limegreen', ls='--', lw=2)

# 文字標籤
exceed_ratio = (energies > energy_threshold_j).mean() * 100
ax.text(0.68, 0.85, f'exceed_ratio: {exceed_ratio:.2f}%',
        transform=ax.transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))

# 圖表設置
plt.xlabel('Kinetic energy (J)', fontsize=12)
plt.ylabel('prob. density', fontsize=12)
plt.title(f'temp. {temp_kelvin} K Kinetic energy distribution', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# 在 Streamlit 上顯示圖表
st.pyplot(fig)
