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

def calculate_approx_exceed_ratio(E_threshold, T):
    # 計算近似的 exceed ratio，使用 e^(-E/kT)
    if E_threshold <= 0:
        return 100.0  # 如果閾值為 0 或負數，所有粒子都超過閾值
    return np.exp(-E_threshold / (k_B * T)) * 100  # 轉換為百分比

# 設定 Streamlit 頁面標題
st.title("Maxwell-Boltzmann 動能分布")
st.markdown("比較不同溫度下動能分布的差異，建議手機橫屏查看。")

# 使用 st.expander 將滑桿摺疊起來，節省空間
with st.expander("分布控制", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        temp_kelvin_1 = st.slider("溫度 (K) - 左側", min_value=100, max_value=1000, step=50, value=300, key="temp1")
        energy_threshold_1 = st.slider("閾值 (×1e⁻²¹ J) - 左側", min_value=0, max_value=50, step=2, value=20, key="threshold1")
    with col2:
        temp_kelvin_2 = st.slider("溫度 (K) - 右側", min_value=100, max_value=1000, step=50, value=500, key="temp2")
        energy_threshold_2 = st.slider("閾值 (×1e⁻²¹ J) - 右側", min_value=0, max_value=50, step=2, value=30, key="threshold2")

# 計算動能閾值（單位：焦耳）
energy_threshold_j_1 = energy_threshold_1 * 1e-21
energy_threshold_j_2 = energy_threshold_2 * 1e-21

# 數據生成 - 左側
mass_kg = 39.95 * atomic_mass
n_particles = 100000
sigma_1 = np.sqrt(k_B * temp_kelvin_1 / mass_kg)
velocities_1 = np.random.normal(0, sigma_1, (n_particles,
