import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

st.set_page_config(
    page_title="Nucleation Energy Barrier - Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide"
)

st.title("Nucleation Activation Energy Barrier Calculator")

st.markdown("""
This tool calculates and visualizes the energy barrier for ice nucleation in aqueous solutions.
The classical nucleation theory describes the free energy change (ΔG) associated with the formation 
of a spherical ice nucleus in a supercooled liquid:

$$\Delta G(r) = 4\pi r^2 \gamma - \frac{4}{3}\pi r^3 \Delta G_v$$

Where:
- $r$ is the radius of the ice nucleus
- $\gamma$ is the ice-water interfacial energy
- $\Delta G_v$ is the volumetric free energy difference between ice and liquid water
""")

# Create sidebar for parameters
st.sidebar.header("Nucleation Parameters")

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=-40.0,
    max_value=0.0,
    value=-10.0,
    step=0.5,
    help="Supercooling temperature below 0°C"
)

gamma = st.sidebar.slider(
    "Ice-Water Interfacial Energy (mJ/m²)",
    min_value=20.0,
    max_value=40.0,
    value=30.0,
    step=0.5,
    help="Surface energy between ice and liquid water"
)

cryoprotectant = st.sidebar.selectbox(
    "Cryoprotectant Type",
    ["None", "Glycerol", "DMSO", "Ethylene Glycol", "Propylene Glycol"]
)

concentration = st.sidebar.slider(
    "Cryoprotectant Concentration (% w/v)",
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=5.0,
    disabled=(cryoprotectant == "None"),
    help="Concentration of cryoprotectant in solution"
)

# Calculate parameters based on inputs
def calculate_delta_gv(temp, cryo_type, cryo_conc):
    # Base volumetric free energy difference (J/m³)
    # Proportional to supercooling degree
    base_delta_gv = 1.6e6 * abs(temp) / 10
    
    # Adjust for cryoprotectant effect
    if cryo_type == "None" or cryo_conc == 0:
        return base_delta_gv
    
    # Different cryoprotectants affect nucleation differently
    cryo_factors = {
        "Glycerol": 0.85,
        "DMSO": 0.75,
        "Ethylene Glycol": 0.80,
        "Propylene Glycol": 0.78
    }
    
    # Calculate reduction factor based on concentration
    reduction = 1.0 - (cryo_factors[cryo_type] * (cryo_conc / 100))
    
    return base_delta_gv * reduction

# Calculate the adjusted interfacial energy
def calculate_adjusted_gamma(base_gamma, cryo_type, cryo_conc):
    if cryo_type == "None" or cryo_conc == 0:
        return base_gamma * 1e-3  # Convert to J/m²
    
    # Different cryoprotectants affect interfacial energy
    gamma_factors = {
        "Glycerol": 1.15,
        "DMSO": 1.25,
        "Ethylene Glycol": 1.18,
        "Propylene Glycol": 1.20
    }
    
    # Calculate increase factor based on concentration
    increase = 1.0 + (gamma_factors[cryo_type] - 1.0) * (cryo_conc / 100)
    
    return (base_gamma * increase) * 1e-3  # Convert to J/m²

# Calculate the nucleation energy barrier
def calculate_energy_barrier(r, gamma, delta_gv):
    return 4 * np.pi * r**2 * gamma - (4/3) * np.pi * r**3 * delta_gv

# Calculate the critical radius
def calculate_critical_radius(gamma, delta_gv):
    return 2 * gamma / delta_gv

# Calculate the maximum energy barrier
def calculate_max_barrier(gamma, delta_gv):
    r_critical = calculate_critical_radius(gamma, delta_gv)
    return calculate_energy_barrier(r_critical, gamma, delta_gv)

# Main calculations
delta_gv = calculate_delta_gv(temperature, cryoprotectant, concentration)
adjusted_gamma = calculate_adjusted_gamma(gamma, cryoprotectant, concentration)
r_critical = calculate_critical_radius(adjusted_gamma, delta_gv)
max_barrier = calculate_max_barrier(adjusted_gamma, delta_gv)

# Display calculated values
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Critical Radius (nm)", f"{r_critical*1e9:.2f}")
with col2:
    st.metric("Maximum Energy Barrier (J)", f"{max_barrier:.2e}")
with col3:
    st.metric("Volumetric Free Energy (J/m³)", f"{delta_gv:.2e}")

# Plot the energy barrier curve
st.subheader("Nucleation Energy Barrier Curve")

r_values = np.linspace(0, r_critical*3, 1000)
energy_values = [calculate_energy_barrier(r, adjusted_gamma, delta_gv) for r in r_values]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(r_values*1e9, energy_values*1e21, 'b-', linewidth=2)
ax.axvline(x=r_critical*1e9, color='r', linestyle='--', label=f'Critical Radius = {r_critical*1e9:.2f} nm')
ax.axhline(y=max_barrier*1e21, color='g', linestyle='--', label=f'Max Barrier = {max_barrier*1e21:.2f} zJ')
ax.set_xlabel('Nucleus Radius (nm)')
ax.set_ylabel('Free Energy Change (zJ)')
ax.set_title('Ice Nucleation Energy Barrier')
ax.grid(True)
ax.legend()

# Mark the critical point
ax.plot(r_critical*1e9, max_barrier*1e21, 'ro')

st.pyplot(fig)

# 3D visualization of energy barrier with temperature and radius
st.subheader("3D Visualization of Energy Barrier")

# Create temperature range
temp_range = np.linspace(-40, -1, 20)
r_range = np.linspace(0, r_critical*4, 50)
R, T = np.meshgrid(r_range, temp_range)

# Calculate energy barrier for each point
Z = np.zeros_like(R)
for i, temp in enumerate(temp_range):
    delta_gv_temp = calculate_delta_gv(temp, cryoprotectant, concentration)
    for j, r in enumerate(r_range):
        Z[i, j] = calculate_energy_barrier(r, adjusted_gamma, delta_gv_temp) * 1e21  # Convert to zJ

fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(R*1e9, T, Z, cmap=cm.coolwarm, alpha=0.8)
ax3d.set_xlabel('Nucleus Radius (nm)')
ax3d.set_ylabel('Temperature (°C)')
ax3d.set_zlabel('Energy Barrier (zJ)')
ax3d.set_title('Energy Barrier as a Function of Radius and Temperature')
fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)

st.pyplot(fig3d)

# Additional information
st.markdown("""
### Nucleation Theory Insights

The plot above shows how the energy barrier changes with nucleus radius. Key points:

1. **Sub-critical nuclei** (r < r_critical): These small ice nuclei are unstable and tend to melt.
2. **Critical nucleus** (r = r_critical): This represents the minimum size an ice nucleus must reach to become stable.
3. **Post-critical nuclei** (r > r_critical): These larger ice nuclei are stable and will continue to grow.

The energy barrier height determines the nucleation rate - higher barriers lead to lower nucleation probability.

### Effects of Cryoprotectants

Cryoprotectants generally:
- Increase the ice-water interfacial energy (γ)
- Decrease the volumetric free energy difference (ΔGv)

Both effects increase the energy barrier, reducing the probability of ice nucleation and promoting vitrification.
""")

# Download data option
if st.button("Generate Report Data"):
    r_data = r_values * 1e9  # Convert to nm
    energy_data = energy_values * 1e21  # Convert to zJ
    
    data = pd.DataFrame({
        'Radius (nm)': r_data,
        'Energy Barrier (zJ)': energy_data
    })
    
    st.dataframe(data)
    
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"nucleation_barrier_T{temperature}_C{concentration}.csv",
        mime="text/csv"
    )