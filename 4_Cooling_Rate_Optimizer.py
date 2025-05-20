import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

st.set_page_config(
    page_title="Cooling Rate Optimizer - Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide"
)

st.title("Cooling Rate Optimizer")

st.markdown("""
This tool helps determine the optimal cooling rate for different biological samples during cryopreservation.
The cooling rate is a critical parameter that affects cell survival:

- **Too Slow**: Excessive cellular dehydration and solution effects damage
- **Too Fast**: Intracellular ice formation causes mechanical damage

The tool uses the "Two-Factor Hypothesis" model to predict cell survival as a function of cooling rate.
""")

# Create sidebar for parameters
st.sidebar.header("Cell Parameters")

cell_type = st.sidebar.selectbox(
    "Cell Type",
    ["Red Blood Cell", "Oocyte", "Sperm", "Embryo", "Stem Cell", "Fibroblast", "Custom"]
)

# Default parameters based on cell type
if cell_type == "Red Blood Cell":
    default_cell_diameter = 8.0
    default_water_permeability = 2.5
    default_activation_energy = 55.0
    default_optimal_cooling_rate = 1000
elif cell_type == "Oocyte":
    default_cell_diameter = 120.0
    default_water_permeability = 0.8
    default_activation_energy = 65.0
    default_optimal_cooling_rate = 0.5
elif cell_type == "Sperm":
    default_cell_diameter = 5.0
    default_water_permeability = 2.2
    default_activation_energy = 48.0
    default_optimal_cooling_rate = 10000
elif cell_type == "Embryo":
    default_cell_diameter = 80.0
    default_water_permeability = 1.0
    default_activation_energy = 60.0
    default_optimal_cooling_rate = 0.8
elif cell_type == "Stem Cell":
    default_cell_diameter = 15.0
    default_water_permeability = 1.5
    default_activation_energy = 58.0
    default_optimal_cooling_rate = 10
elif cell_type == "Fibroblast":
    default_cell_diameter = 20.0
    default_water_permeability = 1.8
    default_activation_energy = 52.0
    default_optimal_cooling_rate = 5
else:  # Custom
    default_cell_diameter = 20.0
    default_water_permeability = 1.5
    default_activation_energy = 55.0
    default_optimal_cooling_rate = 10

# Cell parameters
cell_diameter = st.sidebar.slider(
    "Cell Diameter (μm)",
    min_value=1.0,
    max_value=200.0,
    value=default_cell_diameter,
    step=1.0
)

water_permeability = st.sidebar.slider(
    "Water Permeability (μm/min/atm)",
    min_value=0.1,
    max_value=5.0,
    value=default_water_permeability,
    step=0.1,
    help="Membrane permeability to water at reference temperature"
)

activation_energy = st.sidebar.slider(
    "Activation Energy (kJ/mol)",
    min_value=20.0,
    max_value=80.0,
    value=default_activation_energy,
    step=1.0,
    help="Activation energy for water transport across membrane"
)

# Solution parameters
st.sidebar.header("Solution Parameters")

cryoprotectant = st.sidebar.selectbox(
    "Cryoprotectant",
    ["None", "DMSO", "Glycerol", "Ethylene Glycol", "Propylene Glycol", "Custom"]
)

# Default parameters based on cryoprotectant
if cryoprotectant == "None":
    default_cpa_concentration = 0.0
    permeability_factor = 1.0
elif cryoprotectant == "DMSO":
    default_cpa_concentration = 10.0
    permeability_factor = 0.85
elif cryoprotectant == "Glycerol":
    default_cpa_concentration = 10.0
    permeability_factor = 0.75
elif cryoprotectant == "Ethylene Glycol":
    default_cpa_concentration = 10.0
    permeability_factor = 0.90
elif cryoprotectant == "Propylene Glycol":
    default_cpa_concentration = 10.0
    permeability_factor = 0.80
else:  # Custom
    default_cpa_concentration = 10.0
    permeability_factor = 0.85

cpa_concentration = st.sidebar.slider(
    "CPA Concentration (% v/v)",
    min_value=0.0,
    max_value=40.0,
    value=default_cpa_concentration,
    step=1.0,
    help="Concentration of cryoprotective agent"
)

# Adjust water permeability based on CPA
adjusted_water_permeability = water_permeability * (1.0 - (cpa_concentration / 100) * (1.0 - permeability_factor))

# Simulation parameters
st.sidebar.header("Simulation Parameters")

min_cooling_rate = st.sidebar.number_input(
    "Minimum Cooling Rate (°C/min)",
    min_value=0.01,
    max_value=10.0,
    value=0.1,
    step=0.1,
    format="%.2f"
)

max_cooling_rate = st.sidebar.number_input(
    "Maximum Cooling Rate (°C/min)",
    min_value=10.0,
    max_value=100000.0,
    value=10000.0,
    step=10.0
)

# Calculate optimal cooling rate based on cell parameters
def calculate_optimal_cooling_rate(diameter, permeability, activation_energy):
    # Mazur's formula for optimal cooling rate (simplified)
    # B = constant related to cell properties
    B = 0.064 * permeability * (diameter ** -1) * np.exp(-activation_energy / (8.314 * 273.15))
    optimal_rate = B * 60  # Convert to °C/min
    
    return optimal_rate

# Calculate cell survival based on cooling rate
def calculate_survival(cooling_rate, optimal_rate):
    # Two-factor hypothesis model
    # Slow cooling damage (solution effects)
    slow_cooling_damage = np.exp(-10.0 / cooling_rate)
    
    # Fast cooling damage (intracellular ice formation)
    fast_cooling_damage = np.exp(-(cooling_rate / optimal_rate) ** 2)
    
    # Combined survival probability
    survival = slow_cooling_damage * fast_cooling_damage * 100  # Convert to percentage
    
    return survival

# Calculate optimal cooling rate
optimal_cooling_rate = calculate_optimal_cooling_rate(
    cell_diameter, 
    adjusted_water_permeability, 
    activation_energy
)

# Generate cooling rates (logarithmic scale)
cooling_rates = np.logspace(
    np.log10(min_cooling_rate), 
    np.log10(max_cooling_rate), 
    1000
)

# Calculate survival rates
survival_rates = [calculate_survival(rate, optimal_cooling_rate) for rate in cooling_rates]

# Find the cooling rate with maximum survival
max_survival_idx = np.argmax(survival_rates)
best_cooling_rate = cooling_rates[max_survival_idx]
max_survival = survival_rates[max_survival_idx]

# Display calculated values
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Theoretical Optimal Rate (°C/min)", f"{optimal_cooling_rate:.2f}")
with col2:
    st.metric("Best Cooling Rate (°C/min)", f"{best_cooling_rate:.2f}")
with col3:
    st.metric("Maximum Survival (%)", f"{max_survival:.1f}")

# Plot survival curve
st.subheader("Cell Survival vs. Cooling Rate")

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(cooling_rates, survival_rates, 'b-', linewidth=2)
ax.axvline(x=best_cooling_rate, color='r', linestyle='--', 
           label=f'Best Rate = {best_cooling_rate:.2f} °C/min')

# Mark the maximum survival point
ax.plot(best_cooling_rate, max_survival, 'ro', markersize=8)

# Add slow and fast cooling damage curves
slow_damage = [np.exp(-10.0 / rate) * 100 for rate in cooling_rates]
fast_damage = [np.exp(-(rate / optimal_cooling_rate) ** 2) * 100 for rate in cooling_rates]

ax.semilogx(cooling_rates, slow_damage, 'g--', alpha=0.7, linewidth=1.5, 
            label='Slow Cooling Survival')
ax.semilogx(cooling_rates, fast_damage, 'm--', alpha=0.7, linewidth=1.5, 
            label='Fast Cooling Survival')

ax.set_xlabel('Cooling Rate (°C/min)')
ax.set_ylabel('Cell Survival (%)')
ax.set_title('Predicted Cell Survival as a Function of Cooling Rate')
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.set_ylim(0, 105)
ax.legend()

st.pyplot(fig)

# Create a 3D visualization of cooling rate vs. cell size and permeability
st.subheader("3D Visualization: Optimal Cooling Rate Dependencies")

# Create parameter ranges
diameters = np.linspace(5, 100, 20)
permeabilities = np.linspace(0.5, 3.0, 20)

D, P = np.meshgrid(diameters, permeabilities)
Z = np.zeros_like(D)

for i in range(len(permeabilities)):
    for j in range(len(diameters)):
        Z[i, j] = calculate_optimal_cooling_rate(
            D[i, j], 
            P[i, j], 
            activation_energy
        )

# Create 3D plot
fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(D, P, Z, cmap=cm.coolwarm, alpha=0.8)

# Mark the current cell's position
ax3d.scatter(
    [cell_diameter], 
    [adjusted_water_permeability], 
    [optimal_cooling_rate], 
    color='black', 
    s=100, 
    marker='o'
)

ax3d.set_xlabel('Cell Diameter (μm)')
ax3d.set_ylabel('Water Permeability (μm/min/atm)')
ax3d.set_zlabel('Optimal Cooling Rate (°C/min)')
ax3d.set_title('Optimal Cooling Rate as a Function of Cell Parameters')
ax3d.set_zscale('log')

fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)

st.pyplot(fig3d)

# Create tabs for additional analyses
tab1, tab2 = st.tabs(["Cooling Protocol", "Sensitivity Analysis"])

with tab1:
    st.subheader("Recommended Cooling Protocol")
    
    # Define cooling protocol stages
    stages = [
        {"name": "Equilibration", "temp_start": 22, "temp_end": 22, "rate": 0, "time": 10},
        {"name": "Initial Cooling", "temp_start": 22, "temp_end": 0, "rate": 1, "time": (22-0)/1},
        {"name": "Nucleation", "temp_start": 0, "temp_end": -5, "rate": 1, "time": (0-(-5))/1},
        {"name": "Controlled Freezing", "temp_start": -5, "temp_end": -80, "rate": best_cooling_rate, "time": (-5-(-80))/best_cooling_rate},
        {"name": "Storage Transfer", "temp_start": -80, "temp_end": -196, "rate": 50, "time": (-80-(-196))/50}
    ]
    
    # Create protocol table
    protocol_df = pd.DataFrame(stages)
    st.table(protocol_df[["name", "temp_start", "temp_end", "rate", "time"]])
    
    # Plot cooling profile
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate time points and temperatures
    total_time = sum(stage["time"] for stage in stages)
    time_points = [0]
    temp_points = [stages[0]["temp_start"]]
    
    current_time = 0
    for stage in stages:
        current_time += stage["time"]
        time_points.append(current_time)
        temp_points.append(stage["temp_end"])
    
    # Plot temperature profile
    ax.plot(time_points, temp_points, 'b-', linewidth=2)
    ax.plot(time_points, temp_points, 'ro', markersize=6)
    
    # Add stage labels
    for i, stage in enumerate(stages):
        mid_time = (time_points[i] + time_points[i+1]) / 2
        mid_temp = (temp_points[i] + temp_points[i+1]) / 2
        ax.annotate(stage["name"], (mid_time, mid_temp), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Recommended Cooling Profile')
    ax.grid(True, alpha=0.3)
    
    # Add critical zone highlighting
    critical_zone_start = time_points[2]  # Start of nucleation
    critical_zone_end = time_points[4]    # End of controlled freezing
    ax.axvspan(critical_zone_start, critical_zone_end, alpha=0.2, color='red', 
               label='Critical Zone')
    
    ax.legend()
    
    st.pyplot(fig)
    
    st.markdown("""
    ### Protocol Notes:
    
    1. **Equilibration**: Allow cells to equilibrate with the cryoprotectant solution at room temperature.
    
    2. **Initial Cooling**: Cool to 0°C at a moderate rate to minimize thermal shock.
    
    3. **Nucleation**: Cool to -5°C and induce ice nucleation (seeding) to prevent supercooling.
    
    4. **Controlled Freezing**: Apply the optimal cooling rate calculated for your specific cell type.
    
    5. **Storage Transfer**: Transfer to liquid nitrogen storage (-196°C).
    
    ### Important Considerations:
    
    - **Seeding**: Manual ice nucleation at -5°C is recommended to control extracellular ice formation.
    - **Equipment**: Use a programmable freezer capable of maintaining the calculated optimal rate.
    - **Container**: Use appropriate containers with consistent thermal properties.
    - **Sample Volume**: Minimize sample volume to ensure uniform cooling.
    """)

with tab2:
    st.subheader("Sensitivity Analysis")
    
    # Create parameter variations
    diameter_range = np.linspace(cell_diameter * 0.5, cell_diameter * 1.5, 10)
    permeability_range = np.linspace(adjusted_water_permeability * 0.5, adjusted_water_permeability * 1.5, 10)
    activation_energy_range = np.linspace(activation_energy * 0.8, activation_energy * 1.2, 10)
    
    # Calculate optimal rates for each parameter variation
    diameter_rates = [calculate_optimal_cooling_rate(d, adjusted_water_permeability, activation_energy) for d in diameter_range]
    permeability_rates = [calculate_optimal_cooling_rate(cell_diameter, p, activation_energy) for p in permeability_range]
    energy_rates = [calculate_optimal_cooling_rate(cell_diameter, adjusted_water_permeability, e) for e in activation_energy_range]
    
    # Create sensitivity plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Diameter sensitivity
    ax1.plot(diameter_range, diameter_rates, 'b-', linewidth=2)
    ax1.axvline(x=cell_diameter, color='r', linestyle='--', 
               label=f'Current = {cell_diameter} μm')
    ax1.set_xlabel('Cell Diameter (μm)')
    ax1.set_ylabel('Optimal Cooling Rate (°C/min)')
    ax1.set_title('Sensitivity to Cell Diameter')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Permeability sensitivity
    ax2.plot(permeability_range, permeability_rates, 'g-', linewidth=2)
    ax2.axvline(x=adjusted_water_permeability, color='r', linestyle='--', 
               label=f'Current = {adjusted_water_permeability:.2f}')
    ax2.set_xlabel('Water Permeability (μm/min/atm)')
    ax2.set_ylabel('Optimal Cooling Rate (°C/min)')
    ax2.set_title('Sensitivity to Water Permeability')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Activation energy sensitivity
    ax3.plot(activation_energy_range, energy_rates, 'm-', linewidth=2)
    ax3.axvline(x=activation_energy, color='r', linestyle='--', 
               label=f'Current = {activation_energy} kJ/mol')
    ax3.set_xlabel('Activation Energy (kJ/mol)')
    ax3.set_ylabel('Optimal Cooling Rate (°C/min)')
    ax3.set_title('Sensitivity to Activation Energy')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Calculate sensitivity coefficients
    def calculate_sensitivity(param_values, rate_values, base_param, base_rate):
        # Calculate normalized sensitivity coefficient
        # S = (dR/R)/(dP/P) where R is rate and P is parameter
        sensitivities = []
        for i in range(1, len(param_values)):
            dP = param_values[i] - param_values[i-1]
            dR = rate_values[i] - rate_values[i-1]
            P_avg = (param_values[i] + param_values[i-1]) / 2
            R_avg = (rate_values[i] + rate_values[i-1]) / 2
            
            sensitivity = (dR/R_avg)/(dP/P_avg) if dP != 0 else 0
            sensitivities.append(sensitivity)
        
        return np.mean(sensitivities)
    
    diameter_sensitivity = calculate_sensitivity(
        diameter_range, diameter_rates, cell_diameter, optimal_cooling_rate)
    
    permeability_sensitivity = calculate_sensitivity(
        permeability_range, permeability_rates, adjusted_water_permeability, optimal_cooling_rate)
    
    energy_sensitivity = calculate_sensitivity(
        activation_energy_range, energy_rates, activation_energy, optimal_cooling_rate)
    
    # Display sensitivity coefficients
    sensitivity_data = {
        "Parameter": ["Cell Diameter", "Water Permeability", "Activation Energy"],
        "Sensitivity Coefficient": [diameter_sensitivity, permeability_sensitivity, energy_sensitivity],
        "Impact": ["High" if abs(s) > 1 else "Medium" if abs(s) > 0.5 else "Low" for s in 
                  [diameter_sensitivity, permeability_sensitivity, energy_sensitivity]]
    }
    
    sensitivity_df = pd.DataFrame(sensitivity_data)
    st.table(sensitivity_df)
    
    st.markdown("""
    ### Sensitivity Analysis Interpretation:
    
    The sensitivity coefficient indicates how much the optimal cooling rate changes when a parameter changes:
    
    - **Coefficient > 1**: The parameter has a high impact; small changes cause large changes in the optimal rate
    - **Coefficient 0.5-1**: The parameter has a medium impact
    - **Coefficient < 0.5**: The parameter has a low impact
    
    ### Practical Implications:
    
    - Parameters with high sensitivity require precise measurement and control
    - Consider using a range of cooling rates around the optimum for parameters with high uncertainty
    - For heterogeneous cell populations with varying sizes, use a cooling rate optimized for the most sensitive cells
    """)

# Explanation section
st.markdown("""
## Cooling Rate Theory in Cryopreservation

### The Two-Factor Hypothesis

The survival of cells during freezing is influenced by two competing factors:

1. **Solution Effects Damage (Slow Cooling)**:
   - During slow cooling, extracellular ice formation concentrates solutes outside the cell
   - This creates an osmotic gradient that draws water out of the cell
   - Excessive dehydration, high solute concentration, and prolonged exposure cause damage

2. **Intracellular Ice Formation (Fast Cooling)**:
   - During rapid cooling, cells cannot lose water fast enough to maintain equilibrium
   - The cytoplasm becomes supercooled and eventually forms intracellular ice
   - Ice crystals cause mechanical damage to cellular structures

### Optimal Cooling Rate

The optimal cooling rate balances these two factors:
- Fast enough to minimize solution effects damage
- Slow enough to prevent intracellular ice formation

### Factors Affecting Optimal Cooling Rate

1. **Cell Size**: Smaller cells have higher surface-to-volume ratios, allowing faster water efflux and thus tolerating faster cooling rates.

2. **Membrane Permeability**: Higher water permeability allows faster dehydration, permitting faster cooling rates.

3. **Activation Energy**: Lower activation energy for water transport allows water to move across the membrane more easily at lower temperatures.

4. **Cryoprotectants**: CPAs modify both the likelihood of ice formation and the rate of cellular dehydration.

### Practical Applications

- **Cell-Specific Protocols**: Different cell types require customized cooling rates
- **CPA Optimization**: Cryoprotectants can shift the optimal cooling rate
- **Controlled-Rate Freezers**: Programmable freezers allow precise implementation of optimal cooling profiles
""")