import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

st.set_page_config(
    page_title="Cryoprotectant Toxicity Analyzer - Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide"
)

st.title("Cryoprotectant Toxicity Analyzer")

st.markdown("""
This tool analyzes the toxicity profiles of common cryoprotective agents (CPAs) and helps optimize 
CPA concentration and exposure time to maximize cell survival.

Cryoprotectants are essential for successful cryopreservation but can cause chemical toxicity and osmotic damage.
This tool models CPA toxicity based on concentration, exposure time, and temperature.
""")

# Create sidebar for parameters
st.sidebar.header("Cell Parameters")

cell_type = st.sidebar.selectbox(
    "Cell Type",
    ["Red Blood Cell", "Oocyte", "Sperm", "Embryo", "Stem Cell", "Fibroblast", "Custom"]
)

# Default parameters based on cell type
if cell_type == "Red Blood Cell":
    default_toxicity_tolerance = 0.8
elif cell_type == "Oocyte":
    default_toxicity_tolerance = 0.5
elif cell_type == "Sperm":
    default_toxicity_tolerance = 0.7
elif cell_type == "Embryo":
    default_toxicity_tolerance = 0.4
elif cell_type == "Stem Cell":
    default_toxicity_tolerance = 0.6
elif cell_type == "Fibroblast":
    default_toxicity_tolerance = 0.7
else:  # Custom
    default_toxicity_tolerance = 0.6

toxicity_tolerance = st.sidebar.slider(
    "Toxicity Tolerance Factor",
    min_value=0.1,
    max_value=1.0,
    value=default_toxicity_tolerance,
    step=0.1,
    help="Relative tolerance to CPA toxicity (higher = more tolerant)"
)

# CPA parameters
st.sidebar.header("Cryoprotectant Parameters")

cpa_type = st.sidebar.selectbox(
    "Primary Cryoprotectant",
    ["DMSO", "Glycerol", "Ethylene Glycol", "Propylene Glycol", "Methanol", "Custom"]
)

# Default parameters based on CPA type
if cpa_type == "DMSO":
    default_toxicity_factor = 1.0
    default_permeability = 1.0
    default_max_concentration = 15.0
elif cpa_type == "Glycerol":
    default_toxicity_factor = 0.7
    default_permeability = 0.6
    default_max_concentration = 20.0
elif cpa_type == "Ethylene Glycol":
    default_toxicity_factor = 0.8
    default_permeability = 1.2
    default_max_concentration = 15.0
elif cpa_type == "Propylene Glycol":
    default_toxicity_factor = 0.9
    default_permeability = 0.8
    default_max_concentration = 12.0
elif cpa_type == "Methanol":
    default_toxicity_factor = 0.6
    default_permeability = 1.5
    default_max_concentration = 10.0
else:  # Custom
    default_toxicity_factor = 0.8
    default_permeability = 1.0
    default_max_concentration = 15.0

toxicity_factor = st.sidebar.slider(
    "Relative Toxicity Factor",
    min_value=0.1,
    max_value=2.0,
    value=default_toxicity_factor,
    step=0.1,
    help="Relative toxicity of the CPA (higher = more toxic)"
)

permeability = st.sidebar.slider(
    "Membrane Permeability Factor",
    min_value=0.1,
    max_value=2.0,
    value=default_permeability,
    step=0.1,
    help="Relative permeability of cell membrane to the CPA (higher = faster equilibration)"
)

# Add option for secondary CPA
use_secondary_cpa = st.sidebar.checkbox("Use Secondary Cryoprotectant")

if use_secondary_cpa:
    secondary_cpa_type = st.sidebar.selectbox(
        "Secondary Cryoprotectant",
        ["DMSO", "Glycerol", "Ethylene Glycol", "Propylene Glycol", "Methanol", "Trehalose", "Sucrose"],
        index=5  # Default to Trehalose
    )
    
    secondary_cpa_concentration = st.sidebar.slider(
        f"{secondary_cpa_type} Concentration (% v/v)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5
    )
else:
    secondary_cpa_type = None
    secondary_cpa_concentration = 0.0

# Exposure parameters
st.sidebar.header("Exposure Parameters")

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=40.0,
    value=22.0,
    step=1.0
)

max_exposure_time = st.sidebar.slider(
    "Maximum Exposure Time (min)",
    min_value=1,
    max_value=120,
    value=30,
    step=1
)

# Define toxicity model functions
def calculate_toxicity(concentration, time, temperature, toxicity_factor):
    # Convert temperature to Kelvin for Arrhenius equation
    temp_k = temperature + 273.15
    
    # Base toxicity rate (higher at higher temperatures)
    base_rate = np.exp(-5000 / temp_k)
    
    # Concentration effect (non-linear)
    conc_effect = (concentration / 10) ** 1.5
    
    # Calculate cumulative toxicity
    toxicity = base_rate * conc_effect * toxicity_factor * time
    
    return toxicity

def calculate_survival(toxicity, tolerance):
    # Survival model based on cumulative toxicity
    survival = 100 * np.exp(-toxicity / tolerance)
    return max(0, min(100, survival))

def calculate_equilibration(time, permeability, temperature):
    # Convert temperature to Kelvin
    temp_k = temperature + 273.15
    
    # Temperature effect on permeability (Arrhenius relationship)
    temp_factor = np.exp(2000 * (1/273.15 - 1/temp_k))
    
    # Equilibration percentage (asymptotic approach to 100%)
    equilibration = 100 * (1 - np.exp(-time * permeability * temp_factor / 10))
    
    return equilibration

# Generate concentration range
concentration_range = np.linspace(0, default_max_concentration * 1.5, 50)

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Concentration-Time Analysis", "Temperature Effects", "CPA Comparison"])

with tab1:
    st.subheader("Concentration-Time Analysis")
    
    # Create 2D grid for concentration and time
    concentrations = np.linspace(0, default_max_concentration * 1.5, 50)
    times = np.linspace(0, max_exposure_time, 50)
    C, T = np.meshgrid(concentrations, times)
    
    # Calculate toxicity and survival for each point
    Z_toxicity = np.zeros_like(C)
    Z_survival = np.zeros_like(C)
    Z_equilibration = np.zeros_like(C)
    Z_effective = np.zeros_like(C)
    
    for i in range(len(times)):
        for j in range(len(concentrations)):
            tox = calculate_toxicity(C[i, j], T[i, j], temperature, toxicity_factor)
            Z_toxicity[i, j] = tox
            Z_survival[i, j] = calculate_survival(tox, toxicity_tolerance)
            Z_equilibration[i, j] = calculate_equilibration(T[i, j], permeability, temperature)
            
            # Effective cryoprotection (balance of equilibration and survival)
            Z_effective[i, j] = (Z_equilibration[i, j] / 100) * (Z_survival[i, j] / 100) * 100
    
    # Find optimal point (maximum effective cryoprotection)
    optimal_idx = np.unravel_index(np.argmax(Z_effective), Z_effective.shape)
    optimal_conc = C[optimal_idx]
    optimal_time = T[optimal_idx]
    optimal_effectiveness = Z_effective[optimal_idx]
    optimal_survival = Z_survival[optimal_idx]
    optimal_equilibration = Z_equilibration[optimal_idx]
    
    # Display optimal values
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimal Concentration", f"{optimal_conc:.1f}% v/v")
    with col2:
        st.metric("Optimal Exposure Time", f"{optimal_time:.1f} min")
    with col3:
        st.metric("Effectiveness", f"{optimal_effectiveness:.1f}%")
    
    # Create heatmap of effective cryoprotection
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(C, T, Z_effective, 20, cmap='viridis')
    ax.set_xlabel(f'{cpa_type} Concentration (% v/v)')
    ax.set_ylabel('Exposure Time (min)')
    ax.set_title('Effective Cryoprotection (Equilibration × Survival)')
    
    # Add contour lines for survival and equilibration
    survival_contour = ax.contour(C, T, Z_survival, levels=[90, 80, 70, 50], colors='white', linestyles='dashed', linewidths=1)
    ax.clabel(survival_contour, inline=True, fontsize=8, fmt='%1.0f%% Survival')
    
    equilibration_contour = ax.contour(C, T, Z_equilibration, levels=[90, 95, 99], colors='red', linestyles='dotted', linewidths=1)
    ax.clabel(equilibration_contour, inline=True, fontsize=8, fmt='%1.0f%% Equilibration')
    
    # Mark optimal point
    ax.plot(optimal_conc, optimal_time, 'ro', markersize=10)
    ax.annotate(f'Optimal: {optimal_effectiveness:.1f}%', 
                (optimal_conc, optimal_time), 
                xytext=(10, 10), 
                textcoords='offset points',
                color='white',
                fontweight='bold')
    
    fig.colorbar(contour, ax=ax, label='Effectiveness (%)')
    
    st.pyplot(fig)
    
    # Create line plots for specific concentrations
    st.subheader("Time-Dependent Effects at Different Concentrations")
    
    selected_concentrations = [5, 10, 15, 20]
    if optimal_conc not in selected_concentrations:
        selected_concentrations.append(round(optimal_conc))
        selected_concentrations.sort()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    for conc in selected_concentrations:
        if conc > max(concentrations):
            continue
            
        # Find closest concentration in our grid
        conc_idx = np.argmin(np.abs(concentrations - conc))
        
        # Extract data for this concentration
        time_data = times
        survival_data = Z_survival[:, conc_idx]
        equilibration_data = Z_equilibration[:, conc_idx]
        effective_data = Z_effective[:, conc_idx]
        
        # Plot data
        ax1.plot(time_data, survival_data, label=f'{conc}% v/v')
        ax2.plot(time_data, equilibration_data, label=f'{conc}% v/v')
        ax3.plot(time_data, effective_data, label=f'{conc}% v/v')
    
    ax1.set_ylabel('Cell Survival (%)')
    ax1.set_title('Cell Survival vs. Exposure Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_ylabel('CPA Equilibration (%)')
    ax2.set_title('CPA Equilibration vs. Exposure Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_xlabel('Exposure Time (min)')
    ax3.set_ylabel('Effectiveness (%)')
    ax3.set_title('Overall Effectiveness vs. Exposure Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Temperature Effects on CPA Toxicity and Equilibration")
    
    # Create temperature range
    temperatures = np.linspace(-10, 40, 50)
    
    # Calculate toxicity and equilibration at different temperatures
    toxicity_data = []
    equilibration_data = []
    effectiveness_data = []
    
    for temp in temperatures:
        tox = calculate_toxicity(optimal_conc, optimal_time, temp, toxicity_factor)
        surv = calculate_survival(tox, toxicity_tolerance)
        equil = calculate_equilibration(optimal_time, permeability, temp)
        effect = (equil / 100) * (surv / 100) * 100
        
        toxicity_data.append(tox)
        equilibration_data.append(equil)
        effectiveness_data.append(effect)
    
    # Find optimal temperature
    optimal_temp_idx = np.argmax(effectiveness_data)
    optimal_temp = temperatures[optimal_temp_idx]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot toxicity and survival
    color1 = 'tab:red'
    ax1.plot(temperatures, toxicity_data, color=color1, linewidth=2)
    ax1.set_ylabel('Cumulative Toxicity', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax1b = ax1.twinx()
    color2 = 'tab:blue'
    ax1b.plot(temperatures, [calculate_survival(t, toxicity_tolerance) for t in toxicity_data], 
             color=color2, linewidth=2)
    ax1b.set_ylabel('Cell Survival (%)', color=color2)
    ax1b.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title(f'Toxicity and Survival vs. Temperature ({optimal_conc:.1f}% {cpa_type}, {optimal_time:.1f} min)')
    ax1.grid(True, alpha=0.3)
    
    # Plot equilibration and effectiveness
    color3 = 'tab:green'
    ax2.plot(temperatures, equilibration_data, color=color3, linewidth=2)
    ax2.set_ylabel('CPA Equilibration (%)', color=color3)
    ax2.tick_params(axis='y', labelcolor=color3)
    
    ax2b = ax2.twinx()
    color4 = 'tab:purple'
    ax2b.plot(temperatures, effectiveness_data, color=color4, linewidth=2)
    ax2b.set_ylabel('Overall Effectiveness (%)', color=color4)
    ax2b.tick_params(axis='y', labelcolor=color4)
    
    # Mark optimal temperature
    ax2.axvline(x=optimal_temp, color='black', linestyle='--', alpha=0.7)
    ax2.annotate(f'Optimal: {optimal_temp:.1f}°C', 
                (optimal_temp, 50), 
                xytext=(5, 10), 
                textcoords='offset points')
    
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_title('Equilibration and Effectiveness vs. Temperature')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Create 3D visualization of temperature, concentration, and effectiveness
    st.subheader("3D Visualization: Temperature-Concentration-Effectiveness")
    
    # Create parameter ranges for 3D plot
    temp_range = np.linspace(-10, 40, 15)
    conc_range = np.linspace(0, default_max_concentration * 1.5, 15)
    
    T_mesh, C_mesh = np.meshgrid(temp_range, conc_range)
    Z_effect_3d = np.zeros_like(T_mesh)
    
    for i in range(len(conc_range)):
        for j in range(len(temp_range)):
            tox = calculate_toxicity(C_mesh[i, j], optimal_time, T_mesh[i, j], toxicity_factor)
            surv = calculate_survival(tox, toxicity_tolerance)
            equil = calculate_equilibration(optimal_time, permeability, T_mesh[i, j])
            Z_effect_3d[i, j] = (equil / 100) * (surv / 100) * 100
    
    # Create 3D plot
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    surf = ax3d.plot_surface(T_mesh, C_mesh, Z_effect_3d, cmap=cm.viridis, alpha=0.8)
    
    # Mark optimal point
    optimal_3d_idx = np.unravel_index(np.argmax(Z_effect_3d), Z_effect_3d.shape)
    optimal_temp_3d = T_mesh[optimal_3d_idx]
    optimal_conc_3d = C_mesh[optimal_3d_idx]
    optimal_effect_3d = Z_effect_3d[optimal_3d_idx]
    
    ax3d.scatter([optimal_temp_3d], [optimal_conc_3d], [optimal_effect_3d], 
                color='red', s=100, marker='o')
    
    ax3d.set_xlabel('Temperature (°C)')
    ax3d.set_ylabel(f'{cpa_type} Concentration (% v/v)')
    ax3d.set_zlabel('Effectiveness (%)')
    ax3d.set_title(f'Effectiveness as a Function of Temperature and Concentration (t={optimal_time:.1f} min)')
    
    fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
    
    st.pyplot(fig3d)
    
    # Display optimal values
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimal Temperature", f"{optimal_temp:.1f}°C")
    with col2:
        st.metric("Optimal Concentration", f"{optimal_conc:.1f}% v/v")
    with col3:
        st.metric("Optimal Exposure Time", f"{optimal_time:.1f} min")

with tab3:
    st.subheader("Cryoprotectant Comparison")
    
    # Define CPA properties
    cpas = {
        "DMSO": {"toxicity": 1.0, "permeability": 1.0, "max_conc": 15.0, "color": "blue"},
        "Glycerol": {"toxicity": 0.7, "permeability": 0.6, "max_conc": 20.0, "color": "green"},
        "Ethylene Glycol": {"toxicity": 0.8, "permeability": 1.2, "max_conc": 15.0, "color": "red"},
        "Propylene Glycol": {"toxicity": 0.9, "permeability": 0.8, "max_conc": 12.0, "color": "purple"},
        "Methanol": {"toxicity": 0.6, "permeability": 1.5, "max_conc": 10.0, "color": "orange"}
    }
    
    # Calculate effectiveness for each CPA
    effectiveness_by_cpa = {}
    optimal_params_by_cpa = {}
    
    for cpa_name, properties in cpas.items():
        # Create 2D grid for concentration and time
        cpa_concentrations = np.linspace(0, properties["max_conc"] * 1.5, 30)
        cpa_times = np.linspace(0, max_exposure_time, 30)
        C_cpa, T_cpa = np.meshgrid(cpa_concentrations, cpa_times)
        
        # Calculate effectiveness for each point
        Z_effective_cpa = np.zeros_like(C_cpa)
        
        for i in range(len(cpa_times)):
            for j in range(len(cpa_concentrations)):
                tox = calculate_toxicity(C_cpa[i, j], T_cpa[i, j], temperature, properties["toxicity"])
                surv = calculate_survival(tox, toxicity_tolerance)
                equil = calculate_equilibration(T_cpa[i, j], properties["permeability"], temperature)
                Z_effective_cpa[i, j] = (equil / 100) * (surv / 100) * 100
        
        # Find optimal point
        optimal_idx_cpa = np.unravel_index(np.argmax(Z_effective_cpa), Z_effective_cpa.shape)
        optimal_conc_cpa = C_cpa[optimal_idx_cpa]
        optimal_time_cpa = T_cpa[optimal_idx_cpa]
        optimal_effectiveness_cpa = Z_effective_cpa[optimal_idx_cpa]
        
        # Store results
        effectiveness_by_cpa[cpa_name] = {
            "concentrations": cpa_concentrations,
            "effectiveness": [Z_effective_cpa[np.argmin(np.abs(cpa_times - optimal_time_cpa)), j] 
                             for j in range(len(cpa_concentrations))]
        }
        
        optimal_params_by_cpa[cpa_name] = {
            "concentration": optimal_conc_cpa,
            "time": optimal_time_cpa,
            "effectiveness": optimal_effectiveness_cpa
        }
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cpa_name, data in effectiveness_by_cpa.items():
        ax.plot(data["concentrations"], data["effectiveness"], 
               label=cpa_name, color=cpas[cpa_name]["color"], linewidth=2)
        
        # Mark optimal point
        opt = optimal_params_by_cpa[cpa_name]
        ax.plot(opt["concentration"], opt["effectiveness"], 'o', 
               color=cpas[cpa_name]["color"], markersize=8)
    
    ax.set_xlabel('Concentration (% v/v)')
    ax.set_ylabel('Effectiveness (%)')
    ax.set_title(f'CPA Effectiveness Comparison (T={temperature}°C, t={optimal_time:.1f} min)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
    
    # Create comparison table
    comparison_data = {
        "Cryoprotectant": [],
        "Optimal Concentration (% v/v)": [],
        "Optimal Exposure Time (min)": [],
        "Effectiveness (%)": [],
        "Relative Toxicity": [],
        "Relative Permeability": []
    }
    
    for cpa_name, properties in cpas.items():
        opt = optimal_params_by_cpa[cpa_name]
        comparison_data["Cryoprotectant"].append(cpa_name)
        comparison_data["Optimal Concentration (% v/v)"].append(f"{opt['concentration']:.1f}")
        comparison_data["Optimal Exposure Time (min)"].append(f"{opt['time']:.1f}")
        comparison_data["Effectiveness (%)"].append(f"{opt['effectiveness']:.1f}")
        comparison_data["Relative Toxicity"].append(f"{properties['toxicity']:.1f}")
        comparison_data["Relative Permeability"].append(f"{properties['permeability']:.1f}")
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    # CPA mixtures analysis
    st.subheader("Cryoprotectant Mixtures Analysis")
    
    st.markdown("""
    Cryoprotectant mixtures often provide better protection than single CPAs by:
    
    1. **Reducing toxicity** through lower concentrations of each component
    2. **Combining different protective mechanisms** (colligative, non-colligative)
    3. **Balancing permeability** with both fast and slow penetrating CPAs
    
    Common effective mixtures include:
    """)
    
    # Define common mixtures
    mixtures = {
        "DMSO + Ethylene Glycol": {"components": ["DMSO", "Ethylene Glycol"], "ratio": "1:1", "total_conc": "15%", "effectiveness": 85},
        "Glycerol + Propylene Glycol": {"components": ["Glycerol", "Propylene Glycol"], "ratio": "3:1", "total_conc": "16%", "effectiveness": 78},
        "DMSO + Trehalose": {"components": ["DMSO", "Trehalose"], "ratio": "10% + 0.5M", "total_conc": "10% + 0.5M", "effectiveness": 82},
        "Ethylene Glycol + Sucrose": {"components": ["Ethylene Glycol", "Sucrose"], "ratio": "15% + 0.5M", "total_conc": "15% + 0.5M", "effectiveness": 80},
        "DMSO + Glycerol": {"components": ["DMSO", "Glycerol"], "ratio": "1:2", "total_conc": "15%", "effectiveness": 75}
    }
    
    # Create mixtures table
    mixtures_data = {
        "Mixture": [],
        "Components": [],
        "Ratio": [],
        "Total Concentration": [],
        "Estimated Effectiveness (%)": []
    }
    
    for mix_name, properties in mixtures.items():
        mixtures_data["Mixture"].append(mix_name)
        mixtures_data["Components"].append(", ".join(properties["components"]))
        mixtures_data["Ratio"].append(properties["ratio"])
        mixtures_data["Total Concentration"].append(properties["total_conc"])
        mixtures_data["Estimated Effectiveness (%)"].append(properties["effectiveness"])
    
    mixtures_df = pd.DataFrame(mixtures_data)
    st.table(mixtures_df)
    
    # If secondary CPA is selected, show mixture analysis
    if use_secondary_cpa:
        st.subheader(f"Analysis of {cpa_type} + {secondary_cpa_type} Mixture")
        
        # Create visualization of mixture effectiveness
        primary_range = np.linspace(0, default_max_concentration, 20)
        secondary_range = np.linspace(0, 20, 20)
        
        P, S = np.meshgrid(primary_range, secondary_range)
        Z_mix = np.zeros_like(P)
        
        # Simplified model for mixture effectiveness
        for i in range(len(secondary_range)):
            for j in range(len(primary_range)):
                # Calculate toxicity (assume some synergistic effect)
                primary_tox = calculate_toxicity(P[i, j], optimal_time, temperature, toxicity_factor)
                
                # Secondary CPAs often have different toxicity profiles
                if secondary_cpa_type in ["Trehalose", "Sucrose"]:
                    # Non-penetrating CPAs have lower direct toxicity
                    secondary_tox = calculate_toxicity(S[i, j] * 0.3, optimal_time, temperature, toxicity_factor * 0.5)
                else:
                    secondary_tox = calculate_toxicity(S[i, j], optimal_time, temperature, 
                                                     cpas.get(secondary_cpa_type, {"toxicity": 0.8})["toxicity"])
                
                # Combined toxicity (not purely additive due to potential interactions)
                combined_tox = primary_tox + secondary_tox * 0.8
                
                # Calculate survival
                survival = calculate_survival(combined_tox, toxicity_tolerance)
                
                # Calculate equilibration (primary CPA)
                equilibration = calculate_equilibration(optimal_time, permeability, temperature)
                
                # Calculate effectiveness
                Z_mix[i, j] = (equilibration / 100) * (survival / 100) * 100
                
                # Add bonus for certain combinations known to work well
                if (cpa_type == "DMSO" and secondary_cpa_type in ["Trehalose", "Sucrose"]) or \
                   (cpa_type == "Ethylene Glycol" and secondary_cpa_type in ["Trehalose", "Sucrose"]):
                    # Non-penetrating sugars work well with these CPAs
                    Z_mix[i, j] *= 1.1
                elif (cpa_type == "Glycerol" and secondary_cpa_type == "DMSO") or \
                     (cpa_type == "DMSO" and secondary_cpa_type == "Ethylene Glycol"):
                    # These combinations have synergistic effects
                    Z_mix[i, j] *= 1.05
        
        # Find optimal mixture
        optimal_mix_idx = np.unravel_index(np.argmax(Z_mix), Z_mix.shape)
        optimal_primary = P[optimal_mix_idx]
        optimal_secondary = S[optimal_mix_idx]
        optimal_mix_effectiveness = Z_mix[optimal_mix_idx]
        
        # Create contour plot
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(P, S, Z_mix, 20, cmap='viridis')
        ax.set_xlabel(f'{cpa_type} Concentration (% v/v)')
        ax.set_ylabel(f'{secondary_cpa_type} Concentration (% v/v)')
        ax.set_title(f'Effectiveness of {cpa_type} + {secondary_cpa_type} Mixture')
        
        # Mark optimal point and current selection
        ax.plot(optimal_primary, optimal_secondary, 'ro', markersize=10)
        ax.annotate(f'Optimal: {optimal_mix_effectiveness:.1f}%', 
                   (optimal_primary, optimal_secondary), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   color='white',
                   fontweight='bold')
        
        # Mark current selection
        current_effectiveness = Z_mix[np.argmin(np.abs(secondary_range - secondary_cpa_concentration)), 
                                    np.argmin(np.abs(primary_range - optimal_conc))]
        ax.plot(optimal_conc, secondary_cpa_concentration, 'mo', markersize=8)
        ax.annotate(f'Current: {current_effectiveness:.1f}%', 
                   (optimal_conc, secondary_cpa_concentration), 
                   xytext=(10, -15), 
                   textcoords='offset points',
                   color='white')
        
        fig.colorbar(contour, ax=ax, label='Effectiveness (%)')
        
        st.pyplot(fig)
        
        # Display optimal mixture values
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Optimal {cpa_type}", f"{optimal_primary:.1f}% v/v")
        with col2:
            st.metric(f"Optimal {secondary_cpa_type}", f"{optimal_secondary:.1f}% v/v")
        with col3:
            st.metric("Mixture Effectiveness", f"{optimal_mix_effectiveness:.1f}%", 
                     delta=f"{optimal_mix_effectiveness - optimal_effectiveness:.1f}%")

# Explanation section
st.markdown("""
## Cryoprotectant Toxicity and Optimization

### Types of Cryoprotectants

1. **Penetrating CPAs**:
   - Small molecules that permeate cell membranes
   - Examples: DMSO, glycerol, ethylene glycol, propylene glycol, methanol
   - Mechanism: Prevent intracellular ice formation, reduce solution effects

2. **Non-penetrating CPAs**:
   - Larger molecules that remain extracellular
   - Examples: Sugars (trehalose, sucrose), polymers (hydroxyethyl starch)
   - Mechanism: Osmotic dehydration, membrane stabilization

### Toxicity Mechanisms

Cryoprotectants can cause damage through:

1. **Chemical toxicity**: Direct biochemical interactions with cellular components
2. **Osmotic stress**: Extreme volume changes during addition/removal
3. **Protein denaturation**: Disruption of protein structure and function
4. **Membrane destabilization**: Alteration of membrane fluidity and integrity

### Optimization Strategies

1. **Concentration optimization**: Balance between cryoprotection and toxicity
2. **Exposure time optimization**: Minimize exposure while ensuring equilibration
3. **Temperature control**: Lower temperatures reduce chemical toxicity but slow equilibration
4. **CPA mixtures**: Combine CPAs to reduce individual concentrations and toxicity
5. **Stepwise addition/removal**: Minimize osmotic stress during CPA loading/unloading

### Cell Type Considerations

Different cell types have varying:
- Membrane composition and permeability
- Sensitivity to specific CPAs
- Osmotic tolerance
- Metabolic responses to CPAs

These differences necessitate customized CPA protocols for each cell type.
""")
