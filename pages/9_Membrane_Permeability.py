import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

st.set_page_config(
    page_title="Membrane Permeability Modeler - Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide"
)

st.title("Membrane Permeability Modeler")

st.markdown("""
This tool calculates membrane permeability to water and cryoprotectants at different temperatures.
Membrane permeability is a critical parameter that determines how quickly water and cryoprotective
agents (CPAs) can move across cell membranes during freezing and thawing.

The model calculates:
- Water permeability (Lp) as a function of temperature
- CPA permeability (Ps) as a function of temperature
- Predicted cell volume changes during CPA addition and removal
""")

# Create sidebar for parameters
st.sidebar.header("Cell Parameters")

cell_type = st.sidebar.selectbox(
    "Cell Type",
    ["Red Blood Cell", "Oocyte", "Sperm", "Embryo", "Stem Cell", "Fibroblast", "Custom"]
)

# Default parameters based on cell type
if cell_type == "Red Blood Cell":
    default_lp_reference = 2.5
    default_ea_water = 55.0
    default_cell_diameter = 8.0
    default_inactive_volume = 0.4
elif cell_type == "Oocyte":
    default_lp_reference = 0.8
    default_ea_water = 65.0
    default_cell_diameter = 120.0
    default_inactive_volume = 0.3
elif cell_type == "Sperm":
    default_lp_reference = 2.2
    default_ea_water = 48.0
    default_cell_diameter = 5.0
    default_inactive_volume = 0.6
elif cell_type == "Embryo":
    default_lp_reference = 1.0
    default_ea_water = 60.0
    default_cell_diameter = 80.0
    default_inactive_volume = 0.3
elif cell_type == "Stem Cell":
    default_lp_reference = 1.5
    default_ea_water = 58.0
    default_cell_diameter = 15.0
    default_inactive_volume = 0.35
elif cell_type == "Fibroblast":
    default_lp_reference = 1.8
    default_ea_water = 52.0
    default_cell_diameter = 20.0
    default_inactive_volume = 0.3
else:  # Custom
    default_lp_reference = 1.5
    default_ea_water = 55.0
    default_cell_diameter = 20.0
    default_inactive_volume = 0.3

# Water permeability parameters
lp_reference = st.sidebar.slider(
    "Reference Water Permeability (μm/min/atm)",
    min_value=0.1,
    max_value=5.0,
    value=default_lp_reference,
    step=0.1,
    help="Hydraulic conductivity at reference temperature (usually 20-25°C)"
)

ea_water = st.sidebar.slider(
    "Activation Energy for Water (kJ/mol)",
    min_value=10.0,
    max_value=80.0,
    value=default_ea_water,
    step=1.0,
    help="Energy barrier for water transport across membrane"
)

# Cell geometry parameters
cell_diameter = st.sidebar.slider(
    "Cell Diameter (μm)",
    min_value=1.0,
    max_value=200.0,
    value=default_cell_diameter,
    step=1.0
)

inactive_volume_ratio = st.sidebar.slider(
    "Inactive Volume Ratio",
    min_value=0.1,
    max_value=0.8,
    value=default_inactive_volume,
    step=0.05,
    help="Fraction of cell volume that is osmotically inactive"
)

# CPA parameters
st.sidebar.header("Cryoprotectant Parameters")

cpa_type = st.sidebar.selectbox(
    "Cryoprotectant",
    ["DMSO", "Glycerol", "Ethylene Glycol", "Propylene Glycol", "Methanol", "Custom"]
)

# Default parameters based on CPA type
if cpa_type == "DMSO":
    default_ps_reference = 1.0
    default_ea_cpa = 60.0
    default_reflection_coef = 0.8
elif cpa_type == "Glycerol":
    default_ps_reference = 0.3
    default_ea_cpa = 65.0
    default_reflection_coef = 0.9
elif cpa_type == "Ethylene Glycol":
    default_ps_reference = 1.2
    default_ea_cpa = 55.0
    default_reflection_coef = 0.7
elif cpa_type == "Propylene Glycol":
    default_ps_reference = 0.8
    default_ea_cpa = 58.0
    default_reflection_coef = 0.8
elif cpa_type == "Methanol":
    default_ps_reference = 2.0
    default_ea_cpa = 50.0
    default_reflection_coef = 0.5
else:  # Custom
    default_ps_reference = 1.0
    default_ea_cpa = 60.0
    default_reflection_coef = 0.8

ps_reference = st.sidebar.slider(
    "Reference CPA Permeability (×10⁻³ cm/min)",
    min_value=0.1,
    max_value=3.0,
    value=default_ps_reference,
    step=0.1,
    help="Solute permeability at reference temperature (usually 20-25°C)"
)

ea_cpa = st.sidebar.slider(
    "Activation Energy for CPA (kJ/mol)",
    min_value=10.0,
    max_value=80.0,
    value=default_ea_cpa,
    step=1.0,
    help="Energy barrier for CPA transport across membrane"
)

reflection_coefficient = st.sidebar.slider(
    "Reflection Coefficient",
    min_value=0.0,
    max_value=1.0,
    value=default_reflection_coef,
    step=0.05,
    help="Measure of membrane selectivity (1 = impermeable to solute, 0 = freely permeable)"
)

# Define permeability model functions
def calculate_water_permeability(temperature, lp_ref, ea, reference_temp=293.15):
    """Calculate water permeability at a given temperature using Arrhenius equation."""
    # Convert temperature to Kelvin
    temp_k = temperature + 273.15
    
    # Gas constant (8.314 J/mol·K)
    R = 8.314
    
    # Arrhenius equation: Lp(T) = Lp(Tref) * exp[Ea/R * (1/Tref - 1/T)]
    lp = lp_ref * np.exp((ea * 1000 / R) * (1/reference_temp - 1/temp_k))
    
    return lp

def calculate_cpa_permeability(temperature, ps_ref, ea, reference_temp=293.15):
    """Calculate CPA permeability at a given temperature using Arrhenius equation."""
    # Convert temperature to Kelvin
    temp_k = temperature + 273.15
    
    # Gas constant (8.314 J/mol·K)
    R = 8.314
    
    # Arrhenius equation: Ps(T) = Ps(Tref) * exp[Ea/R * (1/Tref - 1/T)]
    ps = ps_ref * np.exp((ea * 1000 / R) * (1/reference_temp - 1/temp_k))
    
    return ps

def calculate_cell_volume_change(time_points, temperature, lp, ps, sigma, 
                                initial_volume, inactive_volume, 
                                initial_osmolarity, final_osmolarity, 
                                initial_cpa, final_cpa):
    """Calculate cell volume changes during CPA addition or removal."""
    # Constants
    R = 0.08206  # Gas constant (L·atm/mol·K)
    T = temperature + 273.15  # Convert to Kelvin
    
    # Initial conditions
    current_volume = initial_volume
    current_osmolarity = initial_osmolarity
    current_cpa = initial_cpa
    
    # Calculate initial osmotically active volume
    initial_active_volume = initial_volume * (1 - inactive_volume_ratio)
    
    # Arrays to store results
    volumes = [current_volume]
    osmolarities = [current_osmolarity]
    cpa_concentrations = [current_cpa]
    
    # Time step
    dt = time_points[1] - time_points[0]
    
    # Simulate volume changes
    for t in range(1, len(time_points)):
        # Calculate current osmotically active volume
        active_volume = current_volume - (initial_volume * inactive_volume_ratio)
        
        # Calculate water flux
        osmotic_pressure = R * T * (current_osmolarity - final_osmolarity + 
                                   sigma * (current_cpa - final_cpa))
        water_flux = -lp * cell_surface_area * osmotic_pressure
        
        # Calculate CPA flux
        cpa_flux = ps * cell_surface_area * (final_cpa - current_cpa)
        
        # Update volume and concentrations
        new_volume = current_volume + water_flux * dt
        
        # Ensure volume doesn't go below inactive volume
        if new_volume < initial_volume * inactive_volume_ratio:
            new_volume = initial_volume * inactive_volume_ratio
            active_volume = 0
        else:
            active_volume = new_volume - (initial_volume * inactive_volume_ratio)
        
        # Update CPA amount and concentration
        new_cpa_amount = current_cpa * active_volume + cpa_flux * dt
        new_cpa = new_cpa_amount / active_volume if active_volume > 0 else current_cpa
        
        # Update osmolarity (assuming conservation of solutes)
        new_osmolarity = initial_osmolarity * initial_active_volume / active_volume if active_volume > 0 else current_osmolarity
        
        # Store results
        volumes.append(new_volume)
        osmolarities.append(new_osmolarity)
        cpa_concentrations.append(new_cpa)
        
        # Update current values
        current_volume = new_volume
        current_osmolarity = new_osmolarity
        current_cpa = new_cpa
    
    return volumes, osmolarities, cpa_concentrations

# Calculate cell surface area
cell_radius = cell_diameter / 2
cell_volume = (4/3) * np.pi * (cell_radius ** 3)
cell_surface_area = 4 * np.pi * (cell_radius ** 2)

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Temperature Effects", "CPA Addition/Removal", "Permeability Comparison"])

with tab1:
    st.subheader("Temperature Effects on Membrane Permeability")
    
    # Create temperature range
    temperatures = np.linspace(-10, 40, 100)
    
    # Calculate permeabilities
    water_permeabilities = [calculate_water_permeability(t, lp_reference, ea_water) for t in temperatures]
    cpa_permeabilities = [calculate_cpa_permeability(t, ps_reference, ea_cpa) for t in temperatures]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot water permeability
    ax1.plot(temperatures, water_permeabilities, 'b-', linewidth=2)
    ax1.set_ylabel('Water Permeability (μm/min/atm)')
    ax1.set_title('Water Permeability vs. Temperature')
    ax1.grid(True, alpha=0.3)
    
    # Add reference point
    ax1.plot(20, lp_reference, 'ro', markersize=8)
    ax1.annotate(f'Reference: {lp_reference} μm/min/atm at 20°C', 
                (20, lp_reference), 
                xytext=(10, 10), 
                textcoords='offset points')
    
    # Plot CPA permeability
    ax2.plot(temperatures, cpa_permeabilities, 'g-', linewidth=2)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel(f'{cpa_type} Permeability (×10⁻³ cm/min)')
    ax2.set_title(f'{cpa_type} Permeability vs. Temperature')
    ax2.grid(True, alpha=0.3)
    
    # Add reference point
    ax2.plot(20, ps_reference, 'ro', markersize=8)
    ax2.annotate(f'Reference: {ps_reference} ×10⁻³ cm/min at 20°C', 
                (20, ps_reference), 
                xytext=(10, 10), 
                textcoords='offset points')
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Calculate permeability ratios
    st.subheader("Permeability Ratios at Different Temperatures")
    
    # Select specific temperatures
    specific_temps = [-10, 0, 10, 20, 30, 40]
    
    # Create data for table
    temp_data = {
        "Temperature (°C)": specific_temps,
        "Water Permeability (μm/min/atm)": [calculate_water_permeability(t, lp_reference, ea_water) for t in specific_temps],
        f"{cpa_type} Permeability (×10⁻³ cm/min)": [calculate_cpa_permeability(t, ps_reference, ea_cpa) for t in specific_temps],
        "Relative Water Permeability": [calculate_water_permeability(t, lp_reference, ea_water) / lp_reference for t in specific_temps],
        f"Relative {cpa_type} Permeability": [calculate_cpa_permeability(t, ps_reference, ea_cpa) / ps_reference for t in specific_temps]
    }
    
    # Create dataframe
    temp_df = pd.DataFrame(temp_data)
    
    # Format numbers
    temp_df["Water Permeability (μm/min/atm)"] = temp_df["Water Permeability (μm/min/atm)"].map('{:.2f}'.format)
    temp_df[f"{cpa_type} Permeability (×10⁻³ cm/min)"] = temp_df[f"{cpa_type} Permeability (×10⁻³ cm/min)"].map('{:.2f}'.format)
    temp_df["Relative Water Permeability"] = temp_df["Relative Water Permeability"].map('{:.2f}'.format)
    temp_df[f"Relative {cpa_type} Permeability"] = temp_df[f"Relative {cpa_type} Permeability"].map('{:.2f}'.format)
    
    st.table(temp_df)
    
    # Create 3D visualization of permeability vs. temperature and activation energy
    st.subheader("3D Visualization: Permeability vs. Temperature and Activation Energy")
    
    # Create parameter ranges
    temp_range = np.linspace(-10, 40, 20)
    ea_range = np.linspace(20, 80, 20)
    
    T, E = np.meshgrid(temp_range, ea_range)
    Z = np.zeros_like(T)
    
    for i in range(len(ea_range)):
        for j in range(len(temp_range)):
            Z[i, j] = calculate_water_permeability(T[i, j], lp_reference, E[i, j])
    
    # Create 3D plot
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    surf = ax3d.plot_surface(T, E, Z, cmap=cm.viridis, alpha=0.8)
    
    # Mark current parameters
    ax3d.scatter([20], [ea_water], [lp_reference], color='red', s=100, marker='o')
    
    ax3d.set_xlabel('Temperature (°C)')
    ax3d.set_ylabel('Activation Energy (kJ/mol)')
    ax3d.set_zlabel('Water Permeability (μm/min/atm)')
    ax3d.set_title('Water Permeability as a Function of Temperature and Activation Energy')
    
    fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
    
    st.pyplot(fig3d)

with tab2:
    st.subheader("Cell Volume Changes During CPA Addition and Removal")
    
    # Simulation parameters
    simulation_time = st.slider(
        "Simulation Time (min)",
        min_value=1,
        max_value=30,
        value=10,
        step=1
    )
    
    simulation_temp = st.slider(
        "Temperature (°C)",
        min_value=-10,
        max_value=40,
        value=22,
        step=1
    )
    
    # Calculate permeabilities at simulation temperature
    sim_lp = calculate_water_permeability(simulation_temp, lp_reference, ea_water)
    sim_ps = calculate_cpa_permeability(simulation_temp, ps_reference, ea_cpa)
    
    # CPA addition parameters
    initial_osmolarity = 300  # mOsm, isotonic
    final_osmolarity = 300    # mOsm, isotonic (CPA solution is typically prepared isotonic)
    
    initial_cpa = 0.0         # M, no CPA initially
    final_cpa = st.slider(
        "Final CPA Concentration (M)",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1
    )
    
    # Create time points
    time_points = np.linspace(0, simulation_time, 100)
    
    # Calculate volume changes during CPA addition
    addition_volumes, addition_osmolarities, addition_cpa = calculate_cell_volume_change(
        time_points, 
        simulation_temp, 
        sim_lp, 
        sim_ps, 
        reflection_coefficient, 
        cell_volume, 
        inactive_volume_ratio, 
        initial_osmolarity, 
        final_osmolarity, 
        initial_cpa, 
        final_cpa
    )
    
    # Calculate volume changes during CPA removal
    removal_volumes, removal_osmolarities, removal_cpa = calculate_cell_volume_change(
        time_points, 
        simulation_temp, 
        sim_lp, 
        sim_ps, 
        reflection_coefficient, 
        addition_volumes[-1],  # Start from final addition volume
        inactive_volume_ratio, 
        addition_osmolarities[-1],  # Start from final addition osmolarity
        initial_osmolarity,  # Return to initial osmolarity
        addition_cpa[-1],  # Start from final CPA concentration
        initial_cpa  # Return to initial CPA concentration
    )
    
    # Normalize volumes
    normalized_addition_volumes = [v / cell_volume for v in addition_volumes]
    normalized_removal_volumes = [v / cell_volume for v in removal_volumes]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot volume changes during addition
    ax1.plot(time_points, normalized_addition_volumes, 'b-', linewidth=2, label='Cell Volume')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Initial Volume')
    ax1.axhline(y=inactive_volume_ratio, color='red', linestyle='--', alpha=0.7, label='Inactive Volume')
    
    # Add critical volume lines
    ax1.axhline(y=1.4, color='orange', linestyle=':', alpha=0.7, label='Critical Swelling (140%)')
    ax1.axhline(y=0.6, color='orange', linestyle=':', alpha=0.7, label='Critical Shrinkage (60%)')
    
    ax1.set_ylabel('Normalized Volume (V/V₀)')
    ax1.set_title(f'Cell Volume During {cpa_type} Addition ({final_cpa} M)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0.4, 1.6)
    
    # Create a secondary y-axis for CPA concentration
    ax1b = ax1.twinx()
    ax1b.plot(time_points, addition_cpa, 'g--', linewidth=1.5, label='CPA Concentration')
    ax1b.set_ylabel('CPA Concentration (M)', color='g')
    ax1b.tick_params(axis='y', labelcolor='g')
    
    # Plot volume changes during removal
    ax2.plot(time_points, normalized_removal_volumes, 'b-', linewidth=2, label='Cell Volume')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Initial Volume')
    ax2.axhline(y=inactive_volume_ratio, color='red', linestyle='--', alpha=0.7, label='Inactive Volume')
    
    # Add critical volume lines
    ax2.axhline(y=1.4, color='orange', linestyle=':', alpha=0.7, label='Critical Swelling (140%)')
    ax2.axhline(y=0.6, color='orange', linestyle=':', alpha=0.7, label='Critical Shrinkage (60%)')
    
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Normalized Volume (V/V₀)')
    ax2.set_title(f'Cell Volume During {cpa_type} Removal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0.4, 1.6)
    
    # Create a secondary y-axis for CPA concentration
    ax2b = ax2.twinx()
    ax2b.plot(time_points, removal_cpa, 'g--', linewidth=1.5, label='CPA Concentration')
    ax2b.set_ylabel('CPA Concentration (M)', color='g')
    ax2b.tick_params(axis='y', labelcolor='g')
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Calculate key metrics
    min_addition_volume = min(normalized_addition_volumes)
    max_addition_volume = max(normalized_addition_volumes)
    
    min_removal_volume = min(normalized_removal_volumes)
    max_removal_volume = max(normalized_removal_volumes)
    
    # Time to equilibration (95% of final value)
    addition_equilibration_time = None
    target_addition_cpa = 0.95 * final_cpa
    for i, cpa_conc in enumerate(addition_cpa):
        if cpa_conc >= target_addition_cpa:
            addition_equilibration_time = time_points[i]
            break
    
    removal_equilibration_time = None
    target_removal_cpa = 0.05 * final_cpa
    for i, cpa_conc in enumerate(removal_cpa):
        if cpa_conc <= target_removal_cpa:
            removal_equilibration_time = time_points[i]
            break
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Minimum Volume During Addition", f"{min_addition_volume:.2f} × V₀")
        st.metric("Maximum Volume During Removal", f"{max_removal_volume:.2f} × V₀")
    
    with col2:
        st.metric("Maximum Volume During Addition", f"{max_addition_volume:.2f} × V₀")
        st.metric("Minimum Volume During Removal", f"{min_removal_volume:.2f} × V₀")
    
    with col3:
        if addition_equilibration_time:
            st.metric("Time to 95% Equilibration (Addition)", f"{addition_equilibration_time:.2f} min")
        else:
            st.metric("Time to 95% Equilibration (Addition)", "N/A")
        
        if removal_equilibration_time:
            st.metric("Time to 95% Equilibration (Removal)", f"{removal_equilibration_time:.2f} min")
        else:
            st.metric("Time to 95% Equilibration (Removal)", "N/A")
    
    # Check for potential osmotic damage
    if min_addition_volume < 0.6 or min_removal_volume < 0.6:
        st.warning("⚠️ Cell volume drops below critical shrinkage threshold (60% of initial volume). This may cause membrane damage.")
    
    if max_addition_volume > 1.4 or max_removal_volume > 1.4:
        st.warning("⚠️ Cell volume exceeds critical swelling threshold (140% of initial volume). This may cause lysis.")
    
    # Create multi-step CPA addition/removal simulation
    st.subheader("Multi-Step CPA Addition/Removal Protocol")
    
    num_steps = st.slider(
        "Number of Steps",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )
    
    # Create multi-step protocol
    step_times = np.linspace(0, simulation_time, num_steps + 1)[1:]
    step_cpa_levels = np.linspace(0, final_cpa, num_steps + 1)[1:]
    
    # Run multi-step simulation
    multi_step_volumes = []
    multi_step_cpa = []
    
    current_volume = cell_volume
    current_osmolarity = initial_osmolarity
    current_cpa = initial_cpa
    
    for i in range(num_steps):
        # Calculate time points for this step
        if i == 0:
            step_start_time = 0
        else:
            step_start_time = step_times[i-1]
        
        step_end_time = step_times[i]
        step_time_points = np.linspace(step_start_time, step_end_time, 50)
        
        # Calculate target CPA for this step
        target_cpa = step_cpa_levels[i]
        
        # Calculate volume changes for this step
        step_volumes, step_osmolarities, step_cpa_conc = calculate_cell_volume_change(
            step_time_points, 
            simulation_temp, 
            sim_lp, 
            sim_ps, 
            reflection_coefficient, 
            current_volume, 
            inactive_volume_ratio, 
            current_osmolarity, 
            final_osmolarity,  # Keep osmolarity constant
            current_cpa, 
            target_cpa
        )
        
        # Store results
        if i == 0:
            multi_step_volumes.extend(step_volumes)
            multi_step_cpa.extend(step_cpa_conc)
        else:
            # Skip the first point to avoid duplication
            multi_step_volumes.extend(step_volumes[1:])
            multi_step_cpa.extend(step_cpa_conc[1:])
        
        # Update current values for next step
        current_volume = step_volumes[-1]
        current_osmolarity = step_osmolarities[-1]
        current_cpa = step_cpa_conc[-1]
    
    # Create time points for the entire protocol
    multi_step_time_points = np.linspace(0, simulation_time, len(multi_step_volumes))
    
    # Normalize volumes
    normalized_multi_step_volumes = [v / cell_volume for v in multi_step_volumes]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot volume changes
    ax.plot(multi_step_time_points, normalized_multi_step_volumes, 'b-', linewidth=2, label='Cell Volume')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Initial Volume')
    ax.axhline(y=inactive_volume_ratio, color='red', linestyle='--', alpha=0.7, label='Inactive Volume')
    
    # Add critical volume lines
    ax.axhline(y=1.4, color='orange', linestyle=':', alpha=0.7, label='Critical Swelling (140%)')
    ax.axhline(y=0.6, color='orange', linestyle=':', alpha=0.7, label='Critical Shrinkage (60%)')
    
    # Add step markers
    for time in step_times:
        ax.axvline(x=time, color='green', linestyle='-', alpha=0.5)
    
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Normalized Volume (V/V₀)')
    ax.set_title(f'{num_steps}-Step {cpa_type} Addition Protocol')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0.4, 1.6)
    
    # Create a secondary y-axis for CPA concentration
    ax2 = ax.twinx()
    ax2.plot(multi_step_time_points, multi_step_cpa, 'g--', linewidth=1.5, label='CPA Concentration')
    ax2.set_ylabel('CPA Concentration (M)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Add step markers
    for i, time in enumerate(step_times):
        ax2.annotate(f'Step {i+1}', 
                    (time, multi_step_cpa[np.argmin(np.abs(multi_step_time_points - time))]), 
                    xytext=(-30, 10), 
                    textcoords='offset points',
                    color='green')
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Calculate key metrics for multi-step protocol
    min_multi_step_volume = min(normalized_multi_step_volumes)
    max_multi_step_volume = max(normalized_multi_step_volumes)
    
    # Compare with single-step protocol
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Minimum Volume (Multi-Step)", f"{min_multi_step_volume:.2f} × V₀", 
                 delta=f"{min_multi_step_volume - min_addition_volume:.2f}")
    
    with col2:
        st.metric("Maximum Volume (Multi-Step)", f"{max_multi_step_volume:.2f} × V₀", 
                 delta=f"{max_multi_step_volume - max_addition_volume:.2f}")
    
    # Create protocol table
    protocol_data = {
        "Step": list(range(1, num_steps + 1)),
        "Time (min)": [f"{t:.1f}" for t in step_times],
        f"{cpa_type} Concentration (M)": [f"{c:.1f}" for c in step_cpa_levels]
    }
    
    protocol_df = pd.DataFrame(protocol_data)
    st.table(protocol_df)

with tab3:
    st.subheader("Permeability Comparison Across Cell Types and CPAs")
    
    # Define cell types and their properties
    cell_types = {
        "Red Blood Cell": {"lp": 2.5, "ea_water": 55.0, "diameter": 8.0},
        "Oocyte": {"lp": 0.8, "ea_water": 65.0, "diameter": 120.0},
        "Sperm": {"lp": 2.2, "ea_water": 48.0, "diameter": 5.0},
        "Embryo": {"lp": 1.0, "ea_water": 60.0, "diameter": 80.0},
        "Stem Cell": {"lp": 1.5, "ea_water": 58.0, "diameter": 15.0},
        "Fibroblast": {"lp": 1.8, "ea_water": 52.0, "diameter": 20.0}
    }
    
    # Define CPAs and their properties
    cpas = {
        "DMSO": {"ps": 1.0, "ea_cpa": 60.0, "sigma": 0.8},
        "Glycerol": {"ps": 0.3, "ea_cpa": 65.0, "sigma": 0.9},
        "Ethylene Glycol": {"ps": 1.2, "ea_cpa": 55.0, "sigma": 0.7},
        "Propylene Glycol": {"ps": 0.8, "ea_cpa": 58.0, "sigma": 0.8},
        "Methanol": {"ps": 2.0, "ea_cpa": 50.0, "sigma": 0.5}
    }
    
    # Create comparison table for water permeability
    water_perm_data = {
        "Cell Type": list(cell_types.keys()),
        "Reference Lp (μm/min/atm)": [cell_types[c]["lp"] for c in cell_types],
        "Activation Energy (kJ/mol)": [cell_types[c]["ea_water"] for c in cell_types],
        "Lp at 20°C": [cell_types[c]["lp"] for c in cell_types],
        "Lp at 0°C": [calculate_water_permeability(0, cell_types[c]["lp"], cell_types[c]["ea_water"]) for c in cell_types],
        "Lp at -10°C": [calculate_water_permeability(-10, cell_types[c]["lp"], cell_types[c]["ea_water"]) for c in cell_types]
    }
    
    # Format numbers
    water_perm_df = pd.DataFrame(water_perm_data)
    water_perm_df["Lp at 20°C"] = water_perm_df["Lp at 20°C"].map('{:.2f}'.format)
    water_perm_df["Lp at 0°C"] = water_perm_df["Lp at 0°C"].map('{:.2f}'.format)
    water_perm_df["Lp at -10°C"] = water_perm_df["Lp at -10°C"].map('{:.2f}'.format)
    
    st.subheader("Water Permeability Comparison Across Cell Types")
    st.table(water_perm_df)
    
    # Create comparison table for CPA permeability
    cpa_perm_data = {
        "Cryoprotectant": list(cpas.keys()),
        "Reference Ps (×10⁻³ cm/min)": [cpas[c]["ps"] for c in cpas],
        "Activation Energy (kJ/mol)": [cpas[c]["ea_cpa"] for c in cpas],
        "Reflection Coefficient": [cpas[c]["sigma"] for c in cpas],
        "Ps at 20°C": [cpas[c]["ps"] for c in cpas],
        "Ps at 0°C": [calculate_cpa_permeability(0, cpas[c]["ps"], cpas[c]["ea_cpa"]) for c in cpas],
        "Ps at -10°C": [calculate_cpa_permeability(-10, cpas[c]["ps"], cpas[c]["ea_cpa"]) for c in cpas]
    }
    
    # Format numbers
    cpa_perm_df = pd.DataFrame(cpa_perm_data)
    cpa_perm_df["Ps at 20°C"] = cpa_perm_df["Ps at 20°C"].map('{:.2f}'.format)
    cpa_perm_df["Ps at 0°C"] = cpa_perm_df["Ps at 0°C"].map('{:.2f}'.format)
    cpa_perm_df["Ps at -10°C"] = cpa_perm_df["Ps at -10°C"].map('{:.2f}'.format)
    
    st.subheader("CPA Permeability Comparison")
    st.table(cpa_perm_df)
    
    # Create plot comparing water permeability across cell types
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Temperature range
    temps = np.linspace(-10, 40, 100)
    
    # Plot water permeability for each cell type
    for cell_name, properties in cell_types.items():
        lp_values = [calculate_water_permeability(t, properties["lp"], properties["ea_water"]) for t in temps]
        ax.plot(temps, lp_values, linewidth=2, label=cell_name)
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Water Permeability (μm/min/atm)')
    ax.set_title('Water Permeability vs. Temperature for Different Cell Types')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
    
    # Create plot comparing CPA permeability
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot CPA permeability for each CPA
    for cpa_name, properties in cpas.items():
        ps_values = [calculate_cpa_permeability(t, properties["ps"], properties["ea_cpa"]) for t in temps]
        ax.plot(temps, ps_values, linewidth=2, label=cpa_name)
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('CPA Permeability (×10⁻³ cm/min)')
    ax.set_title('CPA Permeability vs. Temperature for Different Cryoprotectants')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
    
    # Create equilibration time comparison
    st.subheader("CPA Equilibration Time Comparison")
    
    # Define standard conditions
    std_temp = 22  # °C
    std_volume = 4/3 * np.pi * (20/2)**3  # μm³, for a 20 μm diameter cell
    std_surface = 4 * np.pi * (20/2)**2  # μm², for a 20 μm diameter cell
    std_inactive = 0.3  # 30% inactive volume
    std_cpa_conc = 1.0  # M
    
    # Calculate equilibration times
    equilibration_data = {
        "Cryoprotectant": [],
        "Permeability at 22°C (×10⁻³ cm/min)": [],
        "Time to 95% Equilibration (min)": []
    }
    
    for cpa_name, properties in cpas.items():
        # Calculate permeability at standard temperature
        ps = calculate_cpa_permeability(std_temp, properties["ps"], properties["ea_cpa"])
        
        # Calculate water permeability at standard temperature
        lp = calculate_water_permeability(std_temp, lp_reference, ea_water)
        
        # Create time points
        eq_time_points = np.linspace(0, 30, 300)  # 30 minutes, 300 points
        
        # Calculate volume changes
        _, _, eq_cpa = calculate_cell_volume_change(
            eq_time_points, 
            std_temp, 
            lp, 
            ps, 
            properties["sigma"], 
            std_volume, 
            std_inactive, 
            300,  # mOsm
            300,  # mOsm
            0.0,  # Initial CPA
            std_cpa_conc  # Final CPA
        )
        
        # Find time to 95% equilibration
        eq_time = None
        target_eq_cpa = 0.95 * std_cpa_conc
        for i, cpa_conc in enumerate(eq_cpa):
            if cpa_conc >= target_eq_cpa:
                eq_time = eq_time_points[i]
                break
        
        # Store results
        equilibration_data["Cryoprotectant"].append(cpa_name)
        equilibration_data["Permeability at 22°C (×10⁻³ cm/min)"].append(ps)
        equilibration_data["Time to 95% Equilibration (min)"].append(eq_time if eq_time else ">30")
    
    # Format numbers
    eq_df = pd.DataFrame(equilibration_data)
    eq_df["Permeability at 22°C (×10⁻³ cm/min)"] = eq_df["Permeability at 22°C (×10⁻³ cm/min)"].map('{:.2f}'.format)
    
    # Sort by equilibration time
    eq_df = eq_df.sort_values("Time to 95% Equilibration (min)")
    
    st.table(eq_df)

# Explanation section
st.markdown("""
## Membrane Permeability Theory in Cryopreservation

### Membrane Transport Processes

During cryopreservation, two main transport processes occur across cell membranes:

1. **Water Transport**: Movement of water in response to osmotic gradients
   - Characterized by hydraulic conductivity (Lp)
   - Driven by differences in osmolarity and hydrostatic pressure

2. **CPA Transport**: Movement of cryoprotective agents
   - Characterized by solute permeability (Ps)
   - Driven by concentration gradients

### Temperature Effects on Permeability

Membrane permeability decreases with temperature according to the Arrhenius relationship:

$$P(T) = P(T_{ref}) \cdot \exp\left[\frac{E_a}{R} \cdot \left(\frac{1}{T_{ref}} - \frac{1}{T}\right)\right]$$

Where:
- $P(T)$ is permeability at temperature $T$
- $P(T_{ref})$ is permeability at reference temperature $T_{ref}$
- $E_a$ is activation energy
- $R$ is the gas constant
- $T$ is absolute temperature in Kelvin

### Key Parameters

1. **Hydraulic Conductivity (Lp)**: Measures water permeability
   - Units: μm/min/atm or μm³/μm²/min/atm
   - Higher values indicate faster water movement

2. **Solute Permeability (Ps)**: Measures CPA permeability
   - Units: cm/min or μm/min
   - Higher values indicate faster CPA movement

3. **Activation Energy (Ea)**: Energy barrier for transport
   - Units: kJ/mol
   - Higher values indicate stronger temperature dependence

4. **Reflection Coefficient (σ)**: Measure of membrane selectivity
   - Dimensionless (0 to 1)
   - 1 = completely impermeable to solute
   - 0 = freely permeable to solute

### Implications for Cryopreservation

1. **CPA Addition/Removal**:
   - Osmotic stress can damage cells if volume changes are too extreme
   - Multi-step protocols can minimize volume excursions
   - Temperature affects equilibration time

2. **Cooling Rate Optimization**:
   - Cells with higher water permeability can tolerate faster cooling
   - Activation energy determines how permeability changes with temperature
   - Optimal cooling rate depends on permeability at subzero temperatures

3. **Cell Type Differences**:
   - Different cell types have different membrane compositions
   - Permeability varies widely between cell types
   - Protocols must be customized for each cell type

4. **CPA Selection**:
   - Different CPAs have different permeability characteristics
   - Fast-penetrating CPAs (e.g., DMSO, methanol) equilibrate quickly
   - Slow-penetrating CPAs (e.g., glycerol) require longer equilibration times
""")
