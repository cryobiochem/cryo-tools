import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

st.set_page_config(
    page_title="Warming Protocol Designer - Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide"
)

st.title("Warming Protocol Designer")

st.markdown("""
This tool helps design optimal warming protocols for cryopreserved samples.
The warming phase is often more critical than cooling for successful cryopreservation,
as it determines whether devitrification (ice crystal formation during warming) occurs
and how osmotic stresses are managed during CPA removal.

The designer optimizes:
- Warming rates to prevent devitrification
- CPA dilution steps to minimize osmotic damage
- Temperature holding steps for controlled thawing
""")

# Create sidebar for parameters
st.sidebar.header("Sample Parameters")

sample_type = st.sidebar.selectbox(
    "Sample Type",
    ["Cell Suspension", "Tissue", "Embryo", "Oocyte", "Sperm", "Custom"]
)

# Default parameters based on sample type
if sample_type == "Cell Suspension":
    default_diameter = 2.0
    default_cpa_concentration = 10.0
    default_vitrified = False
    default_critical_warming_rate = 10.0
elif sample_type == "Tissue":
    default_diameter = 10.0
    default_cpa_concentration = 30.0
    default_vitrified = True
    default_critical_warming_rate = 100.0
elif sample_type == "Embryo":
    default_diameter = 0.2
    default_cpa_concentration = 30.0
    default_vitrified = True
    default_critical_warming_rate = 2000.0
elif sample_type == "Oocyte":
    default_diameter = 0.12
    default_cpa_concentration = 30.0
    default_vitrified = True
    default_critical_warming_rate = 1500.0
elif sample_type == "Sperm":
    default_diameter = 0.005
    default_cpa_concentration = 10.0
    default_vitrified = False
    default_critical_warming_rate = 5.0
else:  # Custom
    default_diameter = 5.0
    default_cpa_concentration = 15.0
    default_vitrified = False
    default_critical_warming_rate = 50.0

sample_diameter = st.sidebar.slider(
    "Sample Diameter (mm)",
    min_value=0.001,
    max_value=20.0,
    value=default_diameter,
    step=0.001,
    format="%.3f"
)

is_vitrified = st.sidebar.checkbox(
    "Sample is Vitrified",
    value=default_vitrified,
    help="Check if the sample was vitrified (glass-like state) rather than frozen with ice crystals"
)

# CPA parameters
st.sidebar.header("Cryoprotectant Parameters")

cpa_type = st.sidebar.selectbox(
    "Primary Cryoprotectant",
    ["DMSO", "Glycerol", "Ethylene Glycol", "Propylene Glycol", "Mixture"]
)

cpa_concentration = st.sidebar.slider(
    "CPA Concentration (% v/v)",
    min_value=5.0,
    max_value=60.0,
    value=default_cpa_concentration,
    step=1.0
)

if is_vitrified:
    critical_warming_rate = st.sidebar.slider(
        "Critical Warming Rate (°C/min)",
        min_value=1.0,
        max_value=10000.0,
        value=default_critical_warming_rate,
        step=1.0,
        help="Minimum warming rate required to prevent devitrification"
    )
else:
    critical_warming_rate = 0.0

# Warming parameters
st.sidebar.header("Warming Parameters")

initial_temp = st.sidebar.slider(
    "Initial Temperature (°C)",
    min_value=-196.0,
    max_value=-80.0,
    value=-196.0,
    step=1.0,
    help="Starting temperature of the sample"
)

target_temp = st.sidebar.slider(
    "Target Temperature (°C)",
    min_value=0.0,
    max_value=37.0,
    value=37.0,
    step=1.0,
    help="Final temperature for the sample"
)

# Define warming protocol functions
def calculate_warming_time(initial_temp, target_temp, warming_rate):
    """Calculate time required to warm from initial to target temperature."""
    return (target_temp - initial_temp) / warming_rate

def calculate_temperature_profile(initial_temp, warming_rate, time_points):
    """Calculate temperature at each time point during warming."""
    temperatures = []
    for t in time_points:
        temp = initial_temp + warming_rate * t
        temperatures.append(min(temp, target_temp))
    
    return temperatures

def calculate_achievable_warming_rate(sample_diameter, method):
    """Calculate the maximum achievable warming rate based on sample size and method."""
    # Simplified model based on heat transfer principles
    # Warming rate ~ k / (ρ * c * d²)
    # where k = thermal conductivity, ρ = density, c = specific heat, d = diameter
    
    # Base rate for a 1mm sample with direct plunge into warm water
    base_rate = 200.0  # °C/min
    
    # Method factors
    method_factors = {
        "Liquid Nitrogen to Air": 0.05,
        "Liquid Nitrogen to Room Temp Water": 1.0,
        "Liquid Nitrogen to Warm Water (37°C)": 2.0,
        "Liquid Nitrogen to Hot Water (45°C)": 2.5,
        "Nitrogen Vapor to Warm Water": 1.5,
        "Microwave Warming": 3.0,
        "Laser Warming": 10.0,
        "Electromagnetic Warming": 8.0
    }
    
    # Calculate rate based on diameter (inverse square relationship)
    # Convert diameter to mm for calculation
    diameter_mm = sample_diameter
    rate = base_rate * (1.0 / diameter_mm)**2 * method_factors[method]
    
    return rate

def design_cpa_dilution_protocol(initial_concentration, steps):
    """Design a step-wise CPA dilution protocol to minimize osmotic shock."""
    # For a given number of steps, calculate optimal concentrations
    # Using a non-linear reduction to account for osmotic considerations
    
    concentrations = []
    
    if steps == 1:
        # Single step: direct dilution to 0
        concentrations = [0.0]
    else:
        # Multi-step: non-linear reduction
        for i in range(1, steps + 1):
            # Use a power function to create more gradual initial steps
            fraction = (1 - (i / steps)**1.5)
            conc = initial_concentration * fraction
            concentrations.append(max(0.0, conc))
    
    return concentrations

def calculate_osmotic_stress(current_conc, next_conc):
    """Calculate relative osmotic stress during a dilution step."""
    # Simplified model: stress is proportional to the relative concentration change
    if current_conc == 0:
        return 0
    
    relative_change = abs(next_conc - current_conc) / current_conc
    stress = relative_change * 100  # Convert to percentage
    
    return min(stress, 100)  # Cap at 100%

# Create tabs for different protocol aspects
tab1, tab2, tab3 = st.tabs(["Warming Rate Analysis", "CPA Dilution Protocol", "Complete Protocol"])

with tab1:
    st.subheader("Warming Rate Analysis")
    
    # Create warming methods selection
    warming_methods = [
        "Liquid Nitrogen to Air",
        "Liquid Nitrogen to Room Temp Water",
        "Liquid Nitrogen to Warm Water (37°C)",
        "Liquid Nitrogen to Hot Water (45°C)",
        "Nitrogen Vapor to Warm Water",
        "Microwave Warming",
        "Laser Warming",
        "Electromagnetic Warming"
    ]
    
    selected_methods = st.multiselect(
        "Select Warming Methods to Compare",
        warming_methods,
        default=["Liquid Nitrogen to Air", "Liquid Nitrogen to Warm Water (37°C)", "Laser Warming"]
    )
    
    # Calculate achievable warming rates for each method
    method_rates = {}
    for method in selected_methods:
        rate = calculate_achievable_warming_rate(sample_diameter, method)
        method_rates[method] = rate
    
    # Create comparison table
    warming_data = {
        "Warming Method": list(method_rates.keys()),
        "Achievable Rate (°C/min)": [f"{rate:.1f}" for rate in method_rates.values()],
        "Warming Time (min)": [f"{calculate_warming_time(initial_temp, target_temp, rate):.2f}" for rate in method_rates.values()],
        "Meets Critical Rate": ["Yes" if rate >= critical_warming_rate else "No" for rate in method_rates.values()]
    }
    
    warming_df = pd.DataFrame(warming_data)
    st.table(warming_df)
    
    # Create warming rate vs. sample size plot
    st.subheader("Warming Rate vs. Sample Size")
    
    # Create diameter range
    diameters = np.logspace(-3, 1.5, 100)  # 0.001 to 30 mm
    
    # Calculate warming rates for each method and diameter
    method_diameter_rates = {}
    for method in selected_methods:
        rates = []
        for d in diameters:
            rate = calculate_achievable_warming_rate(d, method)
            rates.append(rate)
        method_diameter_rates[method] = rates
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot warming rate for each method
    for method, rates in method_diameter_rates.items():
        ax.loglog(diameters, rates, linewidth=2, label=method)
    
    # Add critical warming rate line if vitrified
    if is_vitrified:
        ax.axhline(y=critical_warming_rate, color='red', linestyle='--', 
                  label=f'Critical Rate: {critical_warming_rate} °C/min')
    
    # Mark current sample diameter
    ax.axvline(x=sample_diameter, color='black', linestyle=':', 
              label=f'Current Diameter: {sample_diameter} mm')
    
    ax.set_xlabel('Sample Diameter (mm)')
    ax.set_ylabel('Warming Rate (°C/min)')
    ax.set_title('Achievable Warming Rate vs. Sample Size')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
    
    # Select best warming method
    if selected_methods:
        best_method = max(method_rates.items(), key=lambda x: x[1])
        best_method_name = best_method[0]
        best_method_rate = best_method[1]
        
        st.success(f"**Recommended Warming Method**: {best_method_name} (Rate: {best_method_rate:.1f} °C/min)")
        
        if is_vitrified and best_method_rate < critical_warming_rate:
            st.warning(f"⚠️ The achievable warming rate ({best_method_rate:.1f} °C/min) is below the critical warming rate ({critical_warming_rate} °C/min). Devitrification may occur.")
            
            # Calculate maximum viable diameter
            max_viable_diameter = None
            for d in reversed(diameters):
                rate = calculate_achievable_warming_rate(d, best_method_name)
                if rate >= critical_warming_rate:
                    max_viable_diameter = d
                    break
            
            if max_viable_diameter:
                st.info(f"💡 For {best_method_name}, the maximum sample diameter that can be warmed above the critical rate is {max_viable_diameter:.3f} mm.")
    
    # Create temperature profile for selected method
    if selected_methods:
        st.subheader("Temperature Profile During Warming")
        
        # Use the best method for the profile
        selected_rate = best_method_rate
        
        # Calculate warming time
        warming_time = calculate_warming_time(initial_temp, target_temp, selected_rate)
        
        # Create time points
        time_points = np.linspace(0, warming_time * 1.1, 100)  # Add 10% margin
        
        # Calculate temperature profile
        temperatures = calculate_temperature_profile(initial_temp, selected_rate, time_points)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(time_points, temperatures, 'b-', linewidth=2)
        
        # Add phase transition zone if vitrified
        if is_vitrified:
            # Approximate glass transition temperature based on CPA concentration
            # Higher CPA concentration lowers the glass transition temperature
            tg = -100 + (cpa_concentration / 60) * 20  # Simplified model
            
            # Devitrification risk zone (between Tg and -40°C)
            ax.axhspan(tg, -40, alpha=0.2, color='red', label='Devitrification Risk Zone')
            
            # Add glass transition line
            ax.axhline(y=tg, color='purple', linestyle='--', 
                      label=f'Glass Transition Temp: {tg:.1f}°C')
        
        # Add ice melting line
        melting_point = -0.6 * cpa_concentration  # Simplified depression of freezing point
        ax.axhline(y=melting_point, color='blue', linestyle='--', 
                  label=f'Melting Point: {melting_point:.1f}°C')
        
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Temperature Profile Using {best_method_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
        # Calculate time spent in devitrification risk zone
        if is_vitrified:
            tg = -100 + (cpa_concentration / 60) * 20
            
            time_at_tg = (tg - initial_temp) / selected_rate
            time_at_minus40 = (-40 - initial_temp) / selected_rate
            
            time_in_risk_zone = time_at_minus40 - time_at_tg
            
            st.metric("Time in Devitrification Risk Zone", f"{time_in_risk_zone:.2f} min")
            
            if time_in_risk_zone < 0.5:
                st.success("✅ Rapid passage through devitrification risk zone (< 0.5 min)")
            elif time_in_risk_zone < 1.0:
                st.info("ℹ️ Moderate passage through devitrification risk zone (< 1.0 min)")
            else:
                st.warning("⚠️ Slow passage through devitrification risk zone (> 1.0 min)")

with tab2:
    st.subheader("CPA Dilution Protocol")
    
    # Select number of dilution steps
    num_dilution_steps = st.slider(
        "Number of Dilution Steps",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )
    
    # Design dilution protocol
    dilution_concentrations = design_cpa_dilution_protocol(cpa_concentration, num_dilution_steps)
    
    # Calculate osmotic stress for each step
    osmotic_stresses = []
    current_conc = cpa_concentration
    
    for next_conc in dilution_concentrations:
        stress = calculate_osmotic_stress(current_conc, next_conc)
        osmotic_stresses.append(stress)
        current_conc = next_conc
    
    # Create dilution protocol table
    dilution_data = {
        "Step": list(range(1, num_dilution_steps + 1)),
        f"{cpa_type} Concentration (% v/v)": [f"{conc:.1f}" for conc in dilution_concentrations],
        "Osmotic Stress (%)": [f"{stress:.1f}" for stress in osmotic_stresses]
    }
    
    dilution_df = pd.DataFrame(dilution_data)
    st.table(dilution_df)
    
    # Create dilution protocol plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot concentrations
    steps = list(range(0, num_dilution_steps + 1))
    concentrations = [cpa_concentration] + dilution_concentrations
    
    ax.plot(steps, concentrations, 'b-o', linewidth=2)
    
    ax.set_xlabel('Dilution Step')
    ax.set_ylabel(f'{cpa_type} Concentration (% v/v)')
    ax.set_title('CPA Dilution Protocol')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks
    ax.set_xticks(steps)
    ax.set_xticklabels(['Initial'] + [f'Step {i}' for i in range(1, num_dilution_steps + 1)])
    
    st.pyplot(fig)
    
    # Create osmotic stress plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot osmotic stress
    ax.bar(range(1, num_dilution_steps + 1), osmotic_stresses, color='orange', alpha=0.7)
    
    # Add stress level lines
    ax.axhline(y=30, color='green', linestyle='--', label='Low Stress (30%)')
    ax.axhline(y=50, color='orange', linestyle='--', label='Medium Stress (50%)')
    ax.axhline(y=70, color='red', linestyle='--', label='High Stress (70%)')
    
    ax.set_xlabel('Dilution Step')
    ax.set_ylabel('Osmotic Stress (%)')
    ax.set_title('Osmotic Stress During CPA Dilution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set x-axis ticks
    ax.set_xticks(range(1, num_dilution_steps + 1))
    ax.set_xticklabels([f'Step {i}' for i in range(1, num_dilution_steps + 1)])
    
    st.pyplot(fig)
    
    # Calculate maximum osmotic stress
    max_stress = max(osmotic_stresses)
    max_stress_step = osmotic_stresses.index(max_stress) + 1
    
    # Provide recommendations
    if max_stress > 70:
        st.warning(f"⚠️ High osmotic stress ({max_stress:.1f}%) detected in Step {max_stress_step}. Consider increasing the number of dilution steps.")
    elif max_stress > 50:
        st.info(f"ℹ️ Medium osmotic stress ({max_stress:.1f}%) detected in Step {max_stress_step}. This may be acceptable for robust cell types.")
    else:
        st.success(f"✅ Low osmotic stress levels (max: {max_stress:.1f}%). This protocol should minimize osmotic damage.")
    
    # Calculate equilibration times
    st.subheader("Equilibration Times")
    
    # Simplified model for equilibration time
    # Time ~ volume / (permeability * surface area)
    
    # Estimate cell diameter based on sample type
    if sample_type == "Red Blood Cell":
        cell_diameter = 8.0  # μm
    elif sample_type == "Oocyte":
        cell_diameter = 120.0  # μm
    elif sample_type == "Sperm":
        cell_diameter = 5.0  # μm
    elif sample_type == "Embryo":
        cell_diameter = 80.0  # μm
    else:
        cell_diameter = 20.0  # μm
    
    # Estimate permeability based on CPA type
    if cpa_type == "DMSO":
        permeability_factor = 1.0
    elif cpa_type == "Glycerol":
        permeability_factor = 0.6
    elif cpa_type == "Ethylene Glycol":
        permeability_factor = 1.2
    elif cpa_type == "Propylene Glycol":
        permeability_factor = 0.8
    else:
        permeability_factor = 0.9
    
    # Calculate base equilibration time (minutes)
    base_time = (cell_diameter ** 2) / (permeability_factor * 100)
    
    # Calculate equilibration times for each step
    equilibration_times = []
    
    for i, conc in enumerate(dilution_concentrations):
        # Adjust time based on concentration change
        if i == 0:
            prev_conc = cpa_concentration
        else:
            prev_conc = dilution_concentrations[i-1]
        
        conc_factor = 1.0 + 0.5 * (abs(prev_conc - conc) / max(prev_conc, 1.0))
        
        # Calculate time for this step
        step_time = base_time * conc_factor
        
        equilibration_times.append(step_time)
    
    # Create equilibration time table
    equilibration_data = {
        "Step": list(range(1, num_dilution_steps + 1)),
        "Equilibration Time (min)": [f"{time:.1f}" for time in equilibration_times],
        "Cumulative Time (min)": [f"{sum(equilibration_times[:i+1]):.1f}" for i in range(num_dilution_steps)]
    }
    
    equilibration_df = pd.DataFrame(equilibration_data)
    st.table(equilibration_df)
    
    # Calculate total protocol time
    total_time = sum(equilibration_times)
    st.metric("Total CPA Removal Time", f"{total_time:.1f} min")

with tab3:
    st.subheader("Complete Warming Protocol")
    
    # Use the best warming method if available
    if 'best_method_name' in locals() and 'best_method_rate' in locals():
        warming_method = best_method_name
        warming_rate = best_method_rate
    else:
        # Default values if tab1 wasn't visited
        warming_method = "Liquid Nitrogen to Warm Water (37°C)"
        warming_rate = calculate_achievable_warming_rate(sample_diameter, warming_method)
    
    # Calculate warming time
    warming_time = calculate_warming_time(initial_temp, target_temp, warming_rate)
    
    # Use dilution protocol from tab2
    if 'dilution_concentrations' in locals() and 'equilibration_times' in locals():
        dilution_steps = num_dilution_steps
        dilution_concs = dilution_concentrations
        equilibration_times = equilibration_times
    else:
        # Default values if tab2 wasn't visited
        dilution_steps = 3
        dilution_concs = design_cpa_dilution_protocol(cpa_concentration, 3)
        
        # Calculate default equilibration times
        if sample_type == "Red Blood Cell":
            cell_diameter = 8.0  # μm
        elif sample_type == "Oocyte":
            cell_diameter = 120.0  # μm
        elif sample_type == "Sperm":
            cell_diameter = 5.0  # μm
        elif sample_type == "Embryo":
            cell_diameter = 80.0  # μm
        else:
            cell_diameter = 20.0  # μm
        
        if cpa_type == "DMSO":
            permeability_factor = 1.0
        elif cpa_type == "Glycerol":
            permeability_factor = 0.6
        elif cpa_type == "Ethylene Glycol":
            permeability_factor = 1.2
        elif cpa_type == "Propylene Glycol":
            permeability_factor = 0.8
        else:
            permeability_factor = 0.9
        
        base_time = (cell_diameter ** 2) / (permeability_factor * 100)
        equilibration_times = [base_time * 1.5] * 3  # Simplified default
    
    # Create complete protocol
    protocol_steps = []
    
    # Step 1: Initial warming
    protocol_steps.append({
        "Step": "Initial Warming",
        "Description": f"Warm from {initial_temp}°C to {target_temp}°C using {warming_method}",
        "Temperature": f"{initial_temp}°C → {target_temp}°C",
        "Duration": f"{warming_time:.2f} min",
        "Rate": f"{warming_rate:.1f}°C/min"
    })
    
    # Step 2: Temperature stabilization (if needed)
    if target_temp < 20:
        protocol_steps.append({
            "Step": "Temperature Stabilization",
            "Description": f"Hold at {target_temp}°C to ensure uniform temperature",
            "Temperature": f"{target_temp}°C",
            "Duration": "2.00 min",
            "Rate": "0°C/min"
        })
    
    # Steps 3+: CPA dilution
    for i in range(dilution_steps):
        protocol_steps.append({
            "Step": f"CPA Dilution {i+1}",
            "Description": f"Dilute to {dilution_concs[i]:.1f}% {cpa_type} and equilibrate",
            "Temperature": f"{target_temp}°C",
            "Duration": f"{equilibration_times[i]:.2f} min",
            "Rate": "0°C/min"
        })
    
    # Final step: Transfer to culture medium
    protocol_steps.append({
        "Step": "Final Transfer",
        "Description": "Transfer to culture/storage medium",
        "Temperature": f"{target_temp}°C",
        "Duration": "N/A",
        "Rate": "N/A"
    })
    
    # Create protocol table
    protocol_df = pd.DataFrame(protocol_steps)
    st.table(protocol_df)
    
    # Calculate total protocol time
    total_protocol_time = warming_time + sum(equilibration_times)
    if target_temp < 20:
        total_protocol_time += 2.0  # Add stabilization time
    
    st.metric("Total Protocol Time", f"{total_protocol_time:.2f} min")
    
    # Create protocol visualization
    st.subheader("Protocol Visualization")
    
    # Create time points for visualization
    current_time = 0
    time_points = [0]
    temp_points = [initial_temp]
    cpa_points = [cpa_concentration]
    labels = ["Start"]
    
    # Add warming phase
    warming_times = np.linspace(0, warming_time, 50)[1:]
    for t in warming_times:
        time_points.append(current_time + t)
        temp = initial_temp + warming_rate * t
        temp_points.append(min(temp, target_temp))
        cpa_points.append(cpa_concentration)
    
    current_time += warming_time
    time_points.append(current_time)
    temp_points.append(target_temp)
    cpa_points.append(cpa_concentration)
    labels.append("Warming Complete")
    
    # Add stabilization if needed
    if target_temp < 20:
        current_time += 2.0
        time_points.append(current_time)
        temp_points.append(target_temp)
        cpa_points.append(cpa_concentration)
        labels.append("Stabilization Complete")
    
    # Add CPA dilution steps
    current_cpa = cpa_concentration
    for i in range(dilution_steps):
        current_time += equilibration_times[i]
        time_points.append(current_time)
        temp_points.append(target_temp)
        current_cpa = dilution_concs[i]
        cpa_points.append(current_cpa)
        labels.append(f"Dilution {i+1} Complete")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot temperature profile
    ax1.plot(time_points, temp_points, 'b-', linewidth=2)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Complete Warming Protocol')
    ax1.grid(True, alpha=0.3)
    
    # Add phase transition lines if vitrified
    if is_vitrified:
        # Glass transition temperature
        tg = -100 + (cpa_concentration / 60) * 20
        ax1.axhline(y=tg, color='purple', linestyle='--', 
                   label=f'Glass Transition Temp: {tg:.1f}°C')
        
        # Devitrification risk zone
        ax1.axhspan(tg, -40, alpha=0.2, color='red', label='Devitrification Risk Zone')
    
    # Add ice melting line
    melting_point = -0.6 * cpa_concentration
    ax1.axhline(y=melting_point, color='blue', linestyle='--', 
               label=f'Melting Point: {melting_point:.1f}°C')
    
    ax1.legend()
    
    # Plot CPA concentration
    ax2.plot(time_points, cpa_points, 'g-', linewidth=2)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel(f'{cpa_type} Concentration (% v/v)')
    ax2.grid(True, alpha=0.3)
    
    # Add vertical lines for protocol steps
    for i, (t, label) in enumerate(zip(time_points, labels)):
        if i > 0:  # Skip the first point
            ax1.axvline(x=t, color='gray', linestyle=':', alpha=0.7)
            ax2.axvline(x=t, color='gray', linestyle=':', alpha=0.7)
            
            # Add label to bottom plot only
            ax2.annotate(label, 
                        (t, 0), 
                        xytext=(0, -30), 
                        textcoords='offset points',
                        rotation=90,
                        ha='center',
                        fontsize=8)
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Protocol recommendations
    st.subheader("Protocol Recommendations")
    
    recommendations = []
    
    # Warming method recommendations
    if is_vitrified and warming_rate < critical_warming_rate:
        recommendations.append("⚠️ **Warning**: The achievable warming rate is below the critical rate. Consider reducing sample size or using a more rapid warming method.")
    
    # Sample size recommendations
    if sample_diameter > 5.0 and is_vitrified:
        recommendations.append("⚠️ **Sample Size**: Large samples are difficult to warm rapidly enough to prevent devitrification. Consider dividing into smaller pieces if possible.")
    
    # CPA dilution recommendations
    max_stress = max(osmotic_stresses) if 'osmotic_stresses' in locals() else 0
    if max_stress > 70:
        recommendations.append("⚠️ **CPA Dilution**: High osmotic stress detected. Increase the number of dilution steps to reduce stress.")
    
    # Temperature recommendations
    if target_temp < 20 and not "Temperature Stabilization" in [step["Step"] for step in protocol_steps]:
        recommendations.append("ℹ️ **Temperature**: Consider adding a stabilization step before CPA dilution to ensure uniform temperature.")
    
    # General recommendations
    recommendations.append("✅ **Timing**: Minimize the time between steps to prevent temperature fluctuations.")
    recommendations.append("✅ **Solution Volumes**: Use pre-warmed solutions with volumes at least 10x the sample volume.")
    
    if sample_type in ["Embryo", "Oocyte"]:
        recommendations.append("✅ **Specific Recommendation**: For embryos and oocytes, use a sucrose-containing solution in the final dilution step to prevent osmotic shock.")
    
    # Display recommendations
    for rec in recommendations:
        st.markdown(rec)
    
    # Export protocol option
    st.subheader("Export Protocol")
    
    # Create protocol text
    protocol_text = f"# Warming Protocol for {sample_type}\n\n"
    protocol_text += f"## Sample Information\n"
    protocol_text += f"- Sample Type: {sample_type}\n"
    protocol_text += f"- Sample Diameter: {sample_diameter} mm\n"
    protocol_text += f"- Cryoprotectant: {cpa_type} at {cpa_concentration}% v/v\n"
    protocol_text += f"- Vitrified: {'Yes' if is_vitrified else 'No'}\n\n"
    
    protocol_text += f"## Protocol Steps\n"
    for i, step in enumerate(protocol_steps):
        protocol_text += f"{i+1}. **{step['Step']}**\n"
        protocol_text += f"   - Description: {step['Description']}\n"
        protocol_text += f"   - Temperature: {step['Temperature']}\n"
        protocol_text += f"   - Duration: {step['Duration']}\n"
        protocol_text += f"   - Rate: {step['Rate']}\n\n"
    
    protocol_text += f"## Recommendations\n"
    for rec in recommendations:
        protocol_text += f"- {rec.replace('✅', '').replace('ℹ️', '').replace('⚠️', '')}\n"
    
    # Create download button
    st.download_button(
        label="Download Protocol as Text",
        data=protocol_text,
        file_name=f"warming_protocol_{sample_type.lower().replace(' ', '_')}.txt",
        mime="text/plain"
    )

# Explanation section
st.markdown("""
## Warming Theory in Cryopreservation

### Critical Aspects of Warming

1. **Warming Rate**:
   - For vitrified samples, warming rate is often more critical than cooling rate
   - Rapid warming prevents devitrification (ice formation during warming)
   - Warming rate requirements increase with CPA concentration

2. **Devitrification Risk**:
   - Highest risk occurs between the glass transition temperature (Tg) and -40°C
   - Rapid passage through this zone is essential for vitrified samples
   - Devitrification can cause mechanical damage similar to freezing

3. **CPA Removal**:
   - Osmotic stress during CPA removal can damage cells
   - Step-wise dilution minimizes volume excursions
   - Equilibration time at each step is critical

### Warming Methods

1. **Air Warming**:
   - Slowest method (~10-50°C/min)
   - Simple but high risk of devitrification for vitrified samples
   - Suitable only for slow-frozen samples with low CPA concentrations

2. **Water Bath Warming**:
   - Moderate to fast rates (100-1000°C/min)
   - Standard method for many applications
   - Temperature can be adjusted to optimize rate

3. **Advanced Methods**:
   - Microwave warming: Rapid and volumetric heating
   - Laser warming: Ultra-rapid for small samples
   - Electromagnetic warming: Rapid for metal-doped samples

### Sample Size Considerations

Sample size dramatically affects achievable warming rates:
- Warming rate ∝ 1/d² (inversely proportional to diameter squared)
- Larger samples require longer warming times
- Heat transfer limitations often make vitrification impractical for large samples

### CPA Dilution Strategies

1. **Single-Step Dilution**:
   - Simple but causes severe osmotic stress
   - Suitable only for cells with high osmotic tolerance

2. **Multi-Step Dilution**:
   - Reduces osmotic stress by gradual CPA removal
   - Each step requires equilibration time
   - Non-linear concentration steps often optimal

3. **Non-penetrating Solutes**:
   - Sucrose or other non-penetrating solutes can counterbalance osmotic effects
   - Particularly useful in final dilution steps
""")