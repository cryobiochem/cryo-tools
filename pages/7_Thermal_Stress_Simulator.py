import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

st.title("Thermal Stress Simulator")

st.markdown("""
This tool models thermal stresses that develop during freezing and thawing of biological samples.
Thermal stresses can cause mechanical damage to cells, tissues, and organs during cryopreservation.

The simulator calculates:
- Temperature gradients within the sample
- Thermal expansion/contraction stresses
- Fracture probabilities during cooling and warming
""")

# Create sidebar for parameters
st.sidebar.header("Sample Parameters")

sample_type = st.sidebar.selectbox(
    "Sample Type",
    ["Cell Suspension", "Tissue", "Organ", "Embryo", "Custom"]
)

# Default parameters based on sample type
if sample_type == "Cell Suspension":
    default_diameter = 2.0
    default_thermal_conductivity = 0.5
    default_elastic_modulus = 0.01
    default_thermal_expansion = 80.0
elif sample_type == "Tissue":
    default_diameter = 10.0
    default_thermal_conductivity = 0.4
    default_elastic_modulus = 0.5
    default_thermal_expansion = 65.0
elif sample_type == "Organ":
    default_diameter = 50.0
    default_thermal_conductivity = 0.3
    default_elastic_modulus = 1.0
    default_thermal_expansion = 60.0
elif sample_type == "Embryo":
    default_diameter = 0.2
    default_thermal_conductivity = 0.45
    default_elastic_modulus = 0.05
    default_thermal_expansion = 75.0
else:  # Custom
    default_diameter = 10.0
    default_thermal_conductivity = 0.4
    default_elastic_modulus = 0.5
    default_thermal_expansion = 70.0

sample_diameter = st.sidebar.slider(
    "Sample Diameter (mm)",
    min_value=0.1,
    max_value=100.0,
    value=default_diameter,
    step=0.1
)

thermal_conductivity = st.sidebar.slider(
    "Thermal Conductivity (W/m·K)",
    min_value=0.1,
    max_value=1.0,
    value=default_thermal_conductivity,
    step=0.01,
    help="Heat transfer coefficient of the sample"
)

elastic_modulus = st.sidebar.slider(
    "Elastic Modulus (MPa)",
    min_value=0.01,
    max_value=10.0,
    value=default_elastic_modulus,
    step=0.01,
    help="Stiffness of the sample at low temperature"
)

thermal_expansion = st.sidebar.slider(
    "Thermal Expansion Coefficient (×10⁻⁶/K)",
    min_value=10.0,
    max_value=100.0,
    value=default_thermal_expansion,
    step=1.0,
    help="How much the sample contracts/expands with temperature"
)

# Cooling parameters
st.sidebar.header("Cooling Parameters")

cooling_rate = st.sidebar.slider(
    "Cooling Rate (°C/min)",
    min_value=0.1,
    max_value=1000.0,
    value=10.0,
    step=0.1
)

min_temperature = st.sidebar.slider(
    "Minimum Temperature (°C)",
    min_value=-196.0,
    max_value=-20.0,
    value=-80.0,
    step=1.0
)

# Define thermal stress model functions
def calculate_temperature_profile(radius, time, thermal_diffusivity, surface_temp, initial_temp):
    """Calculate temperature at different radii within a sphere at a given time."""
    # Thermal diffusivity (m²/s) = thermal conductivity / (density * specific heat)
    # For water-based biological samples, approximate as:
    # density ≈ 1000 kg/m³, specific heat ≈ 4000 J/(kg·K)
    
    # Convert radius to meters
    radius_m = radius / 1000.0
    
    # Calculate temperature profile
    temperatures = []
    for r in radius:
        r_m = r / 1000.0  # Convert to meters
        
        if r_m >= radius_m:
            # At or beyond the surface
            temperatures.append(surface_temp)
        else:
            # Inside the sphere
            # Simplified solution for sphere cooling
            # T(r,t) = Ts + (Ti-Ts) * (r/R) * sin(π*r/R) / sin(π) * exp(-α*π²*t/R²)
            # where Ts = surface temp, Ti = initial temp, α = thermal diffusivity
            
            # For simplicity, use a more basic approximation
            relative_position = r_m / radius_m
            temp_factor = np.exp(-thermal_diffusivity * time / (radius_m**2) * 10)
            temp = surface_temp + (initial_temp - surface_temp) * (1 - relative_position) * temp_factor
            temperatures.append(temp)
    
    return temperatures

def calculate_thermal_stress(temp_gradient, elastic_modulus, thermal_expansion, poisson_ratio=0.3):
    """Calculate thermal stress based on temperature gradient."""
    # Convert units
    elastic_modulus_pa = elastic_modulus * 1e6  # MPa to Pa
    thermal_expansion_coef = thermal_expansion * 1e-6  # per K
    
    # Calculate thermal stress (simplified model)
    # σ = E * α * ΔT * (1 - ν)
    # where E = elastic modulus, α = thermal expansion coefficient, 
    # ΔT = temperature difference, ν = Poisson's ratio
    
    stress = elastic_modulus_pa * thermal_expansion_coef * temp_gradient * (1 - poisson_ratio)
    
    # Convert to MPa for display
    return stress / 1e6

def calculate_fracture_probability(stress, critical_stress=1.0):
    """Calculate probability of fracture based on stress level."""
    # Weibull distribution for fracture probability
    # P = 1 - exp(-(σ/σc)^m)
    # where σ = stress, σc = characteristic strength, m = Weibull modulus
    
    # For biological materials, use simplified model
    m = 2.0  # Weibull modulus
    
    if stress <= 0:
        return 0.0
    
    probability = 1.0 - np.exp(-(stress / critical_stress)**m)
    
    return min(1.0, max(0.0, probability)) * 100  # Convert to percentage

# Calculate thermal diffusivity
def calculate_thermal_diffusivity(thermal_conductivity):
    # Approximate values for biological samples
    density = 1000.0  # kg/m³
    specific_heat = 4000.0  # J/(kg·K)
    
    # Thermal diffusivity = thermal conductivity / (density * specific heat)
    diffusivity = thermal_conductivity / (density * specific_heat)
    
    return diffusivity

thermal_diffusivity = calculate_thermal_diffusivity(thermal_conductivity)

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Temperature Gradients", "Thermal Stress Analysis", "Fracture Risk"])

with tab1:
    st.subheader("Temperature Gradients During Cooling")
    
    # Calculate cooling time
    initial_temp = 22.0  # Room temperature (°C)
    cooling_time = (initial_temp - min_temperature) / cooling_rate  # minutes
    
    # Create time points
    times = np.linspace(0, cooling_time, 6)  # 6 time points during cooling
    
    # Create radial positions
    radius = np.linspace(0, sample_diameter/2, 50)  # From center to surface
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color map for different times
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    
    for i, time in enumerate(times):
        # Calculate surface temperature at this time
        # Linear cooling from initial_temp to min_temperature
        surface_temp = initial_temp - (cooling_rate * time)
        surface_temp = max(surface_temp, min_temperature)
        
        # Calculate temperature profile
        temperatures = calculate_temperature_profile(
            radius, 
            time * 60,  # Convert to seconds
            thermal_diffusivity, 
            surface_temp, 
            initial_temp
        )
        
        # Plot temperature profile
        ax.plot(radius, temperatures, color=colors[i], 
               label=f't = {time:.1f} min, Ts = {surface_temp:.1f}°C')
    
    ax.set_xlabel('Radial Position (mm)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Temperature Gradients During Cooling ({cooling_rate}°C/min)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
    
    # Calculate and display maximum temperature gradient
    # Take the middle time point for analysis
    mid_time_idx = len(times) // 2
    mid_time = times[mid_time_idx]
    
    # Calculate surface temperature
    surface_temp = initial_temp - (cooling_rate * mid_time)
    surface_temp = max(surface_temp, min_temperature)
    
    # Calculate temperature profile
    mid_temperatures = calculate_temperature_profile(
        radius, 
        mid_time * 60,
        thermal_diffusivity, 
        surface_temp, 
        initial_temp
    )
    
    # Calculate temperature gradient (°C/mm)
    temp_gradient = (mid_temperatures[0] - mid_temperatures[-1]) / (radius[-1])
    
    # Calculate temperature gradient (°C/m)
    temp_gradient_m = temp_gradient * 1000
    
    st.metric("Maximum Temperature Gradient", f"{temp_gradient:.2f} °C/mm ({temp_gradient_m:.0f} °C/m)")
    
    # Create 2D visualization of temperature field
    st.subheader("2D Temperature Field Visualization")
    
    # Create a 2D grid
    x = np.linspace(-sample_diameter/2, sample_diameter/2, 100)
    y = np.linspace(-sample_diameter/2, sample_diameter/2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate radial distance from center
    R = np.sqrt(X**2 + Y**2)
    
    # Select a specific time point (e.g., middle of cooling)
    vis_time_idx = len(times) // 2
    vis_time = times[vis_time_idx]
    
    # Calculate surface temperature
    vis_surface_temp = initial_temp - (cooling_rate * vis_time)
    vis_surface_temp = max(vis_surface_temp, min_temperature)
    
    # Create temperature field
    Z = np.zeros_like(R)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            r_val = R[i, j]
            if r_val <= sample_diameter/2:
                # Find closest radius in our calculated profile
                r_idx = np.argmin(np.abs(radius - r_val))
                Z[i, j] = mid_temperatures[r_idx]
            else:
                Z[i, j] = vis_surface_temp
    
    # Create 2D plot
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    contour = ax2.contourf(X, Y, Z, 20, cmap='coolwarm')
    
    # Add contour lines
    contour_lines = ax2.contour(X, Y, Z, 10, colors='black', alpha=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f°C')
    
    # Draw sample outline
    circle = plt.Circle((0, 0), sample_diameter/2, fill=False, color='black', linestyle='--')
    ax2.add_patch(circle)
    
    ax2.set_xlabel('Position (mm)')
    ax2.set_ylabel('Position (mm)')
    ax2.set_title(f'Temperature Field at t = {vis_time:.1f} min')
    ax2.set_aspect('equal')
    
    fig2.colorbar(contour, ax=ax2, label='Temperature (°C)')
    
    st.pyplot(fig2)

with tab2:
    st.subheader("Thermal Stress Analysis")
    
    # Calculate cooling time
    cooling_time = (initial_temp - min_temperature) / cooling_rate  # minutes
    
    # Create time points for entire process
    cooling_times = np.linspace(0, cooling_time, 100)
    
    # Calculate stress over time
    center_temps = []
    surface_temps = []
    temp_gradients = []
    thermal_stresses = []
    
    for time in cooling_times:
        # Calculate surface temperature
        surface_temp = initial_temp - (cooling_rate * time)
        surface_temp = max(surface_temp, min_temperature)
        
        # Calculate temperature profile
        temperatures = calculate_temperature_profile(
            radius, 
            time * 60,
            thermal_diffusivity, 
            surface_temp, 
            initial_temp
        )
        
        # Store center and surface temperatures
        center_temps.append(temperatures[0])
        surface_temps.append(temperatures[-1])
        
        # Calculate temperature gradient
        temp_gradient = (temperatures[0] - temperatures[-1]) / (radius[-1])
        temp_gradients.append(temp_gradient)
        
        # Calculate thermal stress
        stress = calculate_thermal_stress(
            temp_gradient * 1000,  # Convert to °C/m
            elastic_modulus,
            thermal_expansion
        )
        thermal_stresses.append(stress)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot temperatures
    ax1.plot(cooling_times, center_temps, 'b-', label='Center Temperature')
    ax1.plot(cooling_times, surface_temps, 'r-', label='Surface Temperature')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature Evolution During Cooling')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot thermal stress
    ax2.plot(cooling_times, thermal_stresses, 'g-', linewidth=2)
    
    # Add critical stress line
    critical_stress = 1.0  # MPa, approximate fracture stress for biological materials
    ax2.axhline(y=critical_stress, color='r', linestyle='--', 
               label=f'Critical Stress: {critical_stress} MPa')
    
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Thermal Stress (MPa)')
    ax2.set_title('Thermal Stress Evolution During Cooling')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Calculate and display maximum stress
    max_stress = max(thermal_stresses)
    max_stress_time = cooling_times[np.argmax(thermal_stresses)]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Maximum Thermal Stress", f"{max_stress:.3f} MPa")
    with col2:
        st.metric("Time of Maximum Stress", f"{max_stress_time:.1f} min")
    
    # Create 3D visualization of stress vs. cooling rate and sample size
    st.subheader("3D Visualization: Stress vs. Cooling Rate and Sample Size")
    
    # Create parameter ranges
    cooling_rates = np.logspace(-1, 3, 20)  # 0.1 to 1000 °C/min
    diameters = np.linspace(1, 50, 20)  # 1 to 50 mm
    
    C, D = np.meshgrid(cooling_rates, diameters)
    Z_stress = np.zeros_like(C)
    
    for i in range(len(diameters)):
        for j in range(len(cooling_rates)):
            # Calculate approximate maximum stress for these parameters
            # Simplified model: stress ~ cooling_rate * diameter^2 / thermal_conductivity
            diameter_factor = (diameters[i] / sample_diameter) ** 2
            rate_factor = cooling_rates[j] / cooling_rate
            
            Z_stress[i, j] = max_stress * diameter_factor * rate_factor
    
    # Create 3D plot
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    surf = ax3d.plot_surface(np.log10(C), D, Z_stress, cmap=cm.viridis, alpha=0.8)
    
    # Mark current parameters
    ax3d.scatter([np.log10(cooling_rate)], [sample_diameter], [max_stress], 
                color='red', s=100, marker='o')
    
    # Add critical stress plane
    x_plane = np.log10(cooling_rates)
    y_plane = diameters
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.ones_like(X_plane) * critical_stress
    
    ax3d.plot_surface(X_plane, Y_plane, Z_plane, color='red', alpha=0.3)
    
    ax3d.set_xlabel('Cooling Rate (°C/min)')
    ax3d.set_ylabel('Sample Diameter (mm)')
    ax3d.set_zlabel('Maximum Thermal Stress (MPa)')
    
    # Set custom tick positions and labels for cooling rate
    rate_ticks = [-1, 0, 1, 2, 3]
    rate_labels = ['0.1', '1', '10', '100', '1000']
    ax3d.set_xticks(rate_ticks)
    ax3d.set_xticklabels(rate_labels)
    
    ax3d.set_title('Maximum Thermal Stress vs. Cooling Rate and Sample Size')
    
    fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
    
    st.pyplot(fig3d)

with tab3:
    st.subheader("Fracture Risk Analysis")
    
    # Calculate fracture probabilities
    fracture_probs = [calculate_fracture_probability(stress) for stress in thermal_stresses]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(cooling_times, fracture_probs, 'r-', linewidth=2)
    
    # Add risk levels
    ax.axhline(y=5, color='green', linestyle='--', label='Low Risk (5%)')
    ax.axhline(y=50, color='orange', linestyle='--', label='Medium Risk (50%)')
    ax.axhline(y=95, color='red', linestyle='--', label='High Risk (95%)')
    
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Fracture Probability (%)')
    ax.set_title('Fracture Risk During Cooling')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)
    
    # Calculate maximum fracture probability
    max_fracture_prob = max(fracture_probs)
    max_fracture_time = cooling_times[np.argmax(fracture_probs)]
    
    # Determine risk level
    if max_fracture_prob < 5:
        risk_level = "Low"
        risk_color = "green"
    elif max_fracture_prob < 50:
        risk_level = "Medium"
        risk_color = "orange"
    else:
        risk_level = "High"
        risk_color = "red"
    
    # Display risk metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Maximum Fracture Probability", f"{max_fracture_prob:.1f}%")
    with col2:
        st.metric("Time of Maximum Risk", f"{max_fracture_time:.1f} min")
    with col3:
        st.markdown(f"<h3 style='color:{risk_color}'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
    
    # Create a safe cooling rate calculator
    st.subheader("Safe Cooling Rate Calculator")
    
    # Calculate safe cooling rate (where fracture probability < 5%)
    safe_rates = []
    test_rates = np.logspace(-1, 3, 50)  # 0.1 to 1000 °C/min
    
    for rate in test_rates:
        # Simplified calculation of maximum stress for this cooling rate
        rate_factor = rate / cooling_rate
        test_max_stress = max_stress * rate_factor
        
        # Calculate fracture probability
        fracture_prob = calculate_fracture_probability(test_max_stress)
        
        if fracture_prob < 5:
            safe_rates.append(rate)
    
    if len(safe_rates) > 0:
        max_safe_rate = max(safe_rates)
    else:
        max_safe_rate = min(test_rates)
    
    st.metric("Maximum Safe Cooling Rate", f"{max_safe_rate:.1f} °C/min")
    
    # Create cooling rate vs. fracture probability plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Calculate fracture probabilities for different cooling rates
    rate_probs = []
    for rate in test_rates:
        rate_factor = rate / cooling_rate
        test_max_stress = max_stress * rate_factor
        fracture_prob = calculate_fracture_probability(test_max_stress)
        rate_probs.append(fracture_prob)
    
    ax2.semilogx(test_rates, rate_probs, 'b-', linewidth=2)
    
    # Mark current cooling rate
    ax2.axvline(x=cooling_rate, color='black', linestyle='--', 
               label=f'Current Rate: {cooling_rate} °C/min')
    
    # Mark maximum safe cooling rate
    ax2.axvline(x=max_safe_rate, color='green', linestyle='--', 
               label=f'Max Safe Rate: {max_safe_rate:.1f} °C/min')
    
    # Add risk levels
    ax2.axhline(y=5, color='green', linestyle=':', label='Low Risk (5%)')
    ax2.axhline(y=50, color='orange', linestyle=':', label='Medium Risk (50%)')
    ax2.axhline(y=95, color='red', linestyle=':', label='High Risk (95%)')
    
    ax2.set_xlabel('Cooling Rate (°C/min)')
    ax2.set_ylabel('Fracture Probability (%)')
    ax2.set_title('Fracture Risk vs. Cooling Rate')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    st.pyplot(fig2)
    
    # Create a table of safe cooling rates for different sample sizes
    st.subheader("Safe Cooling Rates for Different Sample Sizes")
    
    test_diameters = [1, 2, 5, 10, 20, 50, 100]
    safe_rates_by_size = []
    
    for diam in test_diameters:
        # Calculate safe cooling rate for this diameter
        diameter_factor = (diam / sample_diameter) ** 2
        
        # Find maximum rate where fracture probability < 5%
        size_safe_rates = []
        for rate in test_rates:
            rate_factor = rate / cooling_rate
            test_max_stress = max_stress * diameter_factor * rate_factor
            fracture_prob = calculate_fracture_probability(test_max_stress)
            
            if fracture_prob < 5:
                size_safe_rates.append(rate)
        
        if len(size_safe_rates) > 0:
            size_max_safe_rate = max(size_safe_rates)
        else:
            size_max_safe_rate = min(test_rates)
        
        safe_rates_by_size.append(size_max_safe_rate)
    
    # Create dataframe
    safe_rates_df = pd.DataFrame({
        "Sample Diameter (mm)": test_diameters,
        "Maximum Safe Cooling Rate (°C/min)": [f"{rate:.1f}" for rate in safe_rates_by_size]
    })
    
    st.table(safe_rates_df)

# Explanation section
st.markdown("""
## Thermal Stress Theory in Cryopreservation

### Sources of Thermal Stress

During cryopreservation, thermal stresses arise from:

1. **Temperature Gradients**: Different cooling/warming rates between the surface and interior
2. **Thermal Contraction/Expansion**: Materials contract when cooled and expand when warmed
3. **Phase Changes**: Volume changes during freezing/thawing and glass transitions
4. **Material Property Variations**: Different thermal expansion coefficients between components

### Mechanical Consequences

Thermal stresses can lead to:

1. **Cracking and Fracture**: When stresses exceed the material's strength
2. **Delamination**: Separation between different tissue layers or components
3. **Microstructural Damage**: Disruption of cellular architecture and extracellular matrix
4. **Residual Stress**: Persistent stresses that remain after returning to ambient temperature

### Critical Factors

The magnitude of thermal stress depends on:

1. **Sample Size**: Larger samples develop steeper temperature gradients
2. **Cooling/Warming Rate**: Faster rates create larger temperature gradients
3. **Thermal Properties**: Conductivity, specific heat, and diffusivity
4. **Mechanical Properties**: Elastic modulus, strength, and thermal expansion coefficient

### Mitigation Strategies

To reduce thermal stress damage:

1. **Controlled Cooling Rates**: Use slower rates for larger samples
2. **Sample Size Reduction**: Divide large samples into smaller pieces
3. **Cryoprotectant Optimization**: Some CPAs can modify mechanical properties
4. **Anisotropic Cooling**: Control the direction of heat transfer
5. **Stress-Relieving Holds**: Pause at critical temperatures to allow equilibration

### Applications

Understanding thermal stress is crucial for:

1. **Organ Cryopreservation**: Large organs are particularly susceptible to cracking
2. **Tissue Engineering**: Preserving complex engineered tissues with multiple components
3. **Cryosurgery**: Controlling ice ball formation and tissue damage
4. **Cryobanking**: Ensuring long-term storage stability without mechanical damage
""")
