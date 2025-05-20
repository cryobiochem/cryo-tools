import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

st.set_page_config(
    page_title="Storage Temperature Stability Analyzer - Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide"
)

st.title("Storage Temperature Stability Analyzer")

st.markdown("""
This tool analyzes the stability of cryopreserved samples at different storage temperatures.
Long-term storage stability is critical for biobanking, where samples may be stored for
decades before use.

The analyzer models:
- Ice crystal growth during storage (recrystallization)
- Glass transition temperatures and stability
- Effects of temperature fluctuations
- Predicted shelf life at different temperatures
""")

# Create sidebar for parameters
st.sidebar.header("Sample Parameters")

sample_type = st.sidebar.selectbox(
    "Sample Type",
    ["Cell Suspension", "Tissue", "Embryo", "Oocyte", "Sperm", "Custom"]
)

# Default parameters based on sample type
if sample_type == "Cell Suspension":
    default_cpa_concentration = 10.0
    default_vitrified = False
    default_tg = -120.0
elif sample_type == "Tissue":
    default_cpa_concentration = 30.0
    default_vitrified = True
    default_tg = -110.0
elif sample_type == "Embryo":
    default_cpa_concentration = 30.0
    default_vitrified = True
    default_tg = -110.0
elif sample_type == "Oocyte":
    default_cpa_concentration = 30.0
    default_vitrified = True
    default_tg = -110.0
elif sample_type == "Sperm":
    default_cpa_concentration = 10.0
    default_vitrified = False
    default_tg = -120.0
else:  # Custom
    default_cpa_concentration = 15.0
    default_vitrified = False
    default_tg = -115.0

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

# Glass transition temperature
if is_vitrified:
    # Estimate glass transition temperature based on CPA concentration
    estimated_tg = -130 + (cpa_concentration / 60) * 30  # Simplified model
    
    tg = st.sidebar.slider(
        "Glass Transition Temperature (°C)",
        min_value=-150.0,
        max_value=-80.0,
        value=estimated_tg,
        step=1.0,
        help="Temperature below which the sample is in a glass state"
    )
else:
    tg = default_tg

# Storage parameters
st.sidebar.header("Storage Parameters")

storage_temp = st.sidebar.selectbox(
    "Storage Temperature",
    ["-196°C (Liquid Nitrogen)", "-150°C (Vapor Phase Nitrogen)", 
     "-80°C (Ultra-low Freezer)", "-20°C (Standard Freezer)"]
)

# Convert storage temperature to numeric value
if storage_temp == "-196°C (Liquid Nitrogen)":
    storage_temp_value = -196.0
elif storage_temp == "-150°C (Vapor Phase Nitrogen)":
    storage_temp_value = -150.0
elif storage_temp == "-80°C (Ultra-low Freezer)":
    storage_temp_value = -80.0
else:  # -20°C
    storage_temp_value = -20.0

temp_fluctuation = st.sidebar.slider(
    "Temperature Fluctuation (±°C)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.5,
    help="Typical temperature variation during storage"
)

storage_duration = st.sidebar.slider(
    "Storage Duration (years)",
    min_value=1,
    max_value=50,
    value=10,
    step=1
)

# Define stability model functions
def calculate_recrystallization_rate(temperature, tg):
    """Calculate relative recrystallization rate at a given temperature."""
    # Recrystallization is most rapid just below the melting point
    # and decreases as temperature decreases
    # Minimal below Tg (glass transition temperature)
    
    # Melting point depression due to CPA (simplified model)
    melting_point = -0.6 * cpa_concentration
    
    # No recrystallization below Tg
    if temperature <= tg:
        return 0.0
    
    # Maximum rate near melting point, decreasing as temperature decreases
    # Normalized to 1.0 at melting point
    if temperature >= melting_point:
        return 1.0
    
    # Rate between Tg and melting point (non-linear relationship)
    # Higher rate closer to melting point
    normalized_temp = (temperature - tg) / (melting_point - tg)
    rate = normalized_temp ** 2  # Non-linear increase
    
    return rate

def calculate_degradation_rate(temperature):
    """Calculate relative biochemical degradation rate at a given temperature."""
    # Biochemical degradation follows Arrhenius relationship
    # Rate decreases exponentially with decreasing temperature
    
    # Convert temperature to Kelvin
    temp_k = temperature + 273.15
    
    # Reference temperature (0°C)
    ref_temp_k = 273.15
    
    # Activation energy (kJ/mol) - typical for biological systems
    ea = 50.0
    
    # Gas constant (8.314 J/mol·K)
    r = 8.314
    
    # Calculate relative rate using Arrhenius equation
    # Normalized to 1.0 at 0°C
    rate = np.exp((ea * 1000 / r) * (1/ref_temp_k - 1/temp_k))
    
    return rate

def calculate_stability_index(temperature, tg, is_vitrified):
    """Calculate overall stability index at a given temperature."""
    # Stability index combines recrystallization and degradation effects
    # Lower index = more stable
    
    # Calculate component rates
    recryst_rate = calculate_recrystallization_rate(temperature, tg)
    degrad_rate = calculate_degradation_rate(temperature)
    
    # Weight factors depend on whether sample is vitrified
    if is_vitrified:
        # Vitrified samples are more sensitive to recrystallization
        recryst_weight = 0.8
        degrad_weight = 0.2
    else:
        # Frozen samples are less sensitive to recrystallization
        recryst_weight = 0.4
        degrad_weight = 0.6
    
    # Calculate weighted index
    index = recryst_weight * recryst_rate + degrad_weight * degrad_rate
    
    return index

def estimate_shelf_life(temperature, tg, is_vitrified):
    """Estimate shelf life in years at a given storage temperature."""
    # Base shelf life at -196°C (liquid nitrogen)
    base_shelf_life = 100.0  # years
    
    # Calculate stability index
    stability_index = calculate_stability_index(temperature, tg, is_vitrified)
    
    # Shelf life is inversely proportional to stability index
    # Normalized to base shelf life at -196°C
    reference_index = calculate_stability_index(-196, tg, is_vitrified)
    
    # Avoid division by zero
    if reference_index == 0:
        reference_index = 1e-10
    
    # Calculate shelf life
    shelf_life = base_shelf_life * (reference_index / stability_index)
    
    # Cap at reasonable maximum
    return min(shelf_life, 1000.0)

def calculate_crystal_growth(temperature, tg, duration_years):
    """Calculate relative ice crystal growth during storage."""
    # Crystal growth is proportional to recrystallization rate and time
    recryst_rate = calculate_recrystallization_rate(temperature, tg)
    
    # Convert duration to days
    duration_days = duration_years * 365
    
    # Calculate growth (simplified model)
    # Growth is non-linear with time (slows down)
    growth = recryst_rate * np.sqrt(duration_days) / 10.0
    
    # Normalize to [0, 1] range
    return min(growth, 1.0)

def calculate_viability_loss(temperature, tg, is_vitrified, duration_years):
    """Calculate estimated viability loss during storage."""
    # Viability loss depends on stability index and time
    stability_index = calculate_stability_index(temperature, tg, is_vitrified)
    
    # Calculate viability loss (simplified model)
    # Loss increases with time but plateaus
    loss = stability_index * (1.0 - np.exp(-0.05 * duration_years))
    
    # Convert to percentage and cap at 100%
    return min(loss * 100, 100.0)

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Temperature Analysis", "Storage Duration Effects", "Temperature Fluctuations"])

with tab1:
    st.subheader("Storage Temperature Analysis")
    
    # Create temperature range
    temperatures = np.linspace(-200, -10, 100)
    
    # Calculate stability metrics for each temperature
    recryst_rates = [calculate_recrystallization_rate(t, tg) for t in temperatures]
    degrad_rates = [calculate_degradation_rate(t) for t in temperatures]
    stability_indices = [calculate_stability_index(t, tg, is_vitrified) for t in temperatures]
    shelf_lives = [estimate_shelf_life(t, tg, is_vitrified) for t in temperatures]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot rates
    ax1.semilogy(temperatures, recryst_rates, 'b-', linewidth=2, label='Recrystallization Rate')
    ax1.semilogy(temperatures, degrad_rates, 'g-', linewidth=2, label='Degradation Rate')
    ax1.semilogy(temperatures, stability_indices, 'r-', linewidth=2, label='Stability Index')
    
    # Add reference lines
    ax1.axvline(x=tg, color='purple', linestyle='--', 
               label=f'Glass Transition (Tg): {tg}°C')
    
    # Mark storage temperature
    ax1.axvline(x=storage_temp_value, color='black', linestyle=':', 
               label=f'Storage Temp: {storage_temp_value}°C')
    
    ax1.set_ylabel('Relative Rate (log scale)')
    ax1.set_title('Stability Rates vs. Temperature')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot shelf life
    ax2.semilogy(temperatures, shelf_lives, 'r-', linewidth=2)
    
    # Add reference lines
    ax2.axvline(x=tg, color='purple', linestyle='--', 
               label=f'Glass Transition (Tg): {tg}°C')
    
    # Mark storage temperature
    ax2.axvline(x=storage_temp_value, color='black', linestyle=':', 
               label=f'Storage Temp: {storage_temp_value}°C')
    
    # Add horizontal reference lines
    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.7, label='1 year')
    ax2.axhline(y=10, color='gray', linestyle=':', alpha=0.7, label='10 years')
    ax2.axhline(y=100, color='gray', linestyle=':', alpha=0.7, label='100 years')
    
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Estimated Shelf Life (years)')
    ax2.set_title('Shelf Life vs. Temperature')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Calculate stability metrics for current storage temperature
    current_recryst_rate = calculate_recrystallization_rate(storage_temp_value, tg)
    current_degrad_rate = calculate_degradation_rate(storage_temp_value)
    current_stability_index = calculate_stability_index(storage_temp_value, tg, is_vitrified)
    current_shelf_life = estimate_shelf_life(storage_temp_value, tg, is_vitrified)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Recrystallization Rate", f"{current_recryst_rate:.3f}")
    
    with col2:
        st.metric("Degradation Rate", f"{current_degrad_rate:.3e}")
    
    with col3:
        st.metric("Estimated Shelf Life", f"{current_shelf_life:.1f} years")
    
    # Create comparison table for different storage temperatures
    storage_options = [
        {"name": "Liquid Nitrogen", "temp": -196.0},
        {"name": "Vapor Phase Nitrogen", "temp": -150.0},
        {"name": "Ultra-low Freezer", "temp": -80.0},
        {"name": "Standard Freezer", "temp": -20.0}
    ]
    
    comparison_data = {
        "Storage Method": [],
        "Temperature (°C)": [],
        "Recrystallization Rate": [],
        "Degradation Rate": [],
        "Stability Index": [],
        "Estimated Shelf Life (years)": []
    }
    
    for option in storage_options:
        comparison_data["Storage Method"].append(option["name"])
        comparison_data["Temperature (°C)"].append(option["temp"])
        comparison_data["Recrystallization Rate"].append(f"{calculate_recrystallization_rate(option['temp'], tg):.3f}")
        comparison_data["Degradation Rate"].append(f"{calculate_degradation_rate(option['temp']):.3e}")
        comparison_data["Stability Index"].append(f"{calculate_stability_index(option['temp'], tg, is_vitrified):.3e}")
        comparison_data["Estimated Shelf Life (years)"].append(f"{estimate_shelf_life(option['temp'], tg, is_vitrified):.1f}")
    
    comparison_df = pd.DataFrame(comparison_data)
    st.subheader("Storage Method Comparison")
    st.table(comparison_df)
    
    # Provide recommendations
    st.subheader("Storage Recommendations")
    
    if is_vitrified:
        if storage_temp_value > tg + 5:
            st.warning(f"⚠️ The storage temperature ({storage_temp_value}°C) is significantly above the glass transition temperature ({tg}°C). This may lead to devitrification and reduced sample viability.")
            st.info(f"💡 Recommendation: Store vitrified samples below the glass transition temperature, preferably in liquid nitrogen (-196°C) or vapor phase nitrogen (-150°C).")
        elif storage_temp_value > tg:
            st.warning(f"⚠️ The storage temperature ({storage_temp_value}°C) is above the glass transition temperature ({tg}°C). This may lead to slow devitrification over time.")
            st.info(f"💡 Recommendation: Consider lowering the storage temperature below {tg}°C to ensure long-term stability.")
        else:
            st.success(f"✅ The storage temperature ({storage_temp_value}°C) is below the glass transition temperature ({tg}°C), which is optimal for vitrified samples.")
    else:
        if storage_temp_value > -80:
            st.warning(f"⚠️ The storage temperature ({storage_temp_value}°C) may lead to significant recrystallization and degradation for frozen samples.")
            st.info(f"💡 Recommendation: For long-term storage (>1 year), consider using lower temperatures such as -80°C or -196°C.")
        else:
            st.success(f"✅ The storage temperature ({storage_temp_value}°C) is suitable for frozen samples.")
    
    # Create 3D visualization of stability
    st.subheader("3D Visualization: Stability vs. Temperature and CPA Concentration")
    
    # Create parameter ranges
    temp_range = np.linspace(-200, -20, 20)
    conc_range = np.linspace(5, 60, 20)
    
    T, C = np.meshgrid(temp_range, conc_range)
    Z = np.zeros_like(T)
    
    for i in range(len(conc_range)):
        for j in range(len(temp_range)):
            # Estimate Tg for this concentration
            estimated_tg = -130 + (C[i, j] / 60) * 30
            
            # Calculate stability index
            Z[i, j] = calculate_stability_index(T[i, j], estimated_tg, is_vitrified)
    
    # Create 3D plot
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    # Use log scale for Z values
    surf = ax3d.plot_surface(T, C, np.log10(Z + 1e-10), cmap=cm.viridis, alpha=0.8)
    
    # Mark current parameters
    ax3d.scatter([storage_temp_value], [cpa_concentration], 
                [np.log10(current_stability_index + 1e-10)], 
                color='red', s=100, marker='o')
    
    ax3d.set_xlabel('Temperature (°C)')
    ax3d.set_ylabel('CPA Concentration (% v/v)')
    ax3d.set_zlabel('Log Stability Index')
    ax3d.set_title('Stability Index as a Function of Temperature and CPA Concentration')
    
    fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
    
    st.pyplot(fig3d)

with tab2:
    st.subheader("Storage Duration Effects")
    
    # Create duration range
    durations = np.linspace(0, 50, 100)  # 0 to 50 years
    
    # Calculate viability loss and crystal growth over time
    viability_losses = [calculate_viability_loss(storage_temp_value, tg, is_vitrified, d) for d in durations]
    crystal_growths = [calculate_crystal_growth(storage_temp_value, tg, d) * 100 for d in durations]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(durations, viability_losses, 'r-', linewidth=2, label='Viability Loss')
    ax.plot(durations, crystal_growths, 'b-', linewidth=2, label='Relative Crystal Size')
    
    # Mark current storage duration
    ax.axvline(x=storage_duration, color='black', linestyle=':', 
              label=f'Current Duration: {storage_duration} years')
    
    # Calculate current values
    current_viability_loss = calculate_viability_loss(storage_temp_value, tg, is_vitrified, storage_duration)
    current_crystal_growth = calculate_crystal_growth(storage_temp_value, tg, storage_duration) * 100
    
    # Mark current values
    ax.plot(storage_duration, current_viability_loss, 'ro', markersize=8)
    ax.plot(storage_duration, current_crystal_growth, 'bo', markersize=8)
    
    ax.set_xlabel('Storage Duration (years)')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Long-term Storage Effects at {storage_temp_value}°C')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)
    
    # Display current values
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Estimated Viability Loss", f"{current_viability_loss:.1f}%")
    
    with col2:
        st.metric("Relative Crystal Growth", f"{current_crystal_growth:.1f}%")
    
    # Create comparison table for different durations
    duration_options = [1, 5, 10, 20, 50]
    
    duration_data = {
        "Storage Duration (years)": duration_options,
        "Viability Loss (%)": [f"{calculate_viability_loss(storage_temp_value, tg, is_vitrified, d):.1f}" for d in duration_options],
        "Relative Crystal Size (%)": [f"{calculate_crystal_growth(storage_temp_value, tg, d) * 100:.1f}" for d in duration_options]
    }
    
    duration_df = pd.DataFrame(duration_data)
    st.table(duration_df)
    
    # Compare different storage temperatures over time
    st.subheader("Viability Comparison Across Storage Temperatures")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Storage temperature options
    temp_options = [-196, -150, -80, -20]
    temp_labels = ["Liquid Nitrogen (-196°C)", "Vapor Phase (-150°C)", 
                  "Ultra-low Freezer (-80°C)", "Standard Freezer (-20°C)"]
    colors = ['blue', 'green', 'orange', 'red']
    
    for temp, label, color in zip(temp_options, temp_labels, colors):
        # Calculate viability loss over time
        viability_losses = [100 - calculate_viability_loss(temp, tg, is_vitrified, d) for d in durations]
        ax.plot(durations, viability_losses, color=color, linewidth=2, label=label)
    
    # Mark current storage duration
    ax.axvline(x=storage_duration, color='black', linestyle=':', 
              label=f'Current Duration: {storage_duration} years')
    
    ax.set_xlabel('Storage Duration (years)')
    ax.set_ylabel('Estimated Viability (%)')
    ax.set_title('Viability Over Time at Different Storage Temperatures')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)
    
    # Create comparison table for different temperatures at current duration
    temp_duration_data = {
        "Storage Temperature": temp_labels,
        "Viability at Current Duration (%)": [f"{100 - calculate_viability_loss(temp, tg, is_vitrified, storage_duration):.1f}" for temp in temp_options],
        "Time to 50% Viability Loss (years)": []
    }
    
    # Calculate time to 50% viability loss
    for temp in temp_options:
        time_to_50 = None
        for d in np.linspace(0, 1000, 1000):  # Check up to 1000 years
            loss = calculate_viability_loss(temp, tg, is_vitrified, d)
            if loss >= 50:
                time_to_50 = d
                break
        
        if time_to_50 is not None:
            temp_duration_data["Time to 50% Viability Loss (years)"].append(f"{time_to_50:.1f}")
        else:
            temp_duration_data["Time to 50% Viability Loss (years)"].append(">1000")
    
    temp_duration_df = pd.DataFrame(temp_duration_data)
    st.table(temp_duration_df)

with tab3:
    st.subheader("Temperature Fluctuation Analysis")
    
    # Create temperature range around storage temperature
    base_temp = storage_temp_value
    temp_range = np.linspace(base_temp - temp_fluctuation, base_temp + temp_fluctuation, 100)
    
    # Calculate stability metrics across temperature range
    fluctuation_recryst_rates = [calculate_recrystallization_rate(t, tg) for t in temp_range]
    fluctuation_stability_indices = [calculate_stability_index(t, tg, is_vitrified) for t in temp_range]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(temp_range, fluctuation_recryst_rates, 'b-', linewidth=2, label='Recrystallization Rate')
    
    # Create secondary y-axis for stability index
    ax2 = ax.twinx()
    ax2.plot(temp_range, fluctuation_stability_indices, 'r-', linewidth=2, label='Stability Index')
    ax2.set_ylabel('Stability Index', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add reference lines
    ax.axvline(x=base_temp, color='black', linestyle='--', 
              label=f'Base Temperature: {base_temp}°C')
    
    if tg > base_temp - temp_fluctuation and tg < base_temp + temp_fluctuation:
        ax.axvline(x=tg, color='purple', linestyle='--', 
                  label=f'Glass Transition (Tg): {tg}°C')
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Recrystallization Rate', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title(f'Effect of Temperature Fluctuations (±{temp_fluctuation}°C)')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    st.pyplot(fig)
    
    # Calculate stability metrics at extremes
    min_temp = base_temp - temp_fluctuation
    max_temp = base_temp + temp_fluctuation
    
    min_recryst = calculate_recrystallization_rate(min_temp, tg)
    max_recryst = calculate_recrystallization_rate(max_temp, tg)
    
    min_stability = calculate_stability_index(min_temp, tg, is_vitrified)
    max_stability = calculate_stability_index(max_temp, tg, is_vitrified)
    
    # Create comparison table
    fluctuation_data = {
        "Temperature": [f"{min_temp}°C (Minimum)", f"{base_temp}°C (Base)", f"{max_temp}°C (Maximum)"],
        "Recrystallization Rate": [f"{min_recryst:.3f}", f"{current_recryst_rate:.3f}", f"{max_recryst:.3f}"],
        "Stability Index": [f"{min_stability:.3e}", f"{current_stability_index:.3e}", f"{max_stability:.3e}"],
        "Relative Stability": ["100%", f"{(current_stability_index/min_stability)*100:.1f}%", f"{(max_stability/min_stability)*100:.1f}%"]
    }
    
    fluctuation_df = pd.DataFrame(fluctuation_data)
    st.table(fluctuation_df)
    
    # Analyze temperature cycling effects
    st.subheader("Temperature Cycling Effects")
    
    # Create simulation of temperature cycling
    num_cycles = 10
    cycle_temps = []
    cycle_times = []
    
    for i in range(num_cycles):
        # Add low temperature point
        cycle_temps.append(min_temp)
        cycle_times.append(i * 2)
        
        # Add high temperature point
        cycle_temps.append(max_temp)
        cycle_times.append(i * 2 + 1)
    
    # Calculate cumulative recrystallization effect
    cumulative_effect = 0
    cycle_effects = [0]
    
    for i in range(1, len(cycle_temps)):
        # Calculate effect of this temperature change
        temp_change = abs(cycle_temps[i] - cycle_temps[i-1])
        
        # Effect is proportional to temperature change and current temperature
        if cycle_temps[i] > tg:
            # Only count changes above Tg
            effect = 0.01 * temp_change * calculate_recrystallization_rate(cycle_temps[i], tg)
            cumulative_effect += effect
        
        cycle_effects.append(cumulative_effect)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot temperature cycles
    ax1.plot(cycle_times, cycle_temps, 'b-o', linewidth=2)
    
    # Add reference line for Tg
    ax1.axhline(y=tg, color='purple', linestyle='--', 
               label=f'Glass Transition (Tg): {tg}°C')
    
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Simulated Temperature Cycling')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot cumulative effect
    ax2.plot(cycle_times, cycle_effects, 'r-o', linewidth=2)
    ax2.set_xlabel('Cycle Number')
    ax2.set_ylabel('Cumulative Recrystallization Effect')
    ax2.set_title('Cumulative Effect of Temperature Cycling')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Calculate equivalent storage time
    equivalent_time = storage_duration * (1 + cumulative_effect)
    
    st.metric("Equivalent Storage Time", f"{equivalent_time:.1f} years", 
             delta=f"+{equivalent_time - storage_duration:.1f} years")
    
    # Provide recommendations
    st.subheader("Temperature Fluctuation Recommendations")
    
    if is_vitrified:
        if max_temp > tg:
            st.warning(f"⚠️ Temperature fluctuations exceed the glass transition temperature ({tg}°C). This can cause devitrification and significantly reduce sample viability.")
            st.info(f"💡 Recommendation: Reduce temperature fluctuations or lower the base storage temperature to ensure the sample remains below Tg at all times.")
        elif max_recryst > 0:
            st.warning(f"⚠️ Temperature fluctuations may cause slow recrystallization over time, even though the sample remains vitrified.")
            st.info(f"💡 Recommendation: Minimize temperature fluctuations, especially for long-term storage.")
        else:
            st.success(f"✅ Temperature fluctuations are within a safe range below the glass transition temperature.")
    else:
        if max_recryst - min_recryst > 0.1:
            st.warning(f"⚠️ Temperature fluctuations cause significant variation in recrystallization rates, which may lead to ice crystal growth over time.")
            st.info(f"💡 Recommendation: Minimize temperature fluctuations or consider using a more stable storage method.")
        else:
            st.success(f"✅ Temperature fluctuations have minimal impact on ice crystal growth at the current storage temperature.")
    
    # Create 3D visualization of temperature fluctuation effects
    st.subheader("3D Visualization: Fluctuation Effects vs. Base Temperature")
    
    # Create parameter ranges
    base_temps = np.linspace(-200, -20, 20)
    fluctuations = np.linspace(0, 20, 20)
    
    B, F = np.meshgrid(base_temps, fluctuations)
    Z_effect = np.zeros_like(B)
    
    for i in range(len(fluctuations)):
        for j in range(len(base_temps)):
            # Calculate min and max temperatures
            min_t = B[i, j] - F[i, j]
            max_t = B[i, j] + F[i, j]
            
            # Calculate stability indices
            min_s = calculate_stability_index(min_t, tg, is_vitrified)
            max_s = calculate_stability_index(max_t, tg, is_vitrified)
            
            # Calculate effect (ratio of max to min stability)
            if min_s == 0:
                effect = 1.0  # No effect if min stability is 0
            else:
                effect = max_s / min_s
            
            Z_effect[i, j] = effect
    
    # Create 3D plot
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    # Use log scale for Z values
    surf = ax3d.plot_surface(B, F, np.log10(Z_effect), cmap=cm.viridis, alpha=0.8)
    
    # Mark current parameters
    ax3d.scatter([base_temp], [temp_fluctuation], 
                [np.log10(max_stability / min_stability)], 
                color='red', s=100, marker='o')
    
    ax3d.set_xlabel('Base Temperature (°C)')
    ax3d.set_ylabel('Temperature Fluctuation (±°C)')
    ax3d.set_zlabel('Log Effect Ratio')
    ax3d.set_title('Effect of Temperature Fluctuations at Different Base Temperatures')
    
    fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
    
    st.pyplot(fig3d)

# Explanation section
st.markdown("""
## Storage Temperature Theory in Cryopreservation

### Critical Temperatures in Cryopreservation

1. **Glass Transition Temperature (Tg)**:
   - Below Tg, the sample is in a glass state with minimal molecular mobility
   - Vitrified samples must be stored below Tg to prevent devitrification
   - Tg depends on CPA type and concentration (typically -100°C to -130°C)

2. **Devitrification Zone**:
   - Between Tg and approximately -40°C
   - Highest risk of ice crystal formation in vitrified samples
   - Rapid passage through this zone is critical during warming

3. **Recrystallization Zone**:
   - Between approximately -80°C and -20°C
   - Ice crystals can grow and remodel in frozen samples
   - Long-term storage in this zone leads to progressive damage

### Storage Options and Their Properties

1. **Liquid Nitrogen (-196°C)**:
   - Well below Tg of all CPA solutions
   - Minimal molecular mobility and biochemical reactions
   - Essentially indefinite storage time
   - Challenges: cost, need for regular refilling, risk of contamination

2. **Vapor Phase Nitrogen (-150°C to -190°C)**:
   - Still below Tg of most CPA solutions
   - Very low molecular mobility
   - Eliminates risk of cross-contamination from liquid nitrogen
   - Challenges: temperature gradients, higher cost

3. **Ultra-low Freezers (-80°C)**:
   - Above Tg of most CPA solutions
   - Slow recrystallization and biochemical degradation
   - Suitable for medium-term storage of frozen (not vitrified) samples
   - Challenges: progressive ice crystal growth, power requirements

4. **Standard Freezers (-20°C)**:
   - Significant molecular mobility
   - Rapid recrystallization and biochemical degradation
   - Only suitable for short-term storage
   - Not recommended for valuable or sensitive samples

### Degradation Mechanisms During Storage

1. **Recrystallization**:
   - Growth and fusion of ice crystals over time
   - Most rapid just below the melting point
   - Minimal below Tg
   - Causes progressive mechanical damage to cells

2. **Biochemical Degradation**:
   - Slow chemical reactions that damage biomolecules
   - Follow Arrhenius relationship (exponential decrease with decreasing temperature)
   - Still occur at -80°C but extremely slow at -196°C
   - Include oxidation, hydrolysis, and free radical damage

3. **Temperature Fluctuations**:
   - Cause repeated contraction and expansion
   - Accelerate recrystallization
   - Particularly damaging if crossing Tg
   - Cumulative effect over multiple cycles

### Practical Considerations

1. **Sample Type**:
   - Vitrified samples are more sensitive to temperature fluctuations above Tg
   - Different cell types have different sensitivities to storage conditions
   - Complex tissues may be more vulnerable to mechanical damage

2. **Storage Duration**:
   - Short-term storage (<1 year): -80°C may be adequate
   - Medium-term storage (1-10 years): -150°C recommended
   - Long-term storage (>10 years): -196°C recommended

3. **Temperature Monitoring**:
   - Regular monitoring of storage temperature
   - Alarm systems for temperature excursions
   - Backup systems for critical samples
""")