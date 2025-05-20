import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

st.set_page_config(
    page_title="Vitrification Probability Calculator - Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide"
)

st.title("Vitrification Probability Calculator")

st.markdown("""
This tool calculates the probability of vitrification versus crystallization during cryopreservation.

**Vitrification** is the transformation of a liquid into a glass-like amorphous solid without crystallization.
In cryopreservation, vitrification is often preferred over ice crystal formation, as it avoids the mechanical
damage caused by ice crystals.

This calculator models the competition between vitrification and crystallization based on:
- Cooling and warming rates
- Cryoprotectant concentration and type
- Sample volume and geometry
""")

# Create sidebar for parameters
st.sidebar.header("Solution Parameters")

cpa_type = st.sidebar.selectbox(
    "Primary Cryoprotectant",
    ["DMSO", "Ethylene Glycol", "Propylene Glycol", "Glycerol", "Custom"]
)

# Default parameters based on CPA type
if cpa_type == "DMSO":
    default_concentration = 40.0
    default_vitrification_tendency = 1.0
elif cpa_type == "Ethylene Glycol":
    default_concentration = 45.0
    default_vitrification_tendency = 1.1
elif cpa_type == "Propylene Glycol":
    default_concentration = 45.0
    default_vitrification_tendency = 0.9
elif cpa_type == "Glycerol":
    default_concentration = 50.0
    default_vitrification_tendency = 0.8
else:  # Custom
    default_concentration = 45.0
    default_vitrification_tendency = 1.0

cpa_concentration = st.sidebar.slider(
    "CPA Concentration (% w/v)",
    min_value=0.0,
    max_value=60.0,
    value=default_concentration,
    step=1.0,
    help="Concentration of cryoprotective agent"
)

vitrification_tendency = st.sidebar.slider(
    "Vitrification Tendency Factor",
    min_value=0.5,
    max_value=1.5,
    value=default_vitrification_tendency,
    step=0.1,
    help="Relative tendency of the CPA to promote vitrification (higher = better vitrification)"
)

# Add option for secondary CPA
use_secondary_cpa = st.sidebar.checkbox("Use Secondary Cryoprotectant")

if use_secondary_cpa:
    secondary_cpa_type = st.sidebar.selectbox(
        "Secondary Cryoprotectant",
        ["DMSO", "Ethylene Glycol", "Propylene Glycol", "Glycerol", "Trehalose", "Sucrose"],
        index=4  # Default to Trehalose
    )
    
    secondary_cpa_concentration = st.sidebar.slider(
        f"{secondary_cpa_type} Concentration (% w/v)",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=0.5
    )
    
    # Adjust vitrification tendency based on secondary CPA
    if secondary_cpa_type in ["Trehalose", "Sucrose"]:
        # Non-penetrating sugars enhance vitrification
        vitrification_boost = 0.2 * (secondary_cpa_concentration / 10)
    else:
        # Other CPAs provide smaller boost
        vitrification_boost = 0.1 * (secondary_cpa_concentration / 10)
    
    adjusted_vitrification_tendency = vitrification_tendency + vitrification_boost
else:
    secondary_cpa_type = None
    secondary_cpa_concentration = 0.0
    adjusted_vitrification_tendency = vitrification_tendency

# Sample parameters
st.sidebar.header("Sample Parameters")

sample_volume = st.sidebar.slider(
    "Sample Volume (μL)",
    min_value=0.1,
    max_value=1000.0,
    value=100.0,
    step=0.1,
    format="%.1f"
)

sample_geometry = st.sidebar.selectbox(
    "Sample Geometry",
    ["Straw", "Cryovial", "Open Carrier", "Droplet", "Thin Film"]
)

# Default parameters based on geometry
if sample_geometry == "Straw":
    default_surface_area_ratio = 0.6
elif sample_geometry == "Cryovial":
    default_surface_area_ratio = 0.4
elif sample_geometry == "Open Carrier":
    default_surface_area_ratio = 1.2
elif sample_geometry == "Droplet":
    default_surface_area_ratio = 1.5
else:  # Thin Film
    default_surface_area_ratio = 2.0

surface_area_ratio = st.sidebar.slider(
    "Surface Area to Volume Ratio Factor",
    min_value=0.1,
    max_value=2.0,
    value=default_surface_area_ratio,
    step=0.1,
    help="Relative surface area to volume ratio (higher = better heat transfer)"
)

# Cooling and warming parameters
st.sidebar.header("Cooling and Warming Parameters")

cooling_rate = st.sidebar.slider(
    "Cooling Rate (°C/min)",
    min_value=1.0,
    max_value=100000.0,
    value=10000.0,
    step=1.0,
    format="%.1f"
)

warming_rate = st.sidebar.slider(
    "Warming Rate (°C/min)",
    min_value=1.0,
    max_value=100000.0,
    value=20000.0,
    step=1.0,
    format="%.1f"
)

# Define vitrification model functions
def calculate_critical_cooling_rate(concentration, vitrification_tendency, volume, surface_area_ratio):
    # Base critical cooling rate (°C/min) - higher means harder to vitrify
    base_ccr = 1e6  # Pure water
    
    # Effect of CPA concentration (exponential decrease)
    concentration_effect = np.exp(-0.15 * concentration)
    
    # Effect of vitrification tendency (linear)
    tendency_effect = 1 / vitrification_tendency
    
    # Effect of volume and surface area (larger volumes harder to vitrify)
    volume_effect = (volume / 10) ** 0.5
    
    # Surface area effect (higher surface area makes vitrification easier)
    surface_effect = 1 / surface_area_ratio
    
    # Calculate critical cooling rate
    ccr = base_ccr * concentration_effect * tendency_effect * volume_effect * surface_effect
    
    return ccr

def calculate_critical_warming_rate(concentration, vitrification_tendency, volume, surface_area_ratio):
    # Critical warming rate is typically higher than critical cooling rate
    # because devitrification (crystallization during warming) is often more problematic
    ccr = calculate_critical_cooling_rate(concentration, vitrification_tendency, volume, surface_area_ratio)
    
    # Warming typically needs to be 1.5-2x faster than cooling to prevent devitrification
    cwr = ccr * 1.8
    
    return cwr

def calculate_vitrification_probability(cooling_rate, critical_cooling_rate, 
                                       warming_rate, critical_warming_rate):
    # Probability of successful cooling vitrification
    if cooling_rate <= 0:
        cooling_prob = 0
    else:
        cooling_prob = 1 / (1 + np.exp(-2 * np.log10(cooling_rate / critical_cooling_rate)))
    
    # Probability of avoiding devitrification during warming
    if warming_rate <= 0:
        warming_prob = 0
    else:
        warming_prob = 1 / (1 + np.exp(-2 * np.log10(warming_rate / critical_warming_rate)))
    
    # Overall probability (both events must succeed)
    overall_prob = cooling_prob * warming_prob
    
    return overall_prob * 100  # Convert to percentage

# Calculate critical rates and probabilities
critical_cooling_rate = calculate_critical_cooling_rate(
    cpa_concentration + secondary_cpa_concentration * 0.5,  # Adjust for secondary CPA
    adjusted_vitrification_tendency,
    sample_volume,
    surface_area_ratio
)

critical_warming_rate = calculate_critical_warming_rate(
    cpa_concentration + secondary_cpa_concentration * 0.5,
    adjusted_vitrification_tendency,
    sample_volume,
    surface_area_ratio
)

vitrification_probability = calculate_vitrification_probability(
    cooling_rate,
    critical_cooling_rate,
    warming_rate,
    critical_warming_rate
)

# Display calculated values
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Critical Cooling Rate (°C/min)", f"{critical_cooling_rate:.1f}")
with col2:
    st.metric("Critical Warming Rate (°C/min)", f"{critical_warming_rate:.1f}")
with col3:
    st.metric("Vitrification Probability (%)", f"{vitrification_probability:.1f}")

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Cooling & Warming Analysis", "CPA Concentration Analysis", "Sample Volume Analysis"])

with tab1:
    st.subheader("Cooling and Warming Rate Analysis")
    
    # Create 2D grid for cooling and warming rates
    cooling_rates = np.logspace(0, 5, 100)  # 1 to 100,000 °C/min
    warming_rates = np.logspace(0, 5, 100)  # 1 to 100,000 °C/min
    
    C, W = np.meshgrid(cooling_rates, warming_rates)
    Z = np.zeros_like(C)
    
    for i in range(len(warming_rates)):
        for j in range(len(cooling_rates)):
            Z[i, j] = calculate_vitrification_probability(
                C[i, j],
                critical_cooling_rate,
                W[i, j],
                critical_warming_rate
            )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(np.log10(C), np.log10(W), Z, 20, cmap='viridis')
    
    # Add contour lines
    contour_lines = ax.contour(np.log10(C), np.log10(W), Z, 
                              levels=[10, 50, 90, 95, 99], 
                              colors='white', linestyles='dashed')
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.0f%%')
    
    # Mark critical rates
    ax.axvline(x=np.log10(critical_cooling_rate), color='red', linestyle='--', 
              label=f'Critical Cooling Rate: {critical_cooling_rate:.1f} °C/min')
    ax.axhline(y=np.log10(critical_warming_rate), color='orange', linestyle='--',
              label=f'Critical Warming Rate: {critical_warming_rate:.1f} °C/min')
    
    # Mark current rates
    ax.plot(np.log10(cooling_rate), np.log10(warming_rate), 'ro', markersize=10)
    ax.annotate(f'{vitrification_probability:.1f}%', 
               (np.log10(cooling_rate), np.log10(warming_rate)), 
               xytext=(10, 10), 
               textcoords='offset points',
               color='white',
               fontweight='bold')
    
    # Set axis labels with custom tick labels
    ax.set_xlabel('Cooling Rate (°C/min)')
    ax.set_ylabel('Warming Rate (°C/min)')
    
    # Set custom tick positions and labels
    cooling_ticks = [0, 1, 2, 3, 4, 5]
    cooling_labels = ['1', '10', '100', '1,000', '10,000', '100,000']
    ax.set_xticks(cooling_ticks)
    ax.set_xticklabels(cooling_labels)
    
    warming_ticks = [0, 1, 2, 3, 4, 5]
    warming_labels = ['1', '10', '100', '1,000', '10,000', '100,000']
    ax.set_yticks(warming_ticks)
    ax.set_yticklabels(warming_labels)
    
    ax.set_title('Vitrification Probability (%) as a Function of Cooling and Warming Rates')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    fig.colorbar(contour, ax=ax, label='Vitrification Probability (%)')
    
    st.pyplot(fig)
    
    # Create line plots for specific warming rates
    st.subheader("Effect of Cooling Rate at Different Warming Rates")
    
    selected_warming_rates = [100, 1000, 10000, 50000]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for rate in selected_warming_rates:
        # Find closest warming rate in our grid
        rate_idx = np.argmin(np.abs(warming_rates - rate))
        
        # Extract data for this warming rate
        cooling_data = cooling_rates
        prob_data = Z[rate_idx, :]
        
        # Plot data
        ax.semilogx(cooling_data, prob_data, label=f'Warming Rate: {rate} °C/min')
    
    # Mark critical cooling rate
    ax.axvline(x=critical_cooling_rate, color='red', linestyle='--', 
              label=f'Critical Cooling Rate: {critical_cooling_rate:.1f} °C/min')
    
    # Mark current cooling rate
    ax.axvline(x=cooling_rate, color='green', linestyle='-', 
              label=f'Current Cooling Rate: {cooling_rate:.1f} °C/min')
    
    ax.set_xlabel('Cooling Rate (°C/min)')
    ax.set_ylabel('Vitrification Probability (%)')
    ax.set_title('Vitrification Probability vs. Cooling Rate')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)

with tab2:
    st.subheader("CPA Concentration Analysis")
    
    # Create concentration range
    concentrations = np.linspace(0, 60, 100)
    
    # Calculate vitrification probability for each concentration
    cooling_probs = []
    warming_probs = []
    overall_probs = []
    
    for conc in concentrations:
        # Calculate critical rates for this concentration
        ccr = calculate_critical_cooling_rate(
            conc + secondary_cpa_concentration * 0.5,
            adjusted_vitrification_tendency,
            sample_volume,
            surface_area_ratio
        )
        
        cwr = calculate_critical_warming_rate(
            conc + secondary_cpa_concentration * 0.5,
            adjusted_vitrification_tendency,
            sample_volume,
            surface_area_ratio
        )
        
        # Calculate probabilities
        if cooling_rate <= 0:
            cooling_prob = 0
        else:
            cooling_prob = 1 / (1 + np.exp(-2 * np.log10(cooling_rate / ccr)))
        
        if warming_rate <= 0:
            warming_prob = 0
        else:
            warming_prob = 1 / (1 + np.exp(-2 * np.log10(warming_rate / cwr)))
        
        overall_prob = cooling_prob * warming_prob * 100
        
        cooling_probs.append(cooling_prob * 100)
        warming_probs.append(warming_prob * 100)
        overall_probs.append(overall_prob)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(concentrations, cooling_probs, 'b-', label='Cooling Vitrification')
    ax.plot(concentrations, warming_probs, 'r-', label='Warming Stability')
    ax.plot(concentrations, overall_probs, 'g-', linewidth=2, label='Overall Probability')
    
    # Mark current concentration
    ax.axvline(x=cpa_concentration, color='black', linestyle='--', 
              label=f'Current Concentration: {cpa_concentration}% w/v')
    
    # Mark minimum concentration for high probability
    min_conc_idx = np.argmax(np.array(overall_probs) >= 95)
    if min_conc_idx > 0 and min_conc_idx < len(concentrations):
        min_conc = concentrations[min_conc_idx]
        ax.axvline(x=min_conc, color='purple', linestyle=':', 
                  label=f'Min. Conc. for 95% Success: {min_conc:.1f}% w/v')
    
    ax.set_xlabel('CPA Concentration (% w/v)')
    ax.set_ylabel('Probability (%)')
    ax.set_title(f'Vitrification Probability vs. {cpa_type} Concentration')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)
    
    # Create 3D visualization of concentration, cooling rate, and probability
    st.subheader("3D Visualization: Concentration-Cooling Rate-Probability")
    
    # Create parameter ranges for 3D plot
    conc_range = np.linspace(0, 60, 20)
    cool_range = np.logspace(1, 5, 20)  # 10 to 100,000 °C/min
    
    C_mesh, R_mesh = np.meshgrid(conc_range, cool_range)
    Z_prob = np.zeros_like(C_mesh)
    
    for i in range(len(cool_range)):
        for j in range(len(conc_range)):
            # Calculate critical cooling rate for this concentration
            ccr = calculate_critical_cooling_rate(
                C_mesh[i, j] + secondary_cpa_concentration * 0.5,
                adjusted_vitrification_tendency,
                sample_volume,
                surface_area_ratio
            )
            
            # Calculate cooling probability
            if R_mesh[i, j] <= 0:
                cooling_prob = 0
            else:
                cooling_prob = 1 / (1 + np.exp(-2 * np.log10(R_mesh[i, j] / ccr)))
            
            # Calculate critical warming rate
            cwr = calculate_critical_warming_rate(
                C_mesh[i, j] + secondary_cpa_concentration * 0.5,
                adjusted_vitrification_tendency,
                sample_volume,
                surface_area_ratio
            )
            
            # Calculate warming probability
            if warming_rate <= 0:
                warming_prob = 0
            else:
                warming_prob = 1 / (1 + np.exp(-2 * np.log10(warming_rate / cwr)))
            
            # Overall probability
            Z_prob[i, j] = cooling_prob * warming_prob * 100
    
    # Create 3D plot
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    surf = ax3d.plot_surface(C_mesh, np.log10(R_mesh), Z_prob, cmap=cm.viridis, alpha=0.8)
    
    # Mark current point
    ax3d.scatter([cpa_concentration], [np.log10(cooling_rate)], [vitrification_probability], 
                color='red', s=100, marker='o')
    
    # Set axis labels with custom tick labels
    ax3d.set_xlabel('CPA Concentration (% w/v)')
    ax3d.set_ylabel('Cooling Rate (°C/min)')
    ax3d.set_zlabel('Vitrification Probability (%)')
    
    # Set custom tick positions and labels for cooling rate
    rate_ticks = [1, 2, 3, 4, 5]
    rate_labels = ['10', '100', '1,000', '10,000', '100,000']
    ax3d.set_yticks(rate_ticks)
    ax3d.set_yticklabels(rate_labels)
    
    ax3d.set_title('Vitrification Probability as a Function of Concentration and Cooling Rate')
    
    fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
    
    st.pyplot(fig3d)

with tab3:
    st.subheader("Sample Volume and Geometry Analysis")
    
    # Create volume range
    volumes = np.logspace(-1, 3, 100)  # 0.1 to 1000 μL
    
    # Calculate vitrification probability for each volume
    volume_probs = []
    
    for vol in volumes:
        # Calculate critical rates for this volume
        ccr = calculate_critical_cooling_rate(
            cpa_concentration + secondary_cpa_concentration * 0.5,
            adjusted_vitrification_tendency,
            vol,
            surface_area_ratio
        )
        
        cwr = calculate_critical_warming_rate(
            cpa_concentration + secondary_cpa_concentration * 0.5,
            adjusted_vitrification_tendency,
            vol,
            surface_area_ratio
        )
        
        # Calculate overall probability
        prob = calculate_vitrification_probability(
            cooling_rate,
            ccr,
            warming_rate,
            cwr
        )
        
        volume_probs.append(prob)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogx(volumes, volume_probs, 'b-', linewidth=2)
    
    # Mark current volume
    ax.axvline(x=sample_volume, color='red', linestyle='--', 
              label=f'Current Volume: {sample_volume} μL')
    
    # Mark maximum volume for high probability
    max_vol_idx = np.argmax(np.array(volume_probs) < 95)
    if max_vol_idx > 0 and max_vol_idx < len(volumes):
        max_vol = volumes[max_vol_idx]
        ax.axvline(x=max_vol, color='green', linestyle=':', 
                  label=f'Max. Vol. for 95% Success: {max_vol:.1f} μL')
    
    ax.set_xlabel('Sample Volume (μL)')
    ax.set_ylabel('Vitrification Probability (%)')
    ax.set_title(f'Vitrification Probability vs. Sample Volume')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)
    
    # Compare different geometries
    st.subheader("Comparison of Sample Geometries")
    
    # Define geometries and their surface area ratios
    geometries = {
        "Straw": 0.6,
        "Cryovial": 0.4,
        "Open Carrier": 1.2,
        "Droplet": 1.5,
        "Thin Film": 2.0
    }
    
    # Calculate probabilities for each geometry at different volumes
    geometry_data = {}
    
    for geom, sar in geometries.items():
        probs = []
        for vol in volumes:
            ccr = calculate_critical_cooling_rate(
                cpa_concentration + secondary_cpa_concentration * 0.5,
                adjusted_vitrification_tendency,
                vol,
                sar
            )
            
            cwr = calculate_critical_warming_rate(
                cpa_concentration + secondary_cpa_concentration * 0.5,
                adjusted_vitrification_tendency,
                vol,
                sar
            )
            
            prob = calculate_vitrification_probability(
                cooling_rate,
                ccr,
                warming_rate,
                cwr
            )
            
            probs.append(prob)
        
        geometry_data[geom] = probs
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, (geom, probs) in enumerate(geometry_data.items()):
        ax.semilogx(volumes, probs, color=colors[i], linewidth=2, label=geom)
    
    # Mark current volume
    ax.axvline(x=sample_volume, color='black', linestyle='--', 
              label=f'Current Volume: {sample_volume} μL')
    
    ax.set_xlabel('Sample Volume (μL)')
    ax.set_ylabel('Vitrification Probability (%)')
    ax.set_title('Vitrification Probability vs. Sample Volume for Different Geometries')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)
    
    # Create a table of maximum volumes for each geometry
    max_volumes = {}
    
    for geom, probs in geometry_data.items():
        max_vol_idx = np.argmax(np.array(probs) < 95)
        if max_vol_idx > 0 and max_vol_idx < len(volumes):
            max_vol = volumes[max_vol_idx]
        else:
            max_vol = np.nan
        
        max_volumes[geom] = max_vol
    
    # Create dataframe
    geometry_df = pd.DataFrame({
        "Geometry": list(geometries.keys()),
        "Surface Area Ratio Factor": list(geometries.values()),
        "Maximum Volume for 95% Success (μL)": [max_volumes[g] for g in geometries.keys()]
    })
    
    st.table(geometry_df)

# Explanation section
st.markdown("""
## Vitrification Theory in Cryopreservation

### Vitrification vs. Slow Freezing

**Vitrification** is the solidification of a liquid into a glass-like amorphous state without ice crystal formation.
This is achieved by:
1. Using high concentrations of cryoprotectants
2. Applying ultra-rapid cooling rates
3. Optimizing sample geometry for heat transfer

In contrast, **slow freezing** allows controlled ice formation in the extracellular space while dehydrating cells to prevent intracellular ice.

### Critical Factors for Successful Vitrification

1. **Critical Cooling Rate (CCR)**:
   - The minimum cooling rate required to avoid ice nucleation and growth
   - Depends on CPA concentration, sample volume, and geometry
   - Typically ranges from 10°C/min to 100,000°C/min depending on conditions

2. **Critical Warming Rate (CWR)**:
   - The minimum warming rate required to avoid devitrification (crystallization during warming)
   - Often more critical than cooling rate
   - Typically 1.5-2x higher than the critical cooling rate

3. **CPA Concentration**:
   - Higher concentrations lower the critical cooling/warming rates
   - But increase toxicity and osmotic stress
   - Optimal concentration balances vitrification ability with toxicity

4. **Sample Volume and Geometry**:
   - Smaller volumes are easier to vitrify
   - High surface area to volume ratios improve heat transfer
   - Thin, flat geometries vitrify more readily than spherical ones

### Practical Applications

- **Oocyte/Embryo Vitrification**: Uses minimal volumes on specialized carriers
- **Cell Suspension Vitrification**: Uses straws or open pulled straws
- **Tissue Vitrification**: Requires higher CPA concentrations or step-wise loading
- **Organ Vitrification**: Remains challenging due to heat transfer limitations

### Vitrification Mixtures

Combining multiple CPAs often improves vitrification by:
- Reducing the required concentration of each component
- Combining different protective mechanisms
- Balancing permeability and glass-forming properties

Common effective mixtures include:
- DMSO + Ethylene Glycol + Sucrose
- Glycerol + Ethylene Glycol + Polymers
- Propylene Glycol + Trehalose
""")