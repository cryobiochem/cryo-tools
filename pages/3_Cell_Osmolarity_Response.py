import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import io
from PIL import Image

st.set_page_config(
    page_title="Cell Osmolarity Response - Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide"
)

st.title("Cell Osmolarity Response Simulator")

st.markdown("""
This tool simulates how cells inflate or shrink in response to changes in extracellular osmolarity.
During cryopreservation, cells experience significant osmotic challenges due to:

1. Addition of cryoprotective agents (CPAs)
2. Formation of extracellular ice (which concentrates solutes)
3. Removal of CPAs during thawing

The simulation is based on the two-parameter model that describes water and solute transport across cell membranes:

$$\\frac{dV_w}{dt} = L_p A \\left[ (M_{in} - M_{out}) RT + \\sigma \\Delta P \\right]$$

Where:
- $V_w$ is the cell water volume
- $L_p$ is the hydraulic conductivity
- $A$ is the cell surface area
- $M_{in}$ and $M_{out}$ are the intracellular and extracellular osmolarities
- $R$ is the gas constant
- $T$ is the absolute temperature
- $\\sigma$ is the reflection coefficient
- $\\Delta P$ is the hydrostatic pressure difference
""")

# Create sidebar for parameters
st.sidebar.header("Cell Parameters")

cell_type = st.sidebar.selectbox(
    "Cell Type",
    ["Red Blood Cell", "Oocyte", "Hepatocyte", "Fibroblast", "Custom"]
)

# Default parameters based on cell type
if cell_type == "Red Blood Cell":
    default_radius = 4.0
    default_lp = 2.5
    default_surface_area = 140.0
    default_initial_volume = 90.0
    default_inactive_volume = 40.0
elif cell_type == "Oocyte":
    default_radius = 60.0
    default_lp = 1.0
    default_surface_area = 45000.0
    default_initial_volume = 900000.0
    default_inactive_volume = 270000.0
elif cell_type == "Hepatocyte":
    default_radius = 10.0
    default_lp = 1.5
    default_surface_area = 1250.0
    default_initial_volume = 4200.0
    default_inactive_volume = 1500.0
elif cell_type == "Fibroblast":
    default_radius = 7.5
    default_lp = 1.8
    default_surface_area = 700.0
    default_initial_volume = 1800.0
    default_inactive_volume = 600.0
else:  # Custom
    default_radius = 10.0
    default_lp = 2.0
    default_surface_area = 1250.0
    default_initial_volume = 4200.0
    default_inactive_volume = 1500.0

# Cell parameters
if cell_type == "Custom":
    cell_radius = st.sidebar.slider(
        "Cell Radius (μm)",
        min_value=1.0,
        max_value=100.0,
        value=default_radius,
        step=0.5
    )
else:
    cell_radius = default_radius

lp = st.sidebar.slider(
    "Hydraulic Conductivity (μm/min/atm)",
    min_value=0.1,
    max_value=5.0,
    value=default_lp,
    step=0.1,
    help="Membrane permeability to water"
)

surface_area = st.sidebar.slider(
    "Cell Surface Area (μm²)",
    min_value=100.0,
    max_value=50000.0,
    value=default_surface_area,
    step=100.0,
    disabled=(cell_type != "Custom")
)

initial_volume = st.sidebar.slider(
    "Initial Cell Volume (μm³)",
    min_value=100.0,
    max_value=1000000.0,
    value=default_initial_volume,
    step=100.0,
    disabled=(cell_type != "Custom")
)

inactive_volume = st.sidebar.slider(
    "Inactive Cell Volume (μm³)",
    min_value=10.0,
    max_value=300000.0,
    value=default_inactive_volume,
    step=10.0,
    help="Volume occupied by solids (non-osmotically active)"
)

# Solution parameters
st.sidebar.header("Solution Parameters")

initial_osmolarity = st.sidebar.slider(
    "Initial Osmolarity (mOsm)",
    min_value=100,
    max_value=600,
    value=300,
    step=10,
    help="Initial extracellular osmolarity (isotonic = ~300 mOsm)"
)

final_osmolarity = st.sidebar.slider(
    "Final Osmolarity (mOsm)",
    min_value=100,
    max_value=2000,
    value=600,
    step=10,
    help="Final extracellular osmolarity after addition of solutes or CPAs"
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=40.0,
    value=22.0,
    step=1.0
)

# Simulation parameters
st.sidebar.header("Simulation Parameters")

simulation_time = st.sidebar.slider(
    "Simulation Time (min)",
    min_value=1,
    max_value=30,
    value=10,
    step=1
)

# Calculate osmotic response
def calculate_osmotic_response(initial_vol, inactive_vol, initial_osm, final_osm, lp, surface_area, temp, time_points):
    # Convert temperature to Kelvin
    temp_k = temp + 273.15
    
    # Gas constant (atm·L/mol·K)
    R = 0.08206
    
    # Initial intracellular osmolarity (assuming isotonic)
    initial_internal_osm = initial_osm
    
    # Initial osmotically active volume
    initial_active_vol = initial_vol - inactive_vol
    
    # Calculate osmotic response over time
    volumes = []
    internal_osms = []
    
    # Initial conditions
    current_vol = initial_vol
    current_active_vol = initial_active_vol
    
    for t in time_points:
        # Current extracellular osmolarity (step change at t=0)
        external_osm = final_osm
        
        # Current intracellular osmolarity
        internal_osm = initial_internal_osm * initial_active_vol / current_active_vol
        internal_osms.append(internal_osm)
        
        # Store current volume
        volumes.append(current_vol)
        
        if t < time_points[-1]:  # Skip calculation for the last point
            # Calculate volume change rate (dV/dt)
            osmotic_pressure = (internal_osm - external_osm) * R * temp_k
            dv_dt = -lp * surface_area * osmotic_pressure
            
            # Update volume using Euler method
            dt = time_points[1] - time_points[0]
            current_active_vol += dv_dt * dt
            current_vol = current_active_vol + inactive_vol
            
            # Ensure volume doesn't go below inactive volume
            if current_vol < inactive_vol:
                current_vol = inactive_vol
                current_active_vol = 0
    
    return volumes, internal_osms

# Generate time points
time_points = np.linspace(0, simulation_time, 100)

# Calculate osmotic response
volumes, internal_osms = calculate_osmotic_response(
    initial_volume, 
    inactive_volume, 
    initial_osmolarity, 
    final_osmolarity, 
    lp, 
    surface_area, 
    temperature, 
    time_points
)

# Calculate normalized volumes
normalized_volumes = [v / initial_volume for v in volumes]

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Volume Change", "Cell Visualization", "Data Analysis"])

with tab1:
    st.subheader("Cell Volume Response to Osmolarity Change")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot volume change
    color = 'tab:blue'
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Cell Volume (μm³)', color=color)
    ax1.plot(time_points, volumes, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=initial_volume, color='blue', linestyle='--', alpha=0.5, label='Initial Volume')
    
    # Create a secondary y-axis for normalized volume
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Normalized Volume (V/V₀)', color=color)
    ax2.plot(time_points, normalized_volumes, color=color, linestyle=':', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='V/V₀ = 1.0')
    
    # Add a horizontal line for inactive volume
    ax1.axhline(y=inactive_volume, color='green', linestyle='--', alpha=0.5, label='Inactive Volume')
    
    # Add grid and title
    ax1.grid(True, alpha=0.3)
    plt.title(f'Cell Volume Response to Osmolarity Change ({initial_osmolarity} → {final_osmolarity} mOsm)')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Plot internal osmolarity
    st.subheader("Intracellular Osmolarity Change")
    
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(time_points, internal_osms, color='purple', linewidth=2)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Intracellular Osmolarity (mOsm)')
    ax3.axhline(y=initial_osmolarity, color='blue', linestyle='--', alpha=0.5, label='Initial Osmolarity')
    ax3.axhline(y=final_osmolarity, color='red', linestyle='--', alpha=0.5, label='Extracellular Osmolarity')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.title('Intracellular Osmolarity Change Over Time')
    
    fig2.tight_layout()
    st.pyplot(fig2)

with tab2:
    st.subheader("Cell Visualization")
    
    # Create a function to draw a cell
    def draw_cell(volume, max_volume, ax):
        # Calculate radius from volume (assuming spherical cell)
        radius = (3 * volume / (4 * np.pi))**(1/3)
        max_radius = (3 * max_volume / (4 * np.pi))**(1/3)
        
        # Create a circle
        circle = plt.Circle((0, 0), radius, fill=True, alpha=0.6, color='skyblue', edgecolor='blue')
        
        # Set axis limits based on maximum possible radius
        limit = max_radius * 1.5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # Add the circle to the plot
        ax.add_patch(circle)
        
        # Add inactive volume
        inactive_radius = (3 * inactive_volume / (4 * np.pi))**(1/3)
        inactive_circle = plt.Circle((0, 0), inactive_radius, fill=True, alpha=0.8, color='lightgray', edgecolor='gray')
        ax.add_patch(inactive_circle)
        
        # Add scale bar
        ax.plot([limit*0.5, limit*0.5 + 10], [-limit*0.8, -limit*0.8], 'k-', lw=2)
        ax.text(limit*0.5 + 5, -limit*0.75, '10 μm', ha='center')
        
        # Add volume information
        ax.text(0, -limit*0.9, f'Volume: {volume:.0f} μm³ ({volume/initial_volume*100:.1f}% of initial)', 
                ha='center', fontsize=10)
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Create animation frames
    frames = []
    for i in range(0, len(volumes), 5):  # Take every 5th point to reduce number of frames
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_cell(volumes[i], max(initial_volume, max(volumes)), ax)
        plt.title(f'Cell at t = {time_points[i]:.1f} min')
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
        plt.close()
    
    # Display animation frames as a slider
    frame_idx = st.slider("Animation Frame", 0, len(frames)-1, 0)
    st.image(frames[frame_idx])
    
    # Display key frames
    st.subheader("Key Frames")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(frames[0], caption="Initial State")
    
    with col2:
        middle_idx = len(frames) // 2
        st.image(frames[middle_idx], caption=f"t = {time_points[middle_idx*5]:.1f} min")
    
    with col3:
        st.image(frames[-1], caption=f"Final State (t = {simulation_time} min)")

with tab3:
    st.subheader("Data Analysis")
    
    # Calculate key metrics
    equilibrium_volume = volumes[-1]
    volume_change_percent = (equilibrium_volume - initial_volume) / initial_volume * 100
    time_to_50_percent = None
    
    target_volume_change = (equilibrium_volume - initial_volume) * 0.5 + initial_volume
    for i, v in enumerate(volumes):
        if (initial_volume > equilibrium_volume and v <= target_volume_change) or \
           (initial_volume < equilibrium_volume and v >= target_volume_change):
            time_to_50_percent = time_points[i]
            break
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Equilibrium Volume", f"{equilibrium_volume:.1f} μm³")
    
    with col2:
        st.metric("Volume Change", f"{volume_change_percent:.1f}%", 
                 delta=f"{'-' if volume_change_percent < 0 else '+'}{abs(volume_change_percent):.1f}%")
    
    with col3:
        if time_to_50_percent:
            st.metric("Time to 50% Response", f"{time_to_50_percent:.2f} min")
        else:
            st.metric("Time to 50% Response", "N/A")
    
    # Create data table
    data = pd.DataFrame({
        'Time (min)': time_points,
        'Volume (μm³)': volumes,
        'Normalized Volume (V/V₀)': normalized_volumes,
        'Intracellular Osmolarity (mOsm)': internal_osms
    })
    
    st.dataframe(data)
    
    # Download data option
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"osmotic_response_{cell_type.replace(' ', '_')}.csv",
        mime="text/csv"
    )
    
    # Additional analysis
    st.subheader("Osmotic Tolerance Analysis")
    
    # Calculate critical volumes
    critical_shrinkage = 0.5  # 50% of initial volume
    critical_swelling = 1.4   # 140% of initial volume
    
    min_volume_ratio = min(normalized_volumes)
    max_volume_ratio = max(normalized_volumes)
    
    # Check if cell exceeds critical volumes
    if min_volume_ratio < critical_shrinkage:
        st.warning(f"⚠️ Cell shrinks below critical volume ({critical_shrinkage*100}% of initial volume). This may cause membrane damage.")
    
    if max_volume_ratio > critical_swelling:
        st.warning(f"⚠️ Cell swells above critical volume ({critical_swelling*100}% of initial volume). This may cause lysis.")
    
    # Plot critical volumes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_points, normalized_volumes, 'b-', linewidth=2)
    ax.axhline(y=critical_shrinkage, color='red', linestyle='--', label=f'Critical Shrinkage ({critical_shrinkage*100}%)')
    ax.axhline(y=critical_swelling, color='red', linestyle='--', label=f'Critical Swelling ({critical_swelling*100}%)')
    ax.axhline(y=1.0, color='green', linestyle='--', label='Initial Volume')
    
    ax.fill_between(time_points, [critical_shrinkage]*len(time_points), [0]*len(time_points), 
                   color='red', alpha=0.2)
    ax.fill_between(time_points, [critical_swelling]*len(time_points), [2]*len(time_points), 
                   color='red', alpha=0.2)
    
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Normalized Volume (V/V₀)')
    ax.set_title('Cell Volume with Critical Tolerance Limits')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)

# Explanation section
st.markdown("""
### Osmotic Response in Cryopreservation

During cryopreservation, cells experience several osmotic challenges:

1. **CPA Addition**: When cryoprotectants are added, cells initially shrink due to the hyperosmotic extracellular environment, then gradually re-expand as CPAs permeate the cell.

2. **Freezing**: As extracellular ice forms, the remaining unfrozen solution becomes increasingly concentrated, causing cells to dehydrate.

3. **Thawing and CPA Removal**: During thawing and CPA removal, cells can swell beyond their normal volume, potentially causing damage.

### Implications for Cryopreservation Protocols

- **Cooling Rate**: Optimal cooling rates balance the risk of intracellular ice formation against excessive dehydration.
- **CPA Concentration**: Higher CPA concentrations provide better protection against ice formation but increase osmotic stress.
- **Step-wise Addition/Removal**: Gradual addition and removal of CPAs can minimize osmotic shock.

### Cell Type Differences

Different cell types have varying:
- Membrane permeability properties
- Surface area to volume ratios
- Tolerance to volume changes

These differences necessitate customized cryopreservation protocols for each cell type.
""")
