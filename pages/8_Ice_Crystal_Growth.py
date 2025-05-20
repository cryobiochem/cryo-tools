import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import pandas as pd
from matplotlib.animation import FuncAnimation
import io
from PIL import Image

st.set_page_config(
    page_title="Ice Crystal Growth Visualizer - Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide"
)

st.title("Ice Crystal Growth Visualizer")

st.markdown("""
This tool simulates and visualizes ice crystal growth patterns during freezing and thawing.
Ice crystal formation is a critical factor in cryopreservation outcomes, as it can cause
mechanical damage to cells and tissues.

The simulator models:
- Nucleation and growth of ice crystals
- Effects of cooling rate on crystal size and distribution
- Influence of cryoprotectants on ice formation
- Recrystallization during warming
""")

# Create sidebar for parameters
st.sidebar.header("Freezing Parameters")

cooling_rate = st.sidebar.slider(
    "Cooling Rate (°C/min)",
    min_value=0.1,
    max_value=1000.0,
    value=10.0,
    step=0.1,
    format="%.1f"
)

cpa_type = st.sidebar.selectbox(
    "Cryoprotectant",
    ["None", "DMSO", "Glycerol", "Ethylene Glycol", "Propylene Glycol"]
)

# Default parameters based on CPA type
if cpa_type == "None":
    default_concentration = 0.0
    default_nucleation_temp = -5.0
elif cpa_type == "DMSO":
    default_concentration = 10.0
    default_nucleation_temp = -8.0
elif cpa_type == "Glycerol":
    default_concentration = 10.0
    default_nucleation_temp = -7.0
elif cpa_type == "Ethylene Glycol":
    default_concentration = 10.0
    default_nucleation_temp = -7.5
else:  # Propylene Glycol
    default_concentration = 10.0
    default_nucleation_temp = -8.5

cpa_concentration = st.sidebar.slider(
    "CPA Concentration (% v/v)",
    min_value=0.0,
    max_value=40.0,
    value=default_concentration,
    step=1.0,
    disabled=(cpa_type == "None")
)

nucleation_temp = st.sidebar.slider(
    "Nucleation Temperature (°C)",
    min_value=-20.0,
    max_value=-1.0,
    value=default_nucleation_temp,
    step=0.5,
    help="Temperature at which ice nucleation begins"
)

# Simulation parameters
st.sidebar.header("Simulation Parameters")

simulation_size = st.sidebar.slider(
    "Simulation Grid Size",
    min_value=50,
    max_value=200,
    value=100,
    step=10
)

simulation_time = st.sidebar.slider(
    "Simulation Time (min)",
    min_value=1,
    max_value=30,
    value=5,
    step=1
)

# Define ice crystal growth model functions
def initialize_grid(size, nucleation_sites):
    """Initialize the simulation grid with nucleation sites."""
    grid = np.zeros((size, size))
    
    # Add nucleation sites
    for _ in range(nucleation_sites):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        grid[x, y] = 1
    
    return grid

def calculate_nucleation_sites(cooling_rate, cpa_concentration, grid_size):
    """Calculate number of nucleation sites based on cooling rate and CPA concentration."""
    # Base number of sites (higher for faster cooling)
    base_sites = int(np.sqrt(cooling_rate) * grid_size / 20)
    
    # Reduce sites with CPA (CPAs inhibit nucleation)
    cpa_factor = np.exp(-0.05 * cpa_concentration)
    
    sites = int(base_sites * cpa_factor)
    
    # Ensure at least one nucleation site
    return max(1, sites)

def grow_crystals(grid, growth_probability):
    """Grow ice crystals from existing nucleation sites."""
    size = grid.shape[0]
    new_grid = grid.copy()
    
    # For each cell in the grid
    for i in range(size):
        for j in range(size):
            # If the cell is not frozen
            if grid[i, j] == 0:
                # Check neighbors
                neighbors_frozen = 0
                
                # Check 8 surrounding cells (with boundary checking)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size and grid[ni, nj] > 0:
                            neighbors_frozen += 1
                
                # Probability of freezing increases with number of frozen neighbors
                if neighbors_frozen > 0:
                    freeze_prob = growth_probability * (neighbors_frozen / 8)
                    if np.random.random() < freeze_prob:
                        new_grid[i, j] = grid[i, j-1] + 0.1 if j > 0 and grid[i, j-1] > 0 else 1
    
    return new_grid

def calculate_growth_probability(cooling_rate, cpa_concentration, temperature):
    """Calculate growth probability based on cooling rate, CPA, and temperature."""
    # Base probability (higher for slower cooling)
    base_prob = 0.3 * np.exp(-0.01 * cooling_rate)
    
    # Reduce probability with CPA (CPAs inhibit growth)
    cpa_factor = np.exp(-0.03 * cpa_concentration)
    
    # Temperature effect (growth slows as temperature decreases)
    temp_factor = np.exp(0.1 * (temperature + 20))  # +20 to make positive for temps down to -20°C
    
    prob = base_prob * cpa_factor * temp_factor
    
    # Ensure probability is in [0, 1]
    return min(1.0, max(0.0, prob))

def simulate_recrystallization(grid, warming_rate, time_fraction):
    """Simulate recrystallization during warming."""
    size = grid.shape[0]
    new_grid = grid.copy()
    
    # Recrystallization is more pronounced with slower warming
    recrystallization_factor = 0.2 * np.exp(-0.01 * warming_rate)
    
    # Recrystallization increases with time during warming
    time_effect = time_fraction  # 0 to 1
    
    # For each cell in the grid
    for i in range(size):
        for j in range(size):
            # If the cell is frozen
            if grid[i, j] > 0:
                # Check for small crystals (indicated by higher values from growth phase)
                if grid[i, j] > 1.5:
                    # Small crystals may melt
                    if np.random.random() < recrystallization_factor * time_effect:
                        new_grid[i, j] = 0
                else:
                    # Larger crystals may grow even larger
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            
                            ni, nj = i + di, j + dj
                            if 0 <= ni < size and 0 <= nj < size and grid[ni, nj] == 0:
                                if np.random.random() < recrystallization_factor * time_effect * 0.5:
                                    new_grid[ni, nj] = 1
    
    return new_grid

# Calculate parameters for simulation
nucleation_sites = calculate_nucleation_sites(cooling_rate, cpa_concentration, simulation_size)

# Calculate freezing time (time to reach -20°C from nucleation_temp)
freezing_time = (nucleation_temp - (-20)) / cooling_rate  # minutes

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Crystal Growth Simulation", "Cooling Rate Effects", "CPA Effects"])

with tab1:
    st.subheader("Ice Crystal Growth Simulation")
    
    # Initialize simulation
    np.random.seed(42)  # For reproducibility
    initial_grid = initialize_grid(simulation_size, nucleation_sites)
    
    # Run simulation
    time_points = np.linspace(0, simulation_time, 6)  # 6 time points
    simulation_frames = []
    
    current_grid = initial_grid.copy()
    simulation_frames.append(current_grid.copy())
    
    for t in range(1, len(time_points)):
        # Calculate current temperature
        current_temp = nucleation_temp - (cooling_rate * time_points[t])
        current_temp = max(current_temp, -20.0)  # Don't go below -20°C
        
        # Calculate growth probability
        growth_prob = calculate_growth_probability(cooling_rate, cpa_concentration, current_temp)
        
        # Grow crystals
        current_grid = grow_crystals(current_grid, growth_prob)
        simulation_frames.append(current_grid.copy())
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (frame, time) in enumerate(zip(simulation_frames, time_points)):
        # Calculate current temperature
        current_temp = nucleation_temp - (cooling_rate * time)
        current_temp = max(current_temp, -20.0)
        
        # Calculate ice fraction
        ice_fraction = np.sum(frame > 0) / (simulation_size ** 2) * 100
        
        # Create custom colormap: blue for ice, white for water
        cmap = colors.ListedColormap(['white', 'lightblue', 'blue', 'darkblue'])
        bounds = [0, 0.1, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot frame
        im = axes[i].imshow(frame, cmap=cmap, norm=norm, interpolation='none')
        axes[i].set_title(f't = {time:.1f} min, T = {current_temp:.1f}°C\nIce: {ice_fraction:.1f}%')
        axes[i].axis('off')
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Create animation frames
    st.subheader("Animation of Ice Crystal Growth")
    
    # Run a more detailed simulation for animation
    animation_frames = []
    animation_time_points = np.linspace(0, simulation_time, 20)  # 20 time points
    
    current_grid = initial_grid.copy()
    
    for t in range(len(animation_time_points)):
        # Calculate current temperature
        current_temp = nucleation_temp - (cooling_rate * animation_time_points[t])
        current_temp = max(current_temp, -20.0)
        
        # Calculate growth probability
        growth_prob = calculate_growth_probability(cooling_rate, cpa_concentration, current_temp)
        
        # Grow crystals
        current_grid = grow_crystals(current_grid, growth_prob)
        
        # Create figure for this frame
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Calculate ice fraction
        ice_fraction = np.sum(current_grid > 0) / (simulation_size ** 2) * 100
        
        # Create custom colormap
        cmap = colors.ListedColormap(['white', 'lightblue', 'blue', 'darkblue'])
        bounds = [0, 0.1, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot frame
        im = ax.imshow(current_grid, cmap=cmap, norm=norm, interpolation='none')
        ax.set_title(f't = {animation_time_points[t]:.1f} min, T = {current_temp:.1f}°C\nIce: {ice_fraction:.1f}%')
        ax.axis('off')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        animation_frames.append(img)
        plt.close()
    
    # Display animation frames as a slider
    frame_idx = st.slider("Animation Frame", 0, len(animation_frames)-1, 0)
    st.image(animation_frames[frame_idx])
    
    # Display key metrics
    final_ice_fraction = np.sum(simulation_frames[-1] > 0) / (simulation_size ** 2) * 100
    
    # Calculate average crystal size
    from scipy import ndimage
    
    # Label connected components (crystals)
    labeled_array, num_crystals = ndimage.label(simulation_frames[-1] > 0)
    
    # Calculate sizes
    crystal_sizes = []
    for i in range(1, num_crystals + 1):
        crystal_sizes.append(np.sum(labeled_array == i))
    
    if crystal_sizes:
        avg_crystal_size = np.mean(crystal_sizes)
        max_crystal_size = np.max(crystal_sizes)
    else:
        avg_crystal_size = 0
        max_crystal_size = 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Ice Fraction", f"{final_ice_fraction:.1f}%")
    with col2:
        st.metric("Number of Crystals", f"{num_crystals}")
    with col3:
        st.metric("Average Crystal Size", f"{avg_crystal_size:.1f} pixels")
    
    # Create crystal size distribution histogram
    if crystal_sizes:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(crystal_sizes, bins=20, alpha=0.7, color='blue')
        ax.axvline(x=avg_crystal_size, color='red', linestyle='--', 
                  label=f'Average Size: {avg_crystal_size:.1f}')
        ax.set_xlabel('Crystal Size (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Ice Crystal Size Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)

with tab2:
    st.subheader("Effect of Cooling Rate on Ice Crystal Formation")
    
    # Define cooling rates to compare
    test_cooling_rates = [1.0, 10.0, 100.0, 1000.0]
    
    # Run simulations for each cooling rate
    cooling_rate_results = {}
    
    for rate in test_cooling_rates:
        # Calculate nucleation sites for this rate
        sites = calculate_nucleation_sites(rate, cpa_concentration, simulation_size)
        
        # Initialize grid
        np.random.seed(42)  # Same seed for fair comparison
        grid = initialize_grid(simulation_size, sites)
        
        # Run simulation
        current_grid = grid.copy()
        
        # Use fixed number of steps for fair comparison
        for t in range(5):  # 5 steps
            # Calculate current temperature
            current_temp = nucleation_temp - (rate * t / 5 * simulation_time)
            current_temp = max(current_temp, -20.0)
            
            # Calculate growth probability
            growth_prob = calculate_growth_probability(rate, cpa_concentration, current_temp)
            
            # Grow crystals
            current_grid = grow_crystals(current_grid, growth_prob)
        
        # Store results
        cooling_rate_results[rate] = current_grid.copy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, rate in enumerate(test_cooling_rates):
        # Calculate ice fraction
        ice_fraction = np.sum(cooling_rate_results[rate] > 0) / (simulation_size ** 2) * 100
        
        # Label connected components (crystals)
        labeled_array, num_crystals = ndimage.label(cooling_rate_results[rate] > 0)
        
        # Create custom colormap
        cmap = colors.ListedColormap(['white', 'lightblue', 'blue', 'darkblue'])
        bounds = [0, 0.1, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot frame
        im = axes[i].imshow(cooling_rate_results[rate], cmap=cmap, norm=norm, interpolation='none')
        axes[i].set_title(f'Cooling Rate: {rate} °C/min\nIce: {ice_fraction:.1f}%, Crystals: {num_crystals}')
        axes[i].axis('off')
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Create comparison metrics
    cooling_metrics = {
        "Cooling Rate (°C/min)": [],
        "Ice Fraction (%)": [],
        "Number of Crystals": [],
        "Average Crystal Size (pixels)": []
    }
    
    for rate in test_cooling_rates:
        # Calculate ice fraction
        ice_fraction = np.sum(cooling_rate_results[rate] > 0) / (simulation_size ** 2) * 100
        
        # Label connected components (crystals)
        labeled_array, num_crystals = ndimage.label(cooling_rate_results[rate] > 0)
        
        # Calculate average crystal size
        crystal_sizes = []
        for i in range(1, num_crystals + 1):
            crystal_sizes.append(np.sum(labeled_array == i))
        
        avg_size = np.mean(crystal_sizes) if crystal_sizes else 0
        
        # Store metrics
        cooling_metrics["Cooling Rate (°C/min)"].append(rate)
        cooling_metrics["Ice Fraction (%)"].append(f"{ice_fraction:.1f}")
        cooling_metrics["Number of Crystals"].append(num_crystals)
        cooling_metrics["Average Crystal Size (pixels)"].append(f"{avg_size:.1f}")
    
    # Create dataframe
    cooling_df = pd.DataFrame(cooling_metrics)
    st.table(cooling_df)
    
    # Create plots of metrics vs cooling rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract numeric data
    rates = cooling_metrics["Cooling Rate (°C/min)"]
    ice_fractions = [float(x) for x in cooling_metrics["Ice Fraction (%)"]]
    num_crystals = cooling_metrics["Number of Crystals"]
    avg_sizes = [float(x) for x in cooling_metrics["Average Crystal Size (pixels)"]]
    
    # Plot ice fraction
    ax1.semilogx(rates, ice_fractions, 'b-o', linewidth=2)
    ax1.set_xlabel('Cooling Rate (°C/min)')
    ax1.set_ylabel('Ice Fraction (%)')
    ax1.set_title('Ice Fraction vs. Cooling Rate')
    ax1.grid(True, alpha=0.3)
    
    # Plot number of crystals and average size
    ax2.semilogx(rates, num_crystals, 'r-o', linewidth=2)
    ax2.set_xlabel('Cooling Rate (°C/min)')
    ax2.set_ylabel('Number of Crystals', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax2b = ax2.twinx()
    ax2b.semilogx(rates, avg_sizes, 'g--s', linewidth=2)
    ax2b.set_ylabel('Average Crystal Size (pixels)', color='g')
    ax2b.tick_params(axis='y', labelcolor='g')
    
    ax2.set_title('Crystal Number and Size vs. Cooling Rate')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Observations on Cooling Rate Effects:
    
    1. **Fast Cooling (>100°C/min)**:
       - More numerous, smaller ice crystals
       - Higher nucleation rate, lower growth rate
       - Intracellular ice formation more likely
    
    2. **Slow Cooling (<10°C/min)**:
       - Fewer, larger ice crystals
       - Lower nucleation rate, higher growth rate
       - Extracellular ice formation predominant
    
    3. **Optimal Cooling Rate**:
       - Depends on cell type and CPA concentration
       - Balances solution effects damage and intracellular ice formation
       - Often in the range of 1-10°C/min for many cell types without CPAs
    """)

with tab3:
    st.subheader("Effect of Cryoprotectants on Ice Crystal Formation")
    
    # Define CPA concentrations to compare
    test_cpa_concentrations = [0.0, 10.0, 20.0, 40.0]
    
    # Run simulations for each CPA concentration
    cpa_results = {}
    
    for conc in test_cpa_concentrations:
        # Calculate nucleation sites for this concentration
        sites = calculate_nucleation_sites(cooling_rate, conc, simulation_size)
        
        # Initialize grid
        np.random.seed(42)  # Same seed for fair comparison
        grid = initialize_grid(simulation_size, sites)
        
        # Run simulation
        current_grid = grid.copy()
        
        # Use fixed number of steps for fair comparison
        for t in range(5):  # 5 steps
            # Calculate current temperature
            current_temp = nucleation_temp - (cooling_rate * t / 5 * simulation_time)
            current_temp = max(current_temp, -20.0)
            
            # Calculate growth probability
            growth_prob = calculate_growth_probability(cooling_rate, conc, current_temp)
            
            # Grow crystals
            current_grid = grow_crystals(current_grid, growth_prob)
        
        # Store results
        cpa_results[conc] = current_grid.copy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, conc in enumerate(test_cpa_concentrations):
        # Calculate ice fraction
        ice_fraction = np.sum(cpa_results[conc] > 0) / (simulation_size ** 2) * 100
        
        # Label connected components (crystals)
        labeled_array, num_crystals = ndimage.label(cpa_results[conc] > 0)
        
        # Create custom colormap
        cmap = colors.ListedColormap(['white', 'lightblue', 'blue', 'darkblue'])
        bounds = [0, 0.1, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot frame
        im = axes[i].imshow(cpa_results[conc], cmap=cmap, norm=norm, interpolation='none')
        axes[i].set_title(f'CPA Concentration: {conc}% v/v\nIce: {ice_fraction:.1f}%, Crystals: {num_crystals}')
        axes[i].axis('off')
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Create comparison metrics
    cpa_metrics = {
        "CPA Concentration (% v/v)": [],
        "Ice Fraction (%)": [],
        "Number of Crystals": [],
        "Average Crystal Size (pixels)": []
    }
    
    for conc in test_cpa_concentrations:
        # Calculate ice fraction
        ice_fraction = np.sum(cpa_results[conc] > 0) / (simulation_size ** 2) * 100
        
        # Label connected components (crystals)
        labeled_array, num_crystals = ndimage.label(cpa_results[conc] > 0)
        
        # Calculate average crystal size
        crystal_sizes = []
        for i in range(1, num_crystals + 1):
            crystal_sizes.append(np.sum(labeled_array == i))
        
        avg_size = np.mean(crystal_sizes) if crystal_sizes else 0
        
        # Store metrics
        cpa_metrics["CPA Concentration (% v/v)"].append(conc)
        cpa_metrics["Ice Fraction (%)"].append(f"{ice_fraction:.1f}")
        cpa_metrics["Number of Crystals"].append(num_crystals)
        cpa_metrics["Average Crystal Size (pixels)"].append(f"{avg_size:.1f}")
    
    # Create dataframe
    cpa_df = pd.DataFrame(cpa_metrics)
    st.table(cpa_df)
    
    # Create plots of metrics vs CPA concentration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract numeric data
    concentrations = cpa_metrics["CPA Concentration (% v/v)"]
    ice_fractions = [float(x) for x in cpa_metrics["Ice Fraction (%)"]]
    num_crystals = cpa_metrics["Number of Crystals"]
    avg_sizes = [float(x) for x in cpa_metrics["Average Crystal Size (pixels)"]]
    
    # Plot ice fraction
    ax1.plot(concentrations, ice_fractions, 'b-o', linewidth=2)
    ax1.set_xlabel('CPA Concentration (% v/v)')
    ax1.set_ylabel('Ice Fraction (%)')
    ax1.set_title('Ice Fraction vs. CPA Concentration')
    ax1.grid(True, alpha=0.3)
    
    # Plot number of crystals and average size
    ax2.plot(concentrations, num_crystals, 'r-o', linewidth=2)
    ax2.set_xlabel('CPA Concentration (% v/v)')
    ax2.set_ylabel('Number of Crystals', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax2b = ax2.twinx()
    ax2b.plot(concentrations, avg_sizes, 'g--s', linewidth=2)
    ax2b.set_ylabel('Average Crystal Size (pixels)', color='g')
    ax2b.tick_params(axis='y', labelcolor='g')
    
    ax2.set_title('Crystal Number and Size vs. CPA Concentration')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Simulate recrystallization during warming
    st.subheader("Recrystallization During Warming")
    
    # Use the final state from the main simulation
    initial_warming_grid = simulation_frames[-1].copy()
    
    # Run warming simulation
    warming_frames = [initial_warming_grid.copy()]
    warming_time_points = np.linspace(0, simulation_time/2, 3)  # 3 time points
    
    current_warming_grid = initial_warming_grid.copy()
    
    for t in range(1, len(warming_time_points)):
        # Calculate time fraction (0 to 1)
        time_fraction = t / (len(warming_time_points) - 1)
        
        # Simulate recrystallization
        # Assume warming rate is 2x cooling rate
        warming_rate = cooling_rate * 2
        current_warming_grid = simulate_recrystallization(current_warming_grid, warming_rate, time_fraction)
        warming_frames.append(current_warming_grid.copy())
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (frame, time) in enumerate(zip(warming_frames, warming_time_points)):
        # Calculate current temperature during warming
        current_temp = -20.0 + (cooling_rate * 2 * time)
        current_temp = min(current_temp, nucleation_temp)
        
        # Calculate ice fraction
        ice_fraction = np.sum(frame > 0) / (simulation_size ** 2) * 100
        
        # Label connected components (crystals)
        labeled_array, num_crystals = ndimage.label(frame > 0)
        
        # Create custom colormap
        cmap = colors.ListedColormap(['white', 'lightblue', 'blue', 'darkblue'])
        bounds = [0, 0.1, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot frame
        im = axes[i].imshow(frame, cmap=cmap, norm=norm, interpolation='none')
        axes[i].set_title(f'Warming: t = {time:.1f} min, T = {current_temp:.1f}°C\nIce: {ice_fraction:.1f}%, Crystals: {num_crystals}')
        axes[i].axis('off')
    
    fig.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Observations on Cryoprotectant Effects:
    
    1. **Colligative Effects**:
       - CPAs lower the freezing point
       - Reduce the amount of ice formed
       - Increase the unfrozen fraction
    
    2. **Kinetic Effects**:
       - CPAs inhibit nucleation
       - Slow crystal growth
       - Modify crystal morphology
    
    3. **Recrystallization During Warming**:
       - Small crystals melt first
       - Larger crystals grow at the expense of smaller ones
       - Slower warming rates increase recrystallization
       - CPAs can inhibit recrystallization
    
    4. **Vitrification at High Concentrations**:
       - At very high CPA concentrations (>40%), ice formation may be completely prevented
       - The solution forms a glass-like amorphous solid instead of crystals
    """)

# Explanation section
st.markdown("""
## Ice Crystal Formation Theory in Cryopreservation

### Ice Formation Process

Ice formation occurs in two main steps:

1. **Nucleation**: Formation of initial ice nuclei
   - Homogeneous nucleation: Spontaneous formation in pure water
   - Heterogeneous nucleation: Formation facilitated by particles or surfaces
   - Requires overcoming an energy barrier

2. **Growth**: Expansion of ice crystals from nuclei
   - Controlled by heat and mass transfer
   - Crystal morphology depends on cooling rate and solute concentration
   - Dendrites form under rapid cooling conditions

### Factors Affecting Ice Crystal Formation

1. **Cooling Rate**:
   - Fast cooling: Many small crystals (intracellular ice formation)
   - Slow cooling: Few large crystals (extracellular ice formation)

2. **Cryoprotectants**:
   - Reduce nucleation rate
   - Slow crystal growth
   - Modify crystal morphology
   - Can prevent ice formation entirely (vitrification)

3. **Sample Properties**:
   - Cell type and membrane permeability
   - Presence of nucleating agents
   - Viscosity and solute concentration

### Damage Mechanisms

Ice crystals damage cells through:

1. **Mechanical Damage**: Direct physical disruption of membranes and organelles
2. **Solution Effects**: Concentration of solutes in unfrozen fraction
3. **Dehydration**: Osmotic water loss during extracellular freezing
4. **Recrystallization**: Growth of large crystals during warming

### Strategies to Control Ice Formation

1. **Controlled Cooling Rates**: Optimize for specific cell types
2. **Cryoprotectant Optimization**: Balance protection vs. toxicity
3. **Ice Seeding**: Controlled nucleation at high subzero temperatures
4. **Vitrification**: Use of high CPA concentrations to prevent ice formation
5. **Rapid Warming**: Minimize time for recrystallization

### Applications

Understanding ice crystal formation is crucial for:

1. **Cell Banking**: Preserving cells with minimal damage
2. **Tissue Preservation**: Maintaining tissue architecture during freezing
3. **Food Freezing**: Controlling texture and quality
4. **Cryosurgery**: Controlled tissue destruction through freezing
""")
