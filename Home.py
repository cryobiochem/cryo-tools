import streamlit as st

st.set_page_config(
    page_title="Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("Cryopreservation Research Hub")

st.markdown("""
## Welcome to the Cryopreservation Research Hub

This interactive platform provides tools and resources for researchers in the field of cryobiology, 
low temperature physics, and cryopreservation techniques.

### Available Tools

Navigate through the sidebar to access various interactive tools:

1. **Nucleation Energy Barrier Calculator** - Visualize and calculate nucleation activation energy barriers
2. **Cell Osmolarity Response Simulator** - Model cell inflation and shrinking based on solution osmolarity
3. **Cooling Rate Optimizer** - Determine optimal cooling rates for different biological samples
4. **Cryoprotectant Toxicity Analyzer** - Analyze toxicity profiles of common cryoprotective agents
5. **Vitrification Probability Calculator** - Calculate vitrification vs. crystallization probabilities
6. **Thermal Stress Simulator** - Model thermal stresses during freezing and thawing
7. **Ice Crystal Growth Visualizer** - Simulate ice crystal growth patterns
8. **Membrane Permeability Modeler** - Calculate membrane permeability at different temperatures
9. **Warming Protocol Designer** - Design optimal warming protocols
10. **Storage Temperature Stability Analyzer** - Analyze sample stability at different storage temperatures

### Getting Started

Select a tool from the sidebar to begin exploring the interactive simulations and calculators.
""")

# Multipage sidebar
nav = st.navigation(
    {
        "Tools":    [
                     st.Page("2_Nucleation_Energy_Barrier.py", title="Nucleation Energy Barrier Calculator", icon=":material/add_circle:"),
                     st.Page("3_Cell_Osmolarity_Response.py", title="Cell Osmolarity Response Simulator", icon=":material/add_circle:"),
                     st.Page("4_Cooling_Rate_Optimizer.py", title="Cooling Rate Optimizer", icon=":material/add_circle:"),
                     st.Page("5_Cryoprotectant_Toxicity.py", title="Cryoprotectant Toxicity Analyzer", icon=":material/add_circle:"),
                     st.Page("6_Vitrification_Probability.py", title="Vitrification Probability Calculator", icon=":material/add_circle:"),
                     st.Page("7_Thermal_Stress_Simulator.py", title="Thermal Stress Simulator", icon=":material/add_circle:"),
                     st.Page("8_Ice_Crystal_Growth.py", title="Ice Crystal Growth Visualizer", icon=":material/add_circle:"),
                     st.Page("9_Membrane_Permeability.py", title="Membrane Permeability Modeler", icon=":material/add_circle:"),
                     st.Page("10_Warming_Protocol.py", title="Warming Protocol Designer", icon=":material/add_circle:"),
                     st.Page("11_Storage_Temperature.py", title="Storage Temperature Stability Analyzer", icon=":material/add_circle:")
                    ],
         "About":    [
                      st.Page("1_About.py", title="The author", icon=":material/add_circle:")
                     ]

    })
nav.run()

st.image("https://placeholder.svg?height=300&width=700", caption="Cryopreservation Research")

# Footer
st.markdown("---")
st.markdown("© 2025 Cryopreservation Research Hub")
