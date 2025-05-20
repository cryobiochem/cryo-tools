import streamlit as st

st.set_page_config(
    page_title="About - Cryopreservation Research Hub",
    page_icon="❄️",
    layout="wide"
)

st.title("About Me and My Research")

# Create columns for profile and bio
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://placeholder.svg?height=300&width=300", caption="Researcher Photo")

with col2:
    st.markdown("""
    ## Dr. [Your Name]
    **Principal Investigator, Cryobiology Research Laboratory**
    
    I am a researcher specializing in cryopreservation techniques and low-temperature biology. 
    My work focuses on developing novel approaches to preserve biological materials at ultra-low 
    temperatures while maintaining their structural and functional integrity.
    
    ### Research Interests
    - Vitrification techniques for cell and tissue preservation
    - Mathematical modeling of ice nucleation and crystal growth
    - Cryoprotectant development and optimization
    - Thermal and mechanical stress during freezing and thawing
    - Long-term stability of cryopreserved biological materials
    """)

st.markdown("---")

st.header("Education and Training")
st.markdown("""
- **Ph.D. in Biophysics**, University of [University Name], 2018
- **M.S. in Biomedical Engineering**, [University Name], 2014
- **B.S. in Physics**, [University Name], 2012
""")

st.header("Selected Publications")
st.markdown("""
1. [Your Name], et al. (2024). "Novel approaches to vitrification of complex biological tissues." *Journal of Cryobiology*, 108(3), 245-259.

2. [Your Name], et al. (2023). "Mathematical modeling of intracellular ice formation during rapid cooling." *Biophysical Journal*, 125(8), 1678-1692.

3. [Your Name], et al. (2022). "Optimization of warming protocols to minimize devitrification damage." *Cryopreservation Science*, 45(2), 112-128.

4. [Your Name], et al. (2021). "Effects of solution osmolarity on cell volume regulation during cryopreservation." *Cell Preservation Technology*, 19(4), 387-401.

5. [Your Name], et al. (2020). "Thermal gradients and mechanical stress during directional solidification of biological materials." *International Journal of Heat and Mass Transfer*, 156, 119860.
""")

st.header("Current Projects")
st.markdown("""
- **Multicomponent Cryoprotectant Systems**: Developing synergistic cryoprotectant mixtures with reduced toxicity.
- **Na