import streamlit as st
import os
import numpy as np
import pandas as pd

# Set page config once at the entrypoint file app.py
st.set_page_config(
    page_title="KernelVision",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom css injector helper
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # If CSS file doesn't exist, we will create a placeholder style
        pass

# Inject styling
local_css("assets/style.css")

# Initialize Session State
if "dataset" not in st.session_state:
    # Initialize a default dataset (circles) so standard pages don't crash
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=300, noise=0.15, factor=0.5, random_state=42)
    # Map values to a wider range for better plotting/scaling
    X = X * 5.0
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    st.session_state["dataset"] = df
    st.session_state["dataset_name"] = "Concentric Circles (Default)"
    st.session_state["dataset_params"] = {"n_samples": 300, "noise": 0.15, "seed": 42}

# Setup the sidebar layout branding
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: #2563EB; margin-bottom: 5px;'>🔮 KernelVision</h2>
        <p style='color: #6B7280; font-size: 0.9rem;'>SVM Educational Interactive App</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Import pages using the Streamlit 1.35+ Page/Navigation system
home_page = st.Page("pages/home.py", title="Home", icon="🏠", default=True)
dataset_page = st.Page("pages/dataset.py", title="Dataset Generator", icon="📊")
visualizer_page = st.Page("pages/visualizer.py", title="SVM Visualizer", icon="🔮")
comparison_page = st.Page("pages/comparison.py", title="Kernel Comparison", icon="⚖️")
kernel_trick_page = st.Page("pages/kernel_trick.py", title="Kernel Trick", icon="💡")
about_page = st.Page("pages/about.py", title="About", icon="ℹ️")

# Build navigation
pg = st.navigation({
    "KernelVision": [home_page],
    "Sandbox": [dataset_page, visualizer_page, comparison_page],
    "Theory": [kernel_trick_page, about_page]
})

# Run the page logic
pg.run()
