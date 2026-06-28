import streamlit as st
import pandas as pd
import plotly.express as px
from utils.dataset_generator import generate_moons, generate_circles, generate_blobs, generate_xor
from utils.plots import plot_dataset

st.markdown("## 📊 Dataset Generator Workspace")
st.markdown(
    "Configure and generate synthetic 2D datasets. The dataset you create here will automatically carry over to the "
    "**SVM Visualizer** and **Kernel Comparison** pages."
)

# Initialize dataset in session state if not done
if "dataset" not in st.session_state:
    st.switch_page("app.py") # fallback to initialize

# Layout: 1/3 Controls, 2/3 Plots and Statistics
col_ctrl, col_display = st.columns([1, 2.2], gap="large")

with col_ctrl:
    st.markdown("<div class='section-header'><h4>Dataset Parameters</h4></div>", unsafe_allow_html=True)
    
    # Dataset Type selector
    dataset_type = st.selectbox(
        "Select Distribution Pattern",
        ["Concentric Circles", "Two Moons", "Blobs (Gaussian clusters)", "XOR Pattern"],
        index=0,
        help="Select the structural layout of data classes."
    )
    
    # Sample Size
    n_samples = st.slider(
        "Number of Samples",
        min_value=50,
        max_value=1000,
        value=st.session_state.get("dataset_params", {}).get("n_samples", 300),
        step=50,
        help="Total number of points generated in the dataset."
    )
    
    # Noise slider
    noise = st.slider(
        "Noise Level",
        min_value=0.00,
        max_value=1.00,
        value=st.session_state.get("dataset_params", {}).get("noise", 0.15),
        step=0.05,
        help="Amount of randomness/uncertainty added to the class positions. Higher noise makes boundaries overlap."
    )
    
    # Random Seed
    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=st.session_state.get("dataset_params", {}).get("seed", 42),
        step=1,
        help="Fixed seed value to make dataset generation reproducible."
    )
    
    generate_btn = st.button("🔄 Generate Dataset", use_container_width=True)

# Process generation on click or load
if generate_btn:
    with st.spinner("Generating dataset..."):
        if dataset_type == "Concentric Circles":
            df = generate_circles(n_samples=n_samples, noise=noise, random_state=seed)
        elif dataset_type == "Two Moons":
            df = generate_moons(n_samples=n_samples, noise=noise, random_state=seed)
        elif dataset_type == "Blobs (Gaussian clusters)":
            df = generate_blobs(n_samples=n_samples, noise=noise, random_state=seed)
        else: # XOR
            df = generate_xor(n_samples=n_samples, noise=noise, random_state=seed)
            
        # Update session state
        st.session_state["dataset"] = df
        st.session_state["dataset_name"] = dataset_type
        st.session_state["dataset_params"] = {
            "n_samples": n_samples,
            "noise": noise,
            "seed": seed
        }
        st.toast(f"Generated {dataset_type} successfully!", icon="✅")

# Retrieve active dataset from state
df_active = st.session_state["dataset"]
ds_name = st.session_state["dataset_name"]
ds_params = st.session_state["dataset_params"]

with col_display:
    st.markdown(f"<div class='section-header'><h4>Active Dataset: {ds_name}</h4></div>", unsafe_allow_html=True)
    
    # Scatter plot
    fig_scatter = plot_dataset(df_active)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Data stats and distribution subcolumns
    st.markdown("<h4>Dataset Statistics</h4>", unsafe_allow_html=True)
    col_stat1, col_stat2 = st.columns([1, 1.2], gap="medium")
    
    with col_stat1:
        n_total = len(df_active)
        n_class_0 = sum(df_active["label"] == 0)
        n_class_1 = sum(df_active["label"] == 1)
        
        # Injected CSS styles for statistics card
        st.markdown(
            f"""
            <div class="metric-container" style="flex-direction: column; width: 100%;">
                <div class="metric-card">
                    <div class="metric-value">{n_total}</div>
                    <div class="metric-label">Total Points</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #06B6D4;">{n_class_0}</div>
                    <div class="metric-label">Class 0 Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #2563EB;">{n_class_1}</div>
                    <div class="metric-label">Class 1 Samples</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_stat2:
        # Distribution plot
        dist_df = pd.DataFrame({
            "Class": ["Class 0", "Class 1"],
            "Count": [n_class_0, n_class_1],
            "Color": ["Class 0 (Cyan)", "Class 1 (Blue)"]
        })
        fig_bar = px.bar(
            dist_df,
            x="Class",
            y="Count",
            color="Color",
            color_discrete_map={
                "Class 0 (Cyan)": "#06B6D4",
                "Class 1 (Blue)": "#2563EB"
            },
            title="Class Balance Ratio",
            labels={"Count": "Count", "Class": "Class"}
        )
        fig_bar.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            margin=dict(l=20, r=20, t=50, b=20),
            title_font=dict(family="Outfit", size=14, color="#1E293B")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
