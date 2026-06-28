import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_circles

st.markdown("## 💡 The Kernel Trick Explained")
st.markdown(
    "How do support vector machines draw highly complex, non-linear boundaries in a 2D dataset? "
    "They use the **Kernel Trick**. Explore the interactive 3D visual mapping below to build a geometric intuition."
)

# Section 1: Interactive 3D Mapping Visualizer
st.markdown("<div class='section-header'><h3>Interactive 3D Feature Projection</h3></div>", unsafe_allow_html=True)

col_ctrl, col_plot = st.columns([1, 2], gap="large")

with col_ctrl:
    st.markdown(
        """
        <div class="kv-card" style="padding: 1.25rem; margin-bottom: 1.5rem;">
            <h5 style="color:#2563EB;">Projecting 2D to 3D</h5>
            <p style="font-size:0.9rem; line-height: 1.5;">
                In 2D space on the right, the <strong>inner circle</strong> (Class 0) and the <strong>outer ring</strong> (Class 1) 
                are nested. They cannot be separated by any single straight line.
            </p>
            <p style="font-size:0.9rem; line-height: 1.5;">
                We can apply a <strong>feature mapping function</strong> &Phi;(x) that adds a third dimension:
            </p>
            <div style="background-color: #F8FAFC; padding: 10px; border-radius: 6px; font-family: monospace; font-size:0.85rem; border: 1px solid #E2E8F0; text-align: center; margin: 10px 0;">
                x &rarr; [x₁, x₂, z]
            </div>
            <p style="font-size:0.9rem; line-height: 1.5;">
                Select a mapping function below to project the points up into 3D space and see how a flat <strong>hyperplane</strong> can separate them.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    mapping_choice = st.selectbox(
        "Choose Feature Mapping Function for z-axis",
        [
            "Quadratic / Distance: z = x₁² + x₂²",
            "RBF Similarity: z = exp(-0.08 * (x₁² + x₂²))",
            "Linear Projection (Fails): z = x₁ + x₂"
        ],
        index=0
    )
    
    st.info("💡 Pro-Tip: Grab, rotate, and zoom the 3D graph on the right to inspect how the separating plane slices through the gap.")

# Generate dataset for 3D demo
X, y = make_circles(n_samples=500, noise=0.05, factor=0.45, random_state=42)
X = X * 5.0 # scale coordinates

x1 = X[:, 0]
x2 = X[:, 1]

# Calculate z based on chosen mapping
if "Quadratic / Distance" in mapping_choice:
    z = x1**2 + x2**2
    # Threshold for separating hyperplane
    # Inner circle has radius ~2.25 => z ~ 5.06
    # Outer circle has radius ~5.0 => z ~ 25.0
    thresh = 13.5
    plane_z = np.ones((10, 10)) * thresh
    z_label = "Z (x₁² + x₂²)"
    z_range = [0, 30]
elif "RBF Similarity" in mapping_choice:
    z = np.exp(-0.08 * (x1**2 + x2**2))
    # Inner circle z ~ exp(-0.08 * 5) = exp(-0.4) ~ 0.67
    # Outer circle z ~ exp(-0.08 * 25) = exp(-2.0) ~ 0.13
    thresh = 0.4
    plane_z = np.ones((10, 10)) * thresh
    z_label = "Z (exp(-0.08 * ||x||²))"
    z_range = [0, 1.1]
else: # Linear
    z = x1 + x2
    # No simple horizontal separation possible
    thresh = 0.0
    plane_z = np.ones((10, 10)) * thresh
    z_label = "Z (x₁ + x₂)"
    z_range = [-10, 10]

# Build 3D plot
fig_3d = go.Figure()

# Class 0: Inner Circle
fig_3d.add_trace(go.Scatter3d(
    x=x1[y == 0],
    y=x2[y == 0],
    z=z[y == 0],
    mode="markers",
    marker=dict(
        size=4.5,
        color="#06B6D4", # Cyan
        opacity=0.8,
        line=dict(color="white", width=0.5)
    ),
    name="Class 0 (Inner Cluster)"
))

# Class 1: Outer Ring
fig_3d.add_trace(go.Scatter3d(
    x=x1[y == 1],
    y=x2[y == 1],
    z=z[y == 1],
    mode="markers",
    marker=dict(
        size=4.5,
        color="#2563EB", # Blue
        opacity=0.8,
        line=dict(color="white", width=0.5)
    ),
    name="Class 1 (Outer Ring)"
))

# Separating Hyperplane Surface
x_plane = np.linspace(-6, 6, 10)
y_plane = np.linspace(-6, 6, 10)
xx_plane, yy_plane = np.meshgrid(x_plane, y_plane)

fig_3d.add_trace(go.Surface(
    x=xx_plane,
    y=yy_plane,
    z=plane_z,
    colorscale=[[0, "rgba(30, 41, 59, 0.4)"], [1, "rgba(30, 41, 59, 0.4)"]],
    showscale=False,
    opacity=0.5,
    name="Separating Hyperplane",
    hoverinfo="skip"
))

# Style 3D graph
fig_3d.update_layout(
    title="3D Projected Feature Space",
    title_font=dict(family="Outfit", size=18, color="#1E293B"),
    height=600,
    margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(
        xaxis=dict(title="X₁", backgroundcolor="rgb(250, 250, 250)", gridcolor="white", showbackground=True),
        yaxis=dict(title="X₂", backgroundcolor="rgb(250, 250, 250)", gridcolor="white", showbackground=True),
        zaxis=dict(title=z_label, range=z_range, backgroundcolor="rgb(240, 240, 240)", gridcolor="white", showbackground=True),
        aspectratio=dict(x=1, y=1, z=0.8)
    ),
    legend=dict(orientation="h", yanchor="bottom", y=0.9, xanchor="right", x=1)
)

with col_plot:
    st.plotly_chart(fig_3d, use_container_width=True)

# Section 2: Core Mathematics of Kernels
st.markdown("<div class='section-header'><h3>Mathematical Formulations</h3></div>", unsafe_allow_html=True)
st.markdown(
    "To achieve this higher-dimensional separation, we define a **Kernel Function** $K(x, y) = \\langle \\Phi(x), \\Phi(y) \\rangle$. "
    "This allows the SVM algorithm to calculate the dot product of coordinates in the projected space *without* ever explicitly computing $\\Phi(x)$."
)

col_math1, col_math2 = st.columns(2)

with col_math1:
    st.markdown(
        """
        <div class="kv-card">
            <h5 style="color: #2563EB;">1. Linear Kernel</h5>
            <p style="font-size:0.9rem; color: #64748B;">No projection is performed. Calculates the standard dot product in the original input space. Ideal for text and high-dimensional linear boundaries.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.latex(r"K(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T \mathbf{y}")
    
    st.markdown(
        """
        <div class="kv-card" style="margin-top: 1.5rem;">
            <h5 style="color: #06B6D4;">2. Polynomial Kernel</h5>
            <p style="font-size:0.9rem; color: #64748B;">Projects features up to degree <em>d</em>. Controls combinations of terms via the offset parameter <em>r</em> and scale <em>&gamma;</em>.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.latex(r"K(\mathbf{x}, \mathbf{y}) = (\gamma \mathbf{x}^T \mathbf{y} + r)^d")

with col_math2:
    st.markdown(
        """
        <div class="kv-card">
            <h5 style="color: #2563EB;">3. Gaussian RBF Kernel</h5>
            <p style="font-size:0.9rem; color: #64748B;">Radial Basis Function. Measures Euclidean distance closeness. Projects data into an <strong>infinite-dimensional</strong> Hilbert space.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.latex(r"K(\mathbf{x}, \mathbf{y}) = \exp(-\gamma \|\mathbf{x} - \mathbf{y}\|^2)")
    
    st.markdown(
        """
        <div class="kv-card" style="margin-top: 1.5rem;">
            <h5 style="color: #06B6D4;">4. Sigmoid Kernel</h5>
            <p style="font-size:0.9rem; color: #64748B;">Acts like a logistic activation. Originating from neural network mappings where SVM is treated as a 2-layer classifier.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.latex(r"K(\mathbf{x}, \mathbf{y}) = \tanh(\gamma \mathbf{x}^T \mathbf{y} + r)")

# Section 3: Educational Guide
st.markdown("<div class='section-header'><h3>Educational Guide: Choosing the Right Kernel</h3></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="kv-card" style="line-height: 1.6;">
        <h4>How to choose?</h4>
        <ul>
            <li><strong>Start with RBF:</strong> RBF is the default choice for general non-linear datasets because it can model almost any complex shape and maps data to infinite-dimensional spaces.</li>
            <li><strong>Use Linear for text or large features:</strong> If your dataset has a huge number of features (e.g. 10,000+ words in TF-IDF matrices) and few samples, it is highly likely to be linearly separable already. A Linear kernel will be much faster and prevent overfitting.</li>
            <li><strong>Use Polynomial if degree correlation is known:</strong> If you know features interact quadratically or cubically (e.g. physical distance/gravitational math), Polynomial kernels can match that structure nicely.</li>
            <li><strong>Regularization matters:</strong> Even with a powerful kernel like RBF, if you tune regularization parameter <code>C</code> too high, your model will overfit by creating tight circular islands around individual points.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
