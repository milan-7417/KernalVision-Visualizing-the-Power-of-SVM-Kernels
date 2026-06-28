import streamlit as st
import os

# Main Header Area
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=100)
with col_title:
    st.markdown(
        """
        <h1 style='margin-bottom: 0px;'>KernelVision</h1>
        <p style='color: #64748B; font-size: 1.25rem; margin-top: 5px;'>Interactive Exploration of Support Vector Machines & The Kernel Trick</p>
        """,
        unsafe_allow_html=True
    )

# Hero Section
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Demystify High-Dimensional Machine Learning</div>
        <p class="hero-subtitle">
            KernelVision is an educational platform designed to make SVM and kernel functions intuitive.
            Generate non-linear datasets, train models in real-time, compare kernel boundaries side-by-side,
            and visualize the Kernel Trick in 3D.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Display Hero Image if available
if os.path.exists("assets/hero.png"):
    st.image("assets/hero.png", use_container_width=True, caption="Visualizing classification boundaries and 3D projections")

# Core SVM Theory Intro
st.markdown("<div class='section-header'><h3>Understanding Support Vector Machines</h3></div>", unsafe_allow_html=True)
col_theory1, col_theory2 = st.columns(2)

with col_theory1:
    st.markdown(
        """
        <div class="kv-card">
            <h4 style="color: #2563EB;">What is an SVM?</h4>
            <p>
                A <strong>Support Vector Machine (SVM)</strong> is a supervised machine learning model used for classification and regression. 
                Its core objective is to find a <strong>hyperplane</strong> in an N-dimensional space that distinctly classifies data points.
            </p>
            <p>
                The model searches for the hyperplane that maximizes the <strong>margin</strong> (the distance between the boundary and the closest data points from either class).
                These critical boundary-defining points are called <strong>Support Vectors</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_theory2:
    st.markdown(
        """
        <div class="kv-card">
            <h4 style="color: #06B6D4;">The Power of the Kernel Trick</h4>
            <p>
                Real-world data is rarely linearly separable in its original 2D or 3D space. 
                Instead of trying to fit a bendy curve to separate classes, the <strong>Kernel Trick</strong> mathematically 
                projects the data points into a <strong>higher-dimensional space</strong> where they become linearly separable.
            </p>
            <p>
                Crucially, SVM kernels compute these higher-dimensional inner products <em>without</em> ever explicitly transforming the data points, 
                avoiding massive computational cost.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Workflow Section
st.markdown("<div class='section-header'><h3>KernelVision Application Workflow</h3></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="timeline-container">
        <div class="timeline-step">
            <div class="step-num">1</div>
            <div class="step-title">Generate Dataset</div>
            <div class="step-desc">Create Moons, Circles, or XOR distributions. Change sample sizes, noise, and see metrics.</div>
        </div>
        <div class="timeline-step">
            <div class="step-num">2</div>
            <div class="step-title">SVM Visualizer Sandbox</div>
            <div class="step-desc">Tune C, Gamma, and Degree. Real-time rendering of decision boundaries, support vectors, and metrics.</div>
        </div>
        <div class="timeline-step">
            <div class="step-num">3</div>
            <div class="step-title">Kernel Comparison</div>
            <div class="step-desc">Train and compare Linear, Polynomial, RBF, and Sigmoid boundaries in a synchronous 2x2 grid layout.</div>
        </div>
        <div class="timeline-step">
            <div class="step-num">4</div>
            <div class="step-title">Kernel Trick in 3D</div>
            <div class="step-desc">See the math, and rotate a 3D scatter plot showcasing concentric circles separated by a flat plane.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation Buttons
st.markdown("<h4 style='text-align: center; margin-top: 2rem; margin-bottom: 1.5rem;'>Ready to explore? Jump straight to a page:</h4>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Go to Generator", use_container_width=True):
        st.switch_page("pages/dataset.py")

with col2:
    if st.button("🔮 SVM Visualizer Sandbox", use_container_width=True):
        st.switch_page("pages/visualizer.py")

with col3:
    if st.button("⚖️ Compare Kernels", use_container_width=True):
        st.switch_page("pages/comparison.py")

with col4:
    if st.button("💡 See Kernel Trick", use_container_width=True):
        st.switch_page("pages/kernel_trick.py")
