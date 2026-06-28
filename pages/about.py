import streamlit as st

st.markdown("## ℹ️ About KernelVision")
st.markdown("KernelVision is an interactive visual sandbox designed to make Support Vector Machines (SVM) accessible and intuitive.")

# Project Overview & Motivation
st.markdown("<div class='section-header'><h3>Project Overview</h3></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="kv-card" style="line-height: 1.6;">
        <p>
            Understanding the math behind SVM margins, Support Vectors, and the mysterious "Kernel Trick" can be a daunting hurdle for Machine Learning beginners. 
            Typical textbooks show static equations and 2D charts, which fail to capture how decision boundaries morph dynamically in response to parameter tuning.
        </p>
        <p>
            <strong>KernelVision</strong> bridges this gap by offering a fully responsive, code-free visual environment. 
            By allowing users to generate complex non-linear structures, tune hyperparameters in real-time, see comparative performance, and rotate 3D spaces, 
            the application translates dry equations into geometric intuition.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Tech Stack & Libraries
st.markdown("<div class='section-header'><h3>Technology Stack</h3></div>", unsafe_allow_html=True)
col_tech1, col_tech2 = st.columns(2)

with col_tech1:
    st.markdown(
        """
        <div class="kv-card">
            <h5 style="color: #2563EB;">Core Engineering</h5>
            <ul>
                <li><strong>Python:</strong> Core language logic, numerical routines, and model pipelines.</li>
                <li><strong>Streamlit:</strong> High-performance, reactive front-end framework for machine learning apps.</li>
                <li><strong>Scikit-Learn:</strong> Powers the underlying Support Vector Classifier (SVC) fitting and metric calculation.</li>
                <li><strong>Pandas & NumPy:</strong> Handles structured data vectors, arrays, and column mappings.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_tech2:
    st.markdown(
        """
        <div class="kv-card">
            <h5 style="color: #06B6D4;">Data Visualizations</h5>
            <ul>
                <li><strong>Plotly:</strong> Interactive, fully responsive 2D scatter contours, 3D scatter meshes, and 3D surface charts.</li>
                <li><strong>Matplotlib:</strong> Static rendering sub-mechanisms and vector layout support.</li>
                <li><strong>Custom CSS:</strong> Custom UI styling injecting glassmorphism cards, micro-hover animations, and professional metrics display.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# Real World Applications of SVM
st.markdown("<div class='section-header'><h3>Real-World Applications of SVM</h3></div>", unsafe_allow_html=True)
col_app1, col_app2 = st.columns(2)

with col_app1:
    st.markdown(
        """
        <div class="kv-card">
            <h6 style="color: #1E293B;">Bioinformatics & DNA Classification</h6>
            <p style="font-size:0.9rem; color:#64748B;">SVMs with specialized sequence kernels are widely used to identify genes, classify proteins, and identify cancer tissues from microarray gene expression data.</p>
        </div>
        <div class="kv-card" style="margin-top: 1rem;">
            <h6 style="color: #1E293B;">Text Categorization & Spam Detection</h6>
            <p style="font-size:0.9rem; color:#64748B;">Linear SVMs are highly effective at classifying emails as spam, sorting documents, and perform sentiment analysis on text corpora where feature dimensions are high.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_app2:
    st.markdown(
        """
        <div class="kv-card">
            <h6 style="color: #1E293B;">Handwritten Character Recognition</h6>
            <p style="font-size:0.9rem; color:#64748B;">SVMs are used for optical character recognition (OCR) and identifying postal zip codes. They maintain high accuracy even with minimal training datasets.</p>
        </div>
        <div class="kv-card" style="margin-top: 1rem;">
            <h6 style="color: #1E293B;">Image Classification & Face Detection</h6>
            <p style="font-size:0.9rem; color:#64748B;">Splitting sections of images into segmented classes, highlighting human features, and detecting boundary edges in computer vision pipelines.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Future Scope
st.markdown("<div class='section-header'><h3>Future Improvements</h3></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="kv-card">
        <ul>
            <li><strong>Support for Custom Datasets:</strong> Allow users to upload their own CSV files and specify features/labels.</li>
            <li><strong>Support Vector Regression (SVR):</strong> Add a dedicated regression page to showcase margin boundaries on continuous functions.</li>
            <li><strong>Multiclass SVM:</strong> Demonstrate One-vs-Rest (OvR) and One-vs-One (OvO) decision boundaries on 3+ classes.</li>
            <li><strong>Custom Kernel Functions:</strong> Allow users to write and execute custom mathematical formulas for the kernel inner product.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Developer Profile
st.markdown("<div class='section-header'><h3>Developer Information</h3></div>", unsafe_allow_html=True)
col_dev, col_spacer = st.columns([2, 1.5])

with col_dev:
    st.markdown(
        """
        <div class="kv-card" style="border-left: 5px solid #06B6D4;">
            <h5 style="margin-bottom: 2px;">KernelVision Team</h5>
            <p style="font-size:0.85rem; color:#64748B; margin-bottom: 10px;">Machine Learning Educators & Developers</p>
            <p style="font-size:0.9rem; margin-bottom: 15px;">
                We build interactive tools that make machine learning concept visualizations beautiful and easy to understand.
            </p>
            <div style="display: flex; gap: 10px;">
                <a href="https://github.com" target="_blank" style="text-decoration: none; padding: 6px 12px; background-color: #24292F; color: white; border-radius: 4px; font-size: 0.8rem; font-weight:600;">GitHub Repository</a>
                <a href="https://linkedin.com" target="_blank" style="text-decoration: none; padding: 6px 12px; background-color: #0A66C2; color: white; border-radius: 4px; font-size: 0.8rem; font-weight:600;">LinkedIn Profile</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
