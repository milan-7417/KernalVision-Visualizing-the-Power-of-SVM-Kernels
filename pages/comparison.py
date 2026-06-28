import streamlit as st
import numpy as np
from utils.svm_model import train_svm
from utils.plots import plot_kernel_comparison

st.markdown("## ⚖️ Multi-Kernel Comparison")
st.markdown(
    "Observe how all four SVM kernel functions fit decision boundaries on the same dataset. "
    "Use the slider to adjust regularization `C` across all classifiers simultaneously."
)

# Initialize dataset in session state if not done
if "dataset" not in st.session_state:
    st.switch_page("app.py")

# Retrieve active dataset
df_active = st.session_state["dataset"]
X = df_active[["x1", "x2"]].values
y = df_active["label"].values

# Control parameter C for comparison
col_header, col_slider = st.columns([2, 1.5], gap="large")
with col_slider:
    C_val = st.slider(
        "Regularization strength (C) for all models",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.05,
        format="%.2f",
        help="Adjusting C changes the margin width and penalty factor across all 4 models."
    )
with col_header:
    st.markdown(
        f"""
        <div style='background-color: #F8FAFC; padding: 10px 15px; border-radius: 8px; border-left: 4px solid #2563EB;'>
            <strong>Active Dataset:</strong> {st.session_state['dataset_name']}<br>
            <strong>Total samples:</strong> {len(X)}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Train all 4 kernels
with st.spinner("Training all 4 kernels (Linear, Poly, RBF, Sigmoid)..."):
    models = {
        "Linear Kernel": train_svm(X, y, kernel="linear", C=C_val),
        "Polynomial Kernel (degree=3)": train_svm(X, y, kernel="poly", degree=3, C=C_val, coef0=1.0),
        "RBF (Gaussian) Kernel": train_svm(X, y, kernel="rbf", C=C_val, gamma="scale"),
        "Sigmoid Kernel": train_svm(X, y, kernel="sigmoid", C=C_val, coef0=1.0)
    }

# Plot subplots
fig_comparison = plot_kernel_comparison(models, X, y)
st.plotly_chart(fig_comparison, use_container_width=True)

# Comparison Guide Table
st.markdown("<div class='section-header'><h3>Kernel Comparison Reference Guide</h3></div>", unsafe_allow_html=True)

st.markdown(
    """
    <table style="width:100%; border-collapse: collapse; margin-top: 15px; font-size: 0.95rem; text-align: left;">
      <thead>
        <tr style="background-color: #F1F5F9; border-bottom: 2px solid #CBD5E1;">
          <th style="padding: 12px; font-weight: 600; color: #1E293B; width: 15%;">Kernel Type</th>
          <th style="padding: 12px; font-weight: 600; color: #1E293B; width: 20%;">Best Suited For</th>
          <th style="padding: 12px; font-weight: 600; color: #1E293B; width: 25%;">Key Advantages</th>
          <th style="padding: 12px; font-weight: 600; color: #1E293B; width: 25%;">Disadvantages / Risks</th>
          <th style="padding: 12px; font-weight: 600; color: #1E293B; width: 15%;">Complexity</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom: 1px solid #E2E8F0;">
          <td style="padding: 12px; font-weight: 600; color: #2563EB;">Linear</td>
          <td style="padding: 12px;">Linearly separable classes (e.g., text document classification, high feature count).</td>
          <td style="padding: 12px;">Fastest to train. Simple math reduces overfitting risk. Very interpretable coefficients.</td>
          <td style="padding: 12px;">Cannot model non-linear boundaries. Performance is poor if patterns are curved.</td>
          <td style="padding: 12px; font-family: monospace;">O(n_samples × n_features)</td>
        </tr>
        <tr style="border-bottom: 1px solid #E2E8F0; background-color: #F8FAFC;">
          <td style="padding: 12px; font-weight: 600; color: #06B6D4;">Polynomial</td>
          <td style="padding: 12px;">Structured tabular data, image pixel associations, and degree-specific curved relationships.</td>
          <td style="padding: 12px;">Models polynomial curved boundaries. Highly customizable via degree (d) and intercept parameters.</td>
          <td style="padding: 12px;">Overfits easily at higher degrees. Computationally expensive for high degree values.</td>
          <td style="padding: 12px; font-family: monospace;">O(n_samples² × n_features) to O(n_samples³)</td>
        </tr>
        <tr style="border-bottom: 1px solid #E2E8F0;">
          <td style="padding: 12px; font-weight: 600; color: #2563EB;">RBF (Radial Basis Function)</td>
          <td style="padding: 12px;">Default general classifier. Complex patterns, circles, nested boundaries, or unknown distributions.</td>
          <td style="padding: 12px;">Extremely flexible. Maps data to infinite dimensions. Handles complex non-linear structures easily.</td>
          <td style="padding: 12px;">Requires careful tuning of both C and Gamma. High risk of overfitting. Large prediction times.</td>
          <td style="padding: 12px; font-family: monospace;">O(n_samples² × n_features) to O(n_samples³)</td>
        </tr>
        <tr style="border-bottom: 1px solid #CBD5E1; background-color: #F8FAFC;">
          <td style="padding: 12px; font-weight: 600; color: #06B6D4;">Sigmoid</td>
          <td style="padding: 12px;">Neural Network analogy mapping. Mimics sigmoid layers in deep learning representations.</td>
          <td style="padding: 12px;">Draws interesting complex boundaries. Can act as a proxy classifier for shallow neural layers.</td>
          <td style="padding: 12px;">Not positive semi-definite for some parameters, which breaks Mercer's theorem. Tends to over-saturate.</td>
          <td style="padding: 12px; font-family: monospace;">O(n_samples² × n_features) to O(n_samples³)</td>
        </tr>
      </tbody>
    </table>
    """,
    unsafe_allow_html=True
)
