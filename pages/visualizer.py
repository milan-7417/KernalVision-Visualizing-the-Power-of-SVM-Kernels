import streamlit as st
import numpy as np
import pandas as pd
import pickle
from utils.svm_model import train_svm, predict_svm, get_support_vectors_info
from utils.metrics import calculate_metrics, get_confusion_matrix_data
from utils.plots import plot_decision_boundary, plot_heatmap
from sklearn.metrics import classification_report

st.markdown("## 🔮 SVM Visualizer Sandbox")
st.markdown(
    "Explore how different kernel functions and hyperparameters shape the classification decision boundary. "
    "Adjust parameters on the left to see the boundary and metrics update dynamically."
)

# Initialize dataset in session state if not done
if "dataset" not in st.session_state:
    st.switch_page("app.py")

# Retrieve active dataset
df_active = st.session_state["dataset"]
X = df_active[["x1", "x2"]].values
y = df_active["label"].values

# Layout: 1/3 Controls, 2/3 Visualizations
col_ctrl, col_plots = st.columns([1, 2.2], gap="large")

with col_ctrl:
    st.markdown("<div class='section-header'><h4>Model Hyperparameters</h4></div>", unsafe_allow_html=True)
    
    # Kernel Selection
    kernel_choice = st.selectbox(
        "Kernel Function",
        ["rbf", "linear", "poly", "sigmoid"],
        index=0,
        format_func=lambda x: f"{x.upper()} Kernel",
        help="The mathematical function used to project data into higher-dimensional space."
    )
    
    # Regularization parameter C
    C_val = st.slider(
        "Regularization (C)",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.05,
        format="%.2f",
        help="Penalty parameter C of the error term. Controls the trade-off between maximizing margin and minimizing training errors. Higher C creates a harder margin, focusing heavily on classifying training points correctly."
    )
    
    # Gamma (RBF, Poly, Sigmoid only)
    gamma_val = "scale"
    if kernel_choice in ["rbf", "poly", "sigmoid"]:
        gamma_mode = st.radio(
            "Gamma Coefficient (γ)",
            ["scale", "auto", "manual"],
            index=0,
            horizontal=True,
            help="Kernel coefficient. 'scale' uses 1 / (n_features * X.var()), 'auto' uses 1 / n_features."
        )
        if gamma_mode == "manual":
            gamma_val = st.slider(
                "Manual Gamma (γ) Value",
                min_value=0.001,
                max_value=10.0,
                value=0.1,
                step=0.01,
                format="%.3f",
                help="Higher gamma value means individual points have a narrower radius of influence, creating complex, localized boundaries (overfitting danger)."
            )
        else:
            gamma_val = gamma_mode
            
    # Degree (Poly only)
    degree_val = 3
    if kernel_choice == "poly":
        degree_val = st.slider(
            "Polynomial Degree (d)",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
            help="Degree of the polynomial kernel function. Higher degree allows for more complex, highly-curved boundaries."
        )
        
    # Coef0 / independent term (Poly & Sigmoid only)
    coef0_val = 0.0
    if kernel_choice in ["poly", "sigmoid"]:
        coef0_val = st.slider(
            "Independent Term (coef0 / r)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Independent constant term in the kernel mathematical formulation. It scales the relative influence of higher-order vs lower-order terms."
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    # Visual trigger button just in case user wants a hard trigger
    train_clicked = st.button("🚀 Train Model", use_container_width=True)

# Train the model dynamically on every widget update
with st.spinner("Fitting Support Vector Machine Classifier..."):
    # Perform fit
    model = train_svm(
        X=X,
        y=y,
        kernel=kernel_choice,
        C=C_val,
        gamma=gamma_val,
        degree=degree_val,
        coef0=coef0_val
    )
    
    # Run predictions on the training data
    y_pred = predict_svm(model, X)
    metrics = calculate_metrics(y, y_pred)
    sv_info = get_support_vectors_info(model)

with col_plots:
    # 1. Decision Boundary plot
    fig_boundary = plot_decision_boundary(model, X, y)
    st.plotly_chart(fig_boundary, use_container_width=True)
    
    # 2. Performance Metrics
    st.markdown("<div class='section-header'><h4>Performance Metrics & Diagnostics</h4></div>", unsafe_allow_html=True)
    
    col_kpi, col_cm = st.columns([1.5, 1], gap="medium")
    
    with col_kpi:
        st.markdown("<h5>Model KPIs</h5>", unsafe_allow_html=True)
        # Display customized KPI Cards using custom HTML from style.css
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-value">{metrics["accuracy"]:.2%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics["precision"]:.2%}</div>
                    <div class="metric-label">Precision</div>
                </div>
            </div>
            <div class="metric-container" style="margin-top: 0px;">
                <div class="metric-card">
                    <div class="metric-value">{metrics["recall"]:.2%}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics["f1_score"]:.2%}</div>
                    <div class="metric-label">F1-Score</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Support Vector Details Card
        st.markdown(
            f"""
            <div class="kv-card" style="margin-top: 0.5rem; padding: 1rem;">
                <h6 style="margin-bottom: 5px; color: #1E293B;">Support Vector Statistics</h6>
                <p style="font-size: 0.9rem; margin-bottom: 2px;">
                    Total Support Vectors: <strong>{sv_info["n_support"]}</strong> (out of {len(X)} points)
                </p>
                <p style="font-size: 0.85rem; color: #64748B;">
                    Class 0 support vectors: <strong>{sv_info["n_support_per_class"][0]}</strong> | 
                    Class 1 support vectors: <strong>{sv_info["n_support_per_class"][1]}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_cm:
        cm_data = get_confusion_matrix_data(y, y_pred)
        fig_cm = plot_heatmap(cm_data)
        st.plotly_chart(fig_cm, use_container_width=True)

    # 3. Model Downloads Section
    st.markdown("<div class='section-header'><h4>Export Options</h4></div>", unsafe_allow_html=True)
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # Pickle model file
        model_pkl = pickle.dumps(model)
        st.download_button(
            label="💾 Download Trained Model (.pkl)",
            data=model_pkl,
            file_name=f"svm_{kernel_choice}_C{C_val}.pkl",
            mime="application/octet-stream",
            use_container_width=True,
            help="Download the trained Scikit-learn SVC model to deploy in Python code."
        )
        
    with col_dl2:
        # Classification report text
        target_names = ["Class 0", "Class 1"]
        clf_report = classification_report(y, y_pred, target_names=target_names)
        
        st.download_button(
            label="📄 Download Classification Report",
            data=clf_report,
            file_name=f"svm_{kernel_choice}_report.txt",
            mime="text/plain",
            use_container_width=True,
            help="Download standard text evaluation report showing metrics per class."
        )
