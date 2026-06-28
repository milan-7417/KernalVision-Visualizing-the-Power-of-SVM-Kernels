"""
Plotting Utility for KernelVision.
Provides responsive and interactive visualizations using Plotly:
- Scatter plot of datasets
- Decision boundary contours with margin lines and support vectors
- 2x2 multi-kernel comparison subplots
- Confusion matrix heatmaps
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import SVC
from typing import Dict, Any, Union, Tuple

# Shared design parameters
COLORS = {
    0: "#06B6D4",      # Class 0: Cyan
    1: "#2563EB",      # Class 1: Blue
    "bg_cyan": "rgba(6, 182, 212, 0.15)",
    "bg_blue": "rgba(37, 99, 235, 0.15)",
    "border": "#E2E8F0",
    "text": "#1E293B"
}

def plot_dataset(df: pd.DataFrame) -> go.Figure:
    """
    Plots a 2D scatter plot of the generated dataset.
    """
    df_plot = df.copy()
    df_plot["Class"] = df_plot["label"].map({0: "Class 0 (Cyan)", 1: "Class 1 (Blue)"})
    
    fig = px.scatter(
        df_plot,
        x="x1",
        y="x2",
        color="Class",
        color_discrete_map={
            "Class 0 (Cyan)": COLORS[0],
            "Class 1 (Blue)": COLORS[1]
        },
        labels={"x1": "Feature X₁", "x2": "Feature X₂"},
        title="Interactive Dataset Scatter Plot"
    )
    
    # Styling updates
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Plus Jakarta Sans", color=COLORS["text"]),
        title_font=dict(family="Outfit", size=18, color=COLORS["text"]),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#CBD5E1"),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#CBD5E1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="white")))
    return fig

def plot_decision_boundary(model: SVC, X: np.ndarray, y: np.ndarray) -> go.Figure:
    """
    Generates a Plotly scatter plot of the data overlaying the SVM decision boundary,
    decision margins, and highlights the support vectors.
    """
    # Define range limits with padding
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    # Generate meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Calculate decision values
    Z = model.decision_function(grid_points)
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # 1. Background Filled Contours (Class Regions)
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 150),
        y=np.linspace(y_min, y_max, 150),
        z=Z,
        colorscale=[
            [0.0, "rgba(6, 182, 212, 0.15)"],  # Class 0 Region
            [0.5, "rgba(255, 255, 255, 0.2)"],  # Decision threshold zone
            [1.0, "rgba(37, 99, 235, 0.15)"]   # Class 1 Region
        ],
        showscale=False,
        contours=dict(coloring="fill", showlines=False),
        hoverinfo="skip",
        name="Decision Region"
    ))
    
    # 2. Decision boundary line (level = 0)
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 150),
        y=np.linspace(y_min, y_max, 150),
        z=Z,
        contours=dict(
            start=0,
            end=0,
            showlines=True,
            coloring="none"
        ),
        line=dict(color="rgba(30, 41, 59, 0.95)", width=2.5),
        showscale=False,
        hoverinfo="skip",
        name="Decision Boundary (wᵀx + b = 0)"
    ))

    # 3. Margin lines (level = -1 and +1)
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 150),
        y=np.linspace(y_min, y_max, 150),
        z=Z,
        contours=dict(
            start=-1,
            end=1,
            size=2, # draws contours at -1 and 1
            showlines=True,
            coloring="none"
        ),
        line=dict(color="rgba(71, 85, 105, 0.6)", width=1.5, dash="dash"),
        showscale=False,
        hoverinfo="skip",
        name="Margins (wᵀx + b = ±1)"
    ))
    
    # 4. Plot Scatter Points
    mask_0 = (y == 0)
    mask_1 = (y == 1)
    
    # Class 0 Scatter
    fig.add_trace(go.Scatter(
        x=X[mask_0, 0],
        y=X[mask_0, 1],
        mode="markers",
        marker=dict(color=COLORS[0], size=9, line=dict(color="white", width=1.2), opacity=0.85),
        name="Class 0"
    ))
    
    # Class 1 Scatter
    fig.add_trace(go.Scatter(
        x=X[mask_1, 0],
        y=X[mask_1, 1],
        mode="markers",
        marker=dict(color=COLORS[1], size=9, line=dict(color="white", width=1.2), opacity=0.85),
        name="Class 1"
    ))
    
    # 5. Highlight Support Vectors
    svs = model.support_vectors_
    if len(svs) > 0:
        fig.add_trace(go.Scatter(
            x=svs[:, 0],
            y=svs[:, 1],
            mode="markers",
            marker=dict(
                size=14,
                color="rgba(0,0,0,0)",
                line=dict(color="#1E293B", width=2),
                symbol="circle"
            ),
            name="Support Vectors",
            hovertemplate="Support Vector<br>X₁: %{x:.2f}<br>X₂: %{y:.2f}<extra></extra>"
        ))
        
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Plus Jakarta Sans", color=COLORS["text"]),
        title=f"SVM Decision Boundary ({model.kernel.upper()} Kernel)",
        title_font=dict(family="Outfit", size=18, color=COLORS["text"]),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False, range=[x_min, x_max]),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False, range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

def plot_kernel_comparison(models_dict: Dict[str, SVC], X: np.ndarray, y: np.ndarray) -> go.Figure:
    """
    Creates a 2x2 subplot comparison grid showing decision boundaries for four different kernels.
    Ensures scales are consistent.
    """
    # Consistent scale ranges
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    # 2x2 subplot setup
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(models_dict.keys()),
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    # Mesh grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for (kernel_name, model), (row, col) in zip(models_dict.items(), positions):
        Z = model.decision_function(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 1. Background Filled Contour
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            colorscale=[
                [0.0, "rgba(6, 182, 212, 0.1)"],
                [0.5, "rgba(255, 255, 255, 0.1)"],
                [1.0, "rgba(37, 99, 235, 0.1)"]
            ],
            showscale=False,
            contours=dict(coloring="fill", showlines=False),
            hoverinfo="skip"
        ), row=row, col=col)
        
        # 2. Decision boundary (0) and margins (-1, 1)
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            contours=dict(
                start=0,
                end=0,
                showlines=True,
                coloring="none"
            ),
            line=dict(color="rgba(30, 41, 59, 0.85)", width=2),
            showscale=False,
            hoverinfo="skip"
        ), row=row, col=col)
        
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            contours=dict(
                start=-1,
                end=1,
                size=2,
                showlines=True,
                coloring="none"
            ),
            line=dict(color="rgba(100, 116, 139, 0.5)", width=1, dash="dash"),
            showscale=False,
            hoverinfo="skip"
        ), row=row, col=col)
        
        # 3. Data Scatter
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        
        fig.add_trace(go.Scatter(
            x=X[mask_0, 0], y=X[mask_0, 1],
            mode="markers",
            marker=dict(color=COLORS[0], size=5, opacity=0.7),
            showlegend=False
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=X[mask_1, 0], y=X[mask_1, 1],
            mode="markers",
            marker=dict(color=COLORS[1], size=5, opacity=0.7),
            showlegend=False
        ), row=row, col=col)
        
        # 4. Highlight Support Vectors
        svs = model.support_vectors_
        if len(svs) > 0:
            fig.add_trace(go.Scatter(
                x=svs[:, 0], y=svs[:, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    color="rgba(0,0,0,0)",
                    line=dict(color="#1E293B", width=1.2),
                    symbol="circle"
                ),
                showlegend=False
            ), row=row, col=col)
            
    # Set uniform axes limits
    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(range=[x_min, x_max], showgrid=True, gridcolor="#F1F5F9", row=r, col=c)
            fig.update_yaxes(range=[y_min, y_max], showgrid=True, gridcolor="#F1F5F9", row=r, col=c)

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=650,
        font=dict(family="Plus Jakarta Sans", color=COLORS["text"]),
        title="2x2 Kernel Decision Boundary Comparison",
        title_font=dict(family="Outfit", size=20, color=COLORS["text"]),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

def plot_heatmap(conf_matrix: np.ndarray) -> go.Figure:
    """
    Renders a custom confusion matrix heatmap.
    """
    z = conf_matrix
    x = ["Predicted Class 0", "Predicted Class 1"]
    y = ["Actual Class 0", "Actual Class 1"]
    
    # Custom annotations
    annot = [[str(val) for val in row] for row in z]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[
            [0, "#E2E8F0"],     # Soft gray
            [0.5, "#E0F2FE"],   # Very light blue
            [1.0, "#3B82F6"]    # Primary blue
        ],
        showscale=False,
        text=annot,
        texttemplate="%{text}",
        textfont={"size": 18, "family": "Outfit", "color": "#1E293B"},
        hoverinfo="skip"
    ))
    
    fig.update_layout(
        title="Confusion Matrix Heatmap",
        title_font=dict(family="Outfit", size=16, color=COLORS["text"]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=350,
        height=320,
        margin=dict(l=60, r=40, t=60, b=40),
        xaxis=dict(tickfont=dict(family="Plus Jakarta Sans", size=12)),
        yaxis=dict(tickfont=dict(family="Plus Jakarta Sans", size=12))
    )
    
    return fig
