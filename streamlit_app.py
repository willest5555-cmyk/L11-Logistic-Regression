import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(
    page_title="Iris 3D PCA & Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stSidebar"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2f9c98, #6b2677);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #64717d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(28, 38, 48, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .prediction-header {
        font-weight: 600;
        font-size: 1.3rem;
        color: #17212b;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2f9c98;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading & training
@st.cache_resource
def get_trained_pipeline():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    
    # Simple standardized Logistic Regression model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X, y)
    return model, iris.target_names, iris.feature_names

@st.cache_data
def get_pca_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    explained = pca.explained_variance_ratio_
    
    # Train model to get accuracy for reporting
    model, _, _ = get_trained_pipeline()
    accuracy = 0.967 # Benchmark test accuracy from regression_crisp_dm.py
    
    points = []
    for row_index, (coords, target) in enumerate(zip(X_pca, y)):
        points.append({
            "id": int(row_index),
            "species": str(target_names[target]),
            "classIndex": int(target),
            "pc1": round(float(coords[0]), 6),
            "pc2": round(float(coords[1]), 6),
            "pc3": round(float(coords[2]), 6),
        })
        
    payload = {
        "title": "First three PCA dimensions",
        "featureNames": list(iris.feature_names),
        "targetNames": list(target_names),
        "explainedVarianceRatio": [round(float(value), 6) for value in explained],
        "modelAccuracy": round(float(accuracy), 6),
        "points": points
    }
    return payload

# Load models and data
model, target_names, feature_names = get_trained_pipeline()
pca_data = get_pca_data()

# Layout
st.markdown('<div class="main-title">Iris 🌸 3D PCA & Classification Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An interactive machine learning demonstration using Logistic Regression and Three.js 3D Visualizer.</div>', unsafe_allow_html=True)

# Sidebar for interactive prediction
st.sidebar.markdown("### 🎛️ Input Iris Measurements")
sepal_len = st.sidebar.slider("Sepal Length (cm)", 4.3, 7.9, 5.8, step=0.1)
sepal_wid = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.4, 3.0, step=0.1)
petal_len = st.sidebar.slider("Petal Length (cm)", 1.0, 6.9, 4.35, step=0.1)
petal_wid = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3, step=0.1)

# Build sample dataframe
sample = pd.DataFrame(
    [[sepal_len, sepal_wid, petal_len, petal_wid]],
    columns=feature_names
)

# Predict
predicted_class_idx = model.predict(sample)[0]
predicted_class_name = target_names[predicted_class_idx]
pred_probs = model.predict_proba(sample)[0]

# Display Prediction Result in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔮 Prediction Result")
emoji_map = {"setosa": "🟡 Setosa", "versicolor": "🟢 Versicolor", "virginica": "🟣 Virginica"}
st.sidebar.markdown(f"Predicted Species:\n**{emoji_map.get(predicted_class_name, predicted_class_name)}**")

# Display probabilities
for name, prob in zip(target_names, pred_probs):
    st.sidebar.progress(float(prob), text=f"{name.capitalize()}: {prob*100:.1f}%")

# Main content tabs
tab1, tab2 = st.tabs(["🎮 3D PCA Scatter Plot", "📋 Model Details & CRISP-DM"])

with tab1:
    st.markdown("### 3D Interactive Principal Component Analysis")
    st.markdown("Rotate, zoom, and filter data points directly in the 3D canvas below. Hover over points to see details.")
    
    # Load and adapt the HTML for Streamlit embedding
    html_path = Path("iris_pca_3d_demo.html")
    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        
        # 1. Replace relative Three.js imports with CDN versions
        old_importmap = """  <script type="importmap">
    {
      "imports": {
        "three": "./node_modules/three/build/three.module.js"
      }
    }
  </script>"""
        
        new_importmap = """  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.158.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/"
      }
    }
  </script>"""
        
        html_content = html_content.replace(old_importmap, new_importmap)
        
        # Replace script imports
        old_imports = """    import * as THREE from "./node_modules/three/build/three.module.js";
    import { OrbitControls } from "./node_modules/three/examples/jsm/controls/OrbitControls.js";"""
        
        new_imports = """    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";"""
        
        html_content = html_content.replace(old_imports, new_imports)
        
        # 2. Inline the JSON data to avoid fetch requests
        old_fetch = """    try {
      const response = await fetch("./outputs/iris_pca_points.json");
      if (!response.ok) {
        throw new Error("The PCA data file was not found.");
      }
      data = await response.json();
    } catch (error) {
      showError("Run python iris_pca_3d_demo.py, then serve this folder with python -m http.server 8000.");
      throw error;
    }"""
        
        inlined_data = f"data = {json.dumps(pca_data)};"
        html_content = html_content.replace(old_fetch, inlined_data)
        
        # 3. Render HTML component
        components.html(html_content, height=650, scrolling=False)
    else:
        st.error("Error: `iris_pca_3d_demo.html` not found. Please run local files first.")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="prediction-header">📈 Classification Performance Metrics</div>', unsafe_allow_html=True)
        
        # Display Accuracy
        st.markdown(f"Baseline Test Accuracy: <span class=\"metric-value\">96.7%</span>", unsafe_allow_html=True)
        
        # Classification report in table format
        report_data = {
            "Precision": [1.00, 1.00, 0.91],
            "Recall": [1.00, 0.90, 1.00],
            "F1-Score": [1.00, 0.95, 0.95],
            "Support": [10, 10, 10]
        }
        report_df = pd.DataFrame(report_data, index=["Setosa", "Versicolor", "Virginica"])
        st.dataframe(report_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="prediction-header">⚙️ CRISP-DM Methodology</div>', unsafe_allow_html=True)
        st.markdown("""
        The development of this project strictly followed the **CRISP-DM** process:
        1. **Business Understanding**: Define the goal of predicting 3 Iris species based on size features.
        2. **Data Understanding**: Analyze structure and distributions of 150 Iris samples.
        3. **Data Preparation**: Split features with stratification and standardized variables.
        4. **Modeling**: Create a pipeline using Logistic Regression.
        5. **Evaluation**: Assess classification reports, precision/recall curves, and confusion matrix.
        6. **Deployment**: Package classification model and visualizer into this interactive web app.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="prediction-header">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
        confusion_matrix_path = Path("outputs/iris_confusion_matrix.png")
        if confusion_matrix_path.exists():
            st.image(str(confusion_matrix_path), use_container_width=True, caption="Confusion Matrix from Test Set (accuracy: 96.7%)")
        else:
            st.info("Run `python regression_crisp_dm.py` to generate the local confusion matrix plot.")
        st.markdown('</div>', unsafe_allow_html=True)
