import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor, LogisticRegression
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="💳 Credit Risk Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        color: #111;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- TITLE --------------------
st.title("💳 Credit Risk Analyzer")
st.caption("Analyze borrower profiles, predict default risk, and visualize results interactively.")

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ App Controls")

uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV Dataset", type=["csv"])
use_sample = st.sidebar.checkbox("Use Sample Dataset")

# -------------------- DATA LOADING --------------------
if use_sample:
    df = pd.read_csv("3a7927f5-91dd-4869-9177-e61360ed72ae.csv")
    st.sidebar.success("✅ Using sample dataset from repo.")
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset uploaded successfully!")
else:
    df = None

# -------------------- MAIN CONTENT --------------------
if df is not None:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------- ENCODING --------------------
    st.subheader("🧹 Data Preprocessing")
    df_clean = df.copy()

    # Manual encoding example
    if 'loan_grade' in df_clean.columns:
        grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        df_clean['loan_grade'] = df_clean['loan_grade'].map(grade_mapping)

    # Auto-encode categorical columns
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = df_clean[col].astype('category').cat.codes

    # Drop rows with missing values
    df_clean.dropna(inplace=True)

    st.write("✅ Categorical columns encoded and missing values removed.")
    st.dataframe(df_clean.head())

    # -------------------- SELECT TARGET COLUMN --------------------
    target_column = "loan_status"
    if target_column not in df_clean.columns:
        st.error(f"❌ Target column '{target_column}' not found in dataset.")
        st.stop()

    # Encode target if needed
    if df_clean[target_column].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(df_clean[target_column])
    else:
        y = df_clean[target_column]

    X = df_clean.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.sidebar.markdown("### Split Info")
    st.sidebar.write(f"Train samples: {X_train.shape[0]}")
    st.sidebar.write(f"Test samples: {X_test.shape[0]}")

    # -------------------- DEBUG INFO --------------------
    st.sidebar.write("🔍 Target Class Distribution:")
    st.sidebar.write(pd.Series(y).value_counts())

    # -------------------- TABS --------------------
    tab1, tab2, tab3 = st.tabs(["📈 Regression", "🔍 Classification", "🧠 PCA Analysis"])

    # ======================================================
    # REGRESSION
    # ======================================================
    with tab1:
        st.subheader("📈 Regression Models Evaluation")

        regression_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=1.0, random_state=42),
            'ElasticNet': ElasticNet(alpha=1.0, random_state=42),
            'RANSAC': RANSACRegressor(random_state=42),
            'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
            'Random Forest Regressor': RandomForestRegressor(random_state=42),
            'SVR': SVR()
        }

        results = []
        progress = st.progress(0)
        total = len(regression_models)

        for i, (name, model) in enumerate(regression_models.items()):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                results.append([name, mae, mse, rmse, r2])
            except Exception as e:
                results.append([name, "Error", "Error", "Error", "Error"])
                st.warning(f"⚠️ {name} failed: {e}")
            progress.progress((i + 1) / total)

        results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R²"])
        st.dataframe(results_df)

        st.download_button(
            label="⬇️ Download Regression Results",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="regression_results.csv",
            mime="text/csv"
        )

        st.bar_chart(results_df.set_index("Model")["R²"])

    # ======================================================
    # CLASSIFICATION
    # ======================================================
    with tab2:
        st.subheader("🔍 Classification Models Evaluation")

        classification_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(probability=True, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

        results = []
        progress = st.progress(0)
        total = len(classification_models)

        for i, (name, model) in enumerate(classification_models.items()):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
                results.append([name, acc, prec, rec, f1, auc])
            except Exception as e:
                results.append([name, "Error", "Error", "Error", "Error", "Error"])
                st.warning(f"⚠️ {name} failed: {e}")
            progress.progress((i + 1) / total)

        results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])
        st.dataframe(results_df)

        st.download_button(
            label="⬇️ Download Classification Results",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="classification_results.csv",
            mime="text/csv"
        )

        st.bar_chart(results_df.set_index("Model")["Accuracy"])

    # ======================================================
    # PCA
    # ======================================================
    with tab3:
        st.subheader("🧠 Principal Component Analysis (PCA)")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        explained_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(explained_var >= 0.95) + 1
        st.write(f"✅ Number of components for 95% variance: **{
