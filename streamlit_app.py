import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")
st.title("💳 Credit Risk Analyzer")
st.write("Predict borrower default risk and evaluate regression & classification models.")

# -------------------------------
# DATA UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.write("### Data Preview")
    st.dataframe(df.head())

    st.write("### Shape of Data:", df.shape)
    st.write("### Columns:", list(df.columns))

    # -------------------------------
    # DATA ENCODING & CLEANING
    # -------------------------------
    st.header("🧩 Data Preprocessing")

    if st.button("Run Encoding & Modeling"):
        try:
            df_clean = df.dropna()

            # Example categorical columns (change if names differ)
            categorical_nominal = ['person_home_ownership', 'loan_intent', 'age_group']
            df_encoded = pd.get_dummies(df_clean, columns=categorical_nominal, drop_first=True)

            grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
            if 'loan_grade' in df_encoded.columns:
                df_encoded['loan_grade_encoded'] = df_clean['loan_grade'].map(grade_mapping)

            if 'cb_person_default_on_file' in df_encoded.columns:
                df_encoded['cb_person_default_on_file_encoded'] = (df_clean['cb_person_default_on_file'] == 'Y').astype(int)
            
            if 'employment_stability' in df_encoded.columns:
                df_encoded['employment_stability_encoded'] = df_clean['employment_stability'].astype(int)

            df_encoded = df_encoded.drop(
                [col for col in ['loan_grade', 'cb_person_default_on_file', 'employment_stability'] if col in df_encoded.columns],
                axis=1,
                errors='ignore'
            )

            st.success("✅ Encoding completed successfully!")
            st.write("Encoded Dataset Shape:", df_encoded.shape)
            st.dataframe(df_encoded.head())

            # -------------------------------
            # FEATURE SELECTION
            # -------------------------------
            st.header("🔍 Feature Selection")
            target_cols = st.multiselect("Select Target Variable(s):", df_encoded.columns)
            
            if len(target_cols) == 1:
                y = df_encoded[target_cols[0]]
                X = df_encoded.drop(columns=target_cols)
            elif len(target_cols) > 1:
                st.warning("Please select only one target variable at a time.")
                st.stop()
            else:
                st.warning("Please select a target variable.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # -------------------------------
            # REGRESSION MODELS
            # -------------------------------
            st.header("📉 Regression Models")
            regression_models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0, random_state=42),
                'Lasso Regression': Lasso(alpha=1.0, random_state=42),
                'ElasticNet': ElasticNet(alpha=1.0, random_state=42),
                'Polynomial Regression': LinearRegression(),
                'RANSAC': RANSACRegressor(random_state=42),
                'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
                'Random Forest Regressor': RandomForestRegressor(random_state=42),
                'SVR': SVR()
            }

            results = []

            for name, model in regression_models.items():
                try:
                    if name == 'Polynomial Regression':
                        poly = PolynomialFeatures(degree=2, include_bias=False)
                        X_train_poly = poly.fit_transform(X_train)
                        X_test_poly = poly.transform(X_test)
                        model.fit(X_train_poly, y_train)
                        y_pred = model.predict(X_test_poly)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    n, p = len(y_test), X_train.shape[1]
                    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

                    results.append([name, mae, mse, rmse, r2, adj_r2])
                except Exception as e:
                    results.append([name, "Error", str(e), "", "", ""])

            reg_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2", "Adjusted R2"])
            st.dataframe(reg_df)

            # -------------------------------
            # CLASSIFICATION MODELS
            # -------------------------------
            st.header("🧠 Classification Models")
            classification_models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Naive Bayes': GaussianNB(),
                'SVM': SVC(random_state=42, probability=True),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }

            class_results = []
            y_class = (y > y.mean()).astype(int)  # simple conversion for demonstration
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

            for name, model in classification_models.items():
                model.fit(X_train_c, y_train_c)
                y_pred = model.predict(X_test_c)
                y_pred_proba = model.predict_proba(X_test_c)[:, 1] if hasattr(model, 'predict_proba') else None

                accuracy = accuracy_score(y_test_c, y_pred)
                precision = precision_score(y_test_c, y_pred)
                recall = recall_score(y_test_c, y_pred)
                f1 = f1_score(y_test_c, y_pred)
                auc_roc = roc_auc_score(y_test_c, y_pred_proba) if y_pred_proba is not None else np.nan

                class_results.append([name, accuracy, precision, recall, f1, auc_roc])

            class_df = pd.DataFrame(class_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"])
            st.dataframe(class_df)

            # -------------------------------
            # PCA ANALYSIS
            # -------------------------------
            st.header("📊 PCA Analysis")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)

            explained_var = pca.explained_variance_ratio_
            cumsum = np.cumsum(explained_var)
            n_components_95 = np.argmax(cumsum >= 0.95) + 1

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(range(1, len(explained_var) + 1), explained_var, 'bo-')
            ax[0].set_title("Explained Variance by Component")
            ax[0].set_xlabel("Principal Component")
            ax[0].set_ylabel("Explained Variance Ratio")

            ax[1].plot(range(1, len(cumsum) + 1), cumsum, 'ro-')
            ax[1].axhline(y=0.95, color='k', linestyle='--')
            ax[1].set_title("Cumulative Explained Variance")
            ax[1].set_xlabel("Number of Components")
            ax[1].set_ylabel("Cumulative Variance")

            st.pyplot(fig)
            st.success(f"✅ {n_components_95} components explain 95% of variance.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

else:
    st.info("👆 Please upload a CSV file to start analysis.")
