import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Streamlit Page Config
st.set_page_config(
    page_title="Customer Spending Score Predictor",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Customer Spending Score Prediction")
st.markdown("Hard Voting Ensemble Classifier using Random Forest, Extra Trees, XGBoost, and LightGBM.")

# =========================================================
# EXACT DATA PREPROCESSING & MODEL TRAINING PIPELINE
# =========================================================
@st.cache_resource
def run_model_pipeline():
    # Load dataset from Hugging Face and convert to pandas DataFrame
    ds = load_dataset("jason1966/abisheksudarshan_customer-segmentation")
    df = ds['train'].to_pandas()
    df.columns = [c.strip() for c in df.columns]

    # Impute missing values for categorical and numerical features
    df['Ever_Married'] = df['Ever_Married'].fillna('Unknown')
    df['Graduated'] = df['Graduated'].fillna('Unknown')
    df['Profession'] = df['Profession'].fillna('Unknown')
    df['Work_Experience'] = df['Work_Experience'].fillna(df['Work_Experience'].median())
    if 'Family_Size' in df.columns:
        df['Family_Size'] = df['Family_Size'].fillna(df['Family_Size'].median())

    # Feature engineering: create interaction terms, ratios, and binned features
    df_fe = df.copy()

    df_fe['Experience_Age_Ratio'] = df_fe['Work_Experience'] / df_fe['Age']
    df_fe['Is_Married'] = (df_fe['Ever_Married'] == 'Yes').astype(int)
    df_fe['Age_Married_Interact'] = df_fe['Age'] * df_fe['Is_Married']
    df_fe['Age_Group'] = pd.cut(df_fe['Age'], bins=[0, 25, 40, 60, 100], labels=['Young', 'Adult', 'Mid_Age', 'Senior'])
    df_fe['Career_Stage'] = pd.cut(df_fe['Work_Experience'], bins=[-1, 1, 5, 10, 50], labels=['Junior', 'Mid', 'Senior', 'Veteran'])

    df_fe['Graduated_Int'] = (df_fe['Graduated'] == 'Yes').astype(int)
    df_fe['Age_Graduated_Interact'] = df_fe['Age'] * df_fe['Graduated_Int']
    df_fe['Married_Profession'] = df_fe['Ever_Married'].astype(str) + "_" + df_fe['Profession'].astype(str)

    if 'Family_Size' in df_fe.columns:
        df_fe['Age_Per_Family_Member'] = df_fe['Age'] / (df_fe['Family_Size'] + 1)

    # Separate features (X) and target variable (y)
    features_to_drop = ['Spending_Score', 'Customer_ID', 'ID']
    features_to_drop = [col for col in features_to_drop if col in df_fe.columns]

    X = df_fe.drop(columns=features_to_drop)
    y_raw = df_fe['Spending_Score']

    # Encode target categorical labels into integer values
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)

    # One-hot encode categorical feature columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Split data into train and test sets (80/20 ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize numerical features
    num_cols = X_encoded.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Initialize base models for ensemble
    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
    )

    et_model = ExtraTreesClassifier(
        n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
    )

    xgb_model = XGBClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'
    )

    lgb_model = LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1
    )

    # Build Hard Voting Ensemble Classifier
    voting_hard_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('et', et_model),
            ('xgb', xgb_model),
            ('lgbm', lgb_model)
        ],
        voting='hard'
    )

    # Train Hard Voting Ensemble model
    voting_hard_model.fit(X_train, y_train)

    # Evaluate model predictions
    y_pred_hard = voting_hard_model.predict(X_test)

    target_names = target_encoder.classes_
    acc_hard = accuracy_score(y_test, y_pred_hard)
    report_text = classification_report(y_test, y_pred_hard, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred_hard)

    return (voting_hard_model, scaler, target_encoder, X_encoded.columns, 
            num_cols, acc_hard, report_text, cm, df)

# Load pipeline once
(voting_hard_model, scaler, target_encoder, feature_columns, 
 num_cols, acc_hard, report_text, cm, df_raw) = run_model_pipeline()

# =========================================================
# STREAMLIT USER INTERFACE
# # =========================================================

st.subheader("Input Customer Details for Live Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ['Select Option', 'Male', 'Female'])
    age = st.slider("Age", min_value=18, max_value=90, value=35)
    ever_married = st.selectbox("Ever Married?", ['Select Option', 'Yes', 'No', 'Unknown'])

with col2:
    graduated = st.selectbox("Graduated?", ['Select Option', 'Yes', 'No', 'Unknown'])
    profession = st.selectbox("Profession", ['Select Option', 'Artist', 'Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Artist', 'Executive', 'Doctor', 'Homemaker', 'Marketing', 'Unknown'])
    work_exp = st.number_input("Work Experience (Years)", min_value=0, max_value=55, value=3)

with col3:
    family_size = st.number_input("Family Size", min_value=1, max_value=10, value=2)

if st.button("🚀 Predict Spending Score", type="primary"):
    # Check if any field still has the default "Select option"
    if gender == "Select option" or ever_married == "Select option" or graduated == "Select option" or profession == "Select option":
        st.error("⚠️ Please select valid options for all dropdown fields before predicting!")
    elif work_exp > age - 16:
        st.error("⚠️ Work experience cannot exceed (Age - 16). Please input right numbers!")
    else:
        # Build single customer input DataFrame
        input_df = pd.DataFrame([{
            'Gender': gender,
            'Ever_Married': ever_married,
            'Age': age,
            'Graduated': graduated,
            'Profession': profession,
            'Work_Experience': work_exp,
            'Family_Size': family_size
        }])

    
    # Apply exact feature engineering steps
    input_df['Experience_Age_Ratio'] = input_df['Work_Experience'] / input_df['Age']
    input_df['Is_Married'] = (input_df['Ever_Married'] == 'Yes').astype(int)
    input_df['Age_Married_Interact'] = input_df['Age'] * input_df['Is_Married']
    input_df['Age_Group'] = pd.cut(input_df['Age'], bins=[0, 25, 40, 60, 100], labels=['Young', 'Adult', 'Mid_Age', 'Senior'])
    input_df['Career_Stage'] = pd.cut(input_df['Work_Experience'], bins=[-1, 1, 5, 10, 50], labels=['Junior', 'Mid', 'Senior', 'Veteran'])

    input_df['Graduated_Int'] = (input_df['Graduated'] == 'Yes').astype(int)
    input_df['Age_Graduated_Interact'] = input_df['Age'] * input_df['Graduated_Int']
    input_df['Married_Profession'] = input_df['Ever_Married'].astype(str) + "_" + input_df['Profession'].astype(str)
    input_df['Age_Per_Family_Member'] = input_df['Age'] / (input_df['Family_Size'] + 1)

    # Categorical One-Hot Encoding
    cat_cols_input = input_df.select_dtypes(include=['object', 'category']).columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=cat_cols_input, drop_first=True)

    # Schema Alignment with training set
    input_aligned = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in input_encoded.columns:
        if col in input_aligned.columns:
            input_aligned[col] = input_encoded[col]

    # Scaling numerical features
    input_aligned[num_cols] = scaler.transform(input_aligned[num_cols])

    # Model Prediction
    pred_class_idx = voting_hard_model.predict(input_aligned)[0]
    pred_label = target_encoder.inverse_transform([pred_class_idx])[0]

    st.markdown("---")
    st.subheader("Prediction Result")
    if pred_label == "High":
        st.success(f"Predicted Spending Score Tier: **{pred_label}** 💎")
    elif pred_label == "Average":
        st.info(f"Predicted Spending Score Tier: **{pred_label}** 📈")
    else:
        st.warning(f"Predicted Spending Score Tier: **{pred_label}** 🏷️")
