# Cardiovascular Prognostic Model App

import streamlit as st
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler

# Sidebar navigation with dropdown
options = st.sidebar.selectbox('Select a Page:', ["Overview", "Interpretability Engine"])

# Load the models
logistic_model = joblib.load("logistic_model_optuna.pkl")
gb_model = joblib.load("gb_model_optuna.pkl")

# Preprocessing function for the data
def preprocess_data(df):
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({1: 0, 2: 1})  # Map male to 0, female to 1
    
    if 'cholesterol' in df.columns:
        df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 0 else 0)  # Binarize cholesterol
    
    if 'gluc' in df.columns:
        df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 0 else 0)  # Binarize glucose

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median())
    
    return df

def load_data():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file, sep=";")
            if len(data.columns) == 1:
                data = pd.read_csv(uploaded_file, sep=",")
            return preprocess_data(data)
        except Exception as e:
            st.error(f"Error loading the data: {e}")
            return None
    return None

def display_lime_explanation(lime_explanation, prediction, patient_stats):
    st.write("### LIME Explanation (Feature Contribution)")
    
    if 'age' in patient_stats.columns:
        patient_stats['age (years)'] = patient_stats['age'] / 365.25

    st.write("#### Patient Stats Summary:")
    st.write(patient_stats)

    explanation_df = pd.DataFrame(lime_explanation.as_list(), columns=['Feature', 'Contribution'])
    explanation_df['Contribution'] = explanation_df['Contribution'].apply(lambda x: f"{x:.4f}")
    
    st.table(explanation_df)

    sorted_explanation = sorted(lime_explanation.as_list(), key=lambda x: abs(x[1]), reverse=True)
    most_contributing = sorted_explanation[:3]
    least_contributing = sorted_explanation[-3:]

    st.write("#### Detailed Explanation (Narrative):")
    
    prediction_text = "high" if prediction == 1 else "low"
    explanation_paragraph = f"The predicted cardiovascular outcome for this patient is **{prediction_text} risk** for the following reasons:\n\n"

    explanation_paragraph += "**Features that contributed the most to this prediction:**\n"
    for feature, contribution in most_contributing:
        sign = "positively" if contribution > 0 else "negatively"
        explanation_paragraph += f"- The feature '{feature}' contributed {sign} to the model's overall prediction. Its impact on the prediction was {abs(contribution):.4f}.\n"

    explanation_paragraph += "\n**Features that contributed the least to this prediction:**\n"
    for feature, contribution in least_contributing:
        sign = "positively" if contribution > 0 else "negatively"
        explanation_paragraph += f"- The feature '{feature}' contributed {sign} to the prediction, with an impact of {abs(contribution):.4f}.\n"

    explanation_paragraph += """
    Features can either push the model toward predicting a higher or lower risk. A **positive contribution** means the feature supports the current prediction, making it stronger—whether that’s high or low risk. 
    A **negative contribution** means the feature works against the current prediction, pulling it in the opposite direction. These contributions only show how the model adjusts its prediction, not direct causes of health risk.
    """

    st.write(explanation_paragraph)

if options == "Overview":
    st.title("Cardiovascular Prognostic Model")
    st.header("Dataset Summary")
    st.write("""
    This dataset contains several features related to cardiovascular health, including age, gender, height, weight, blood pressure, cholesterol levels, glucose levels, and lifestyle habits such as smoking, alcohol intake, and physical activity. The target variable `cardio` indicates whether a person has cardiovascular disease (1) or not (0).
    """)
    st.header("Model Comparison")
    st.write("""
    We trained two models for predicting cardiovascular disease:
    - **Logistic Regression**: A linear model that calculates the probability of cardiovascular disease based on the input features. This model is simple but effective for binary classification problems.
    - **Gradient Boosting Trees**: A more advanced model that builds an ensemble of decision trees to make predictions. Each tree is built to correct the mistakes of the previous trees, capturing more complex patterns.
    """)
    logistic_metrics = {
        "Model": "Logistic Regression",
        "Accuracy": 0.78,
        "Precision": 0.76,
        "Recall": 0.79,
        "F1-Score": 0.77,
        "AUC": 0.84
    }

    gb_metrics = {
        "Model": "Gradient Boosting",
        "Accuracy": 0.82,
        "Precision": 0.80,
        "Recall": 0.83,
        "F1-Score": 0.81,
        "AUC": 0.87
    }

    performance_df = pd.DataFrame([logistic_metrics, gb_metrics])
    st.dataframe(performance_df)

    st.write("The **Gradient Boosting Trees** model performed slightly better in terms of accuracy and AUC compared to the **Logistic Regression** model.")
    st.header("Conclusion")
    st.write("""
    Both models are effective in predicting cardiovascular disease. Gradient Boosting Trees, while more complex, handle feature interactions better, leading to improved performance. Logistic Regression, however, provides simpler and more interpretable predictions.
    """)

elif options == "Interpretability Engine":
    st.title("Interpretability Engine")
    st.write("""
    **LIME** (Local Interpretable Model-agnostic Explanations) can be used to explain individual predictions by building simple interpretable models for each prediction.
    - **Logistic Regression**: LIME helps clarify which features (e.g., age, cholesterol levels) drove the prediction, making it easier to understand the model's decision.
    - **Gradient Boosting Trees**: For this more complex model, LIME creates simple explanations by approximating non-linear relationships, helping to explain the influence of specific features on the prediction.

    By applying LIME, we can make predictions from both models more transparent and generate explanations that are easier to understand.
    """)
    
    st.write("""
    **Instructions**:
    1. Select a model in the sidebar (either Logistic Regression or Gradient Boosting).
    2. Upload the **cardio_train** dataset using the 'Upload CSV file' button.
    3. Select a **patient ID** (from the 'ID' column of the dataset) using the dropdown.
    4. Hit the **Generate Prediction Explanation** button to generate an interpretability explanation.
    """)

    model_choice = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Gradient Boosting"])

    df = load_data()
    if df is not None:
        if 'cardio' not in df.columns:
            st.write("Error: The dataset must contain a 'cardio' column (target variable).")
        else:
            X = df.drop(columns=['cardio'])
            y = df['cardio']

            model = logistic_model if model_choice == "Logistic Regression" else gb_model

            scaler = StandardScaler()
            columns_to_scale = X.columns.difference(['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
            X_scaled = X.copy()
            X_scaled[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

            available_ids = df.iloc[:, 0].tolist()
            patient_id = st.selectbox("Choose a patient ID for explanation", available_ids)

            if st.button("Generate Prediction Explanation"):
                patient_stats = X.loc[df[df.iloc[:, 0] == patient_id].index[0]].to_frame().T
                if 'age' in patient_stats.columns:
                    patient_stats['age (years)'] = patient_stats['age'] / 365.25

                prediction = model.predict([X_scaled.loc[df[df.iloc[:, 0] == patient_id].index[0]]])[0]

                categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

                explainer = LimeTabularExplainer(X_scaled.values, feature_names=X.columns, class_names=['No Cardio', 'Cardio'], 
                                                 categorical_features=[X.columns.get_loc(f) for f in categorical_features], 
                                                 discretize_continuous=True)
                
                exp = explainer.explain_instance(X_scaled.loc[df[df.iloc[:, 0] == patient_id].index[0]].values, model.predict_proba, num_features=5)

                display_lime_explanation(exp, prediction, patient_stats)
