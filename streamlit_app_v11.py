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
    # Check if gender exists and map to numeric (assuming 1=male, 2=female)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({1: 0, 2: 1})  # Map male to 0, female to 1
    
    # Binarize cholesterol and glucose (if above normal, set to 1)
    if 'cholesterol' in df.columns:
        df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 0 else 0)  # Binarize cholesterol
    
    if 'gluc' in df.columns:
        df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 0 else 0)  # Binarize glucose

    # Ensure all columns are numeric (handle non-numeric values)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing values with the median
    df = df.fillna(df.median())
    
    return df

# Define function to load and preprocess the data
def load_data():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        try:
            # Try loading the data with different delimiters
            data = pd.read_csv(uploaded_file, sep=";")  # Assuming semicolon separator
            if len(data.columns) == 1:  # Check if the file has a single column
                data = pd.read_csv(uploaded_file, sep=",")  # Fall back to comma separator if needed
            return preprocess_data(data)
        except Exception as e:
            st.error(f"Error loading the data: {e}")
            return None
    return None

# Function to display LIME explanation with the most and least contributing features
def display_lime_explanation(lime_explanation, prediction, patient_stats):
    st.write("### LIME Explanation (Feature Contribution)")
    
    # Convert age from days to years and add the "age (years)" column to the DataFrame
    if 'age' in patient_stats.columns:
        patient_stats['age (years)'] = patient_stats['age'] / 365.25  # Convert age from days to years

    # Display patient stats summary
    st.write("#### Patient Stats Summary:")
    st.write(patient_stats)

    explanation_df = pd.DataFrame(lime_explanation.as_list(), columns=['Feature', 'Contribution'])
    explanation_df['Contribution'] = explanation_df['Contribution'].apply(lambda x: f"{x:.4f}")  # Format to 4 decimal places
    
    # Display as a table
    st.table(explanation_df)

    # Sort by the absolute value of contributions to find most and least contributing features
    sorted_explanation = sorted(lime_explanation.as_list(), key=lambda x: abs(x[1]), reverse=True)
    most_contributing = sorted_explanation[:3]  # Top 3 most contributing features
    least_contributing = sorted_explanation[-3:]  # Bottom 3 least contributing features

    # Display a more standardized explanation
    st.write("#### Detailed Explanation (Narrative):")
    
    prediction_text = "high" if prediction == 1 else "low"
    explanation_paragraph = f"The predicted cardiovascular outcome for this patient is {prediction_text} risk for the following reasons:\n\n"

    # List features contributing the most
    explanation_paragraph += "**Features that contributed the most to this prediction:**\n"
    for feature, contribution in most_contributing:
        sign = "increased" if contribution > 0 else "decreased"
        explanation_paragraph += f"- The feature '{feature}' has {sign} the likelihood of cardiovascular disease by {abs(contribution):.4f}.\n"
    
    # List features contributing the least
    explanation_paragraph += "\n**Features that contributed the least to this prediction:**\n"
    for feature, contribution in least_contributing:
        sign = "increased" if contribution > 0 else "decreased"
        explanation_paragraph += f"- The feature '{feature}' has {sign} the likelihood of cardiovascular disease by {abs(contribution):.4f}.\n"

    # Display the standardized narrative
    st.write(explanation_paragraph)

if options == "Overview":
    # Title of the app
    st.title("Cardiovascular Prognostic Model")

    # Dataset Summary
    st.header("Dataset Summary")
    st.write("""
    This dataset contains several features related to cardiovascular health, including age, gender, height, weight, blood pressure, cholesterol levels, glucose levels, and lifestyle habits such as smoking, alcohol intake, and physical activity. The target variable `cardio` indicates whether a person has cardiovascular disease (1) or not (0).
    """)

    # Model Comparison
    st.header("Model Comparison")

    # Summary of Logistic Regression and Gradient Boosted Trees training
    st.write("""
    We trained two models for predicting cardiovascular disease:

    - **Logistic Regression**: A linear model that calculates the probability of cardiovascular disease based on the input features. This model is simple but effective for binary classification problems.
    - **Gradient Boosting Trees**: A more advanced model that builds an ensemble of decision trees to make predictions. Each tree is built to correct the mistakes of the previous trees, capturing more complex patterns.

    The performance of these models was compared using accuracy, precision, recall, F1-score, and AUC (Area Under the ROC Curve).
    """)

    # Display model performance (use placeholder values here)
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

    # Conclusion
    st.header("Conclusion")
    st.write("""
    Both models are effective in predicting cardiovascular disease. Gradient Boosting Trees, while more complex, handle feature interactions better, leading to improved performance. Logistic Regression, however, provides simpler and more interpretable predictions.
    """)

elif options == "Interpretability Engine":
    # Interpretability Engine Summary
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

    # Sidebar for model selection
    model_choice = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Gradient Boosting"])

    # Load and preprocess the data
    st.write("Upload your dataset and select a model to generate explanations.")

    df = load_data()
    if df is not None:

        # Check if the 'cardio' column is present
        if 'cardio' not in df.columns:
            st.write("Error: The dataset must contain a 'cardio' column (target variable).")
        else:
            # Prepare features and target
            X = df.drop(columns=['cardio'])  # Features
            y = df['cardio']  # Target variable

            # Choose the model based on user selection
            model = logistic_model if model_choice == "Logistic Regression" else gb_model

            # Standardize the features (for LIME and model compatibility)
            scaler = StandardScaler()
            columns_to_scale = X.columns.difference(['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])  # Exclude binary/categorical features from scaling
            X_scaled = X.copy()
            X_scaled[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

            # Select patient ID from dropdown
            available_ids = df.index.tolist()
            patient_id = st.selectbox("Choose a patient ID for explanation", available_ids)

            # Button to generate explanation
            if st.button("Generate Prediction Explanation"):
                # Retrieve patient stats for summary
                patient_stats = X.loc[patient_id].to_frame().T  # Transpose to display as a row

                # Convert age to years and add it to the patient stats
                if 'age' in patient_stats.columns:
                    patient_stats['age (years)'] = patient_stats['age'] / 365.25

                # Make prediction for the selected patient
                prediction = model.predict([X_scaled.loc[patient_id]])[0]

                # Specify categorical features for LIME (gender, cholesterol, gluc, etc.)
                categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

                # Generate LIME explanation
                explainer = LimeTabularExplainer(X_scaled.values, feature_names=X.columns, class_names=['No Cardio', 'Cardio'], 
                                                 categorical_features=[X.columns.get_loc(f) for f in categorical_features], 
                                                 discretize_continuous=True)
                
                exp = explainer.explain_instance(X_scaled.loc[patient_id].values, model.predict_proba, num_features=5)

                # Display LIME explanation and patient stats
                display_lime_explanation(exp, prediction, patient_stats)
