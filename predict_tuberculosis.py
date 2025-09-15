
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the trained Random Forest model and the StandardScaler object
loaded_rf_model = joblib.load('rf_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Identify the categorical feature column
categorical_features = ['Entity']

# Identify the numerical feature columns that were scaled (excluding the target)
# Make sure these columns match the ones used during training and scaling.
numerical_features_to_scale = [
    'Total (estimated) polio cases',
    'Reported cases of guinea worm disease in humans',
    'Number of new cases of rabies, in both sexes aged all ages',
    'Number of new cases of malaria, in both sexes aged all ages',
    'Number of new cases of hiv/aids, in both sexes aged all ages'
]


# Create a ColumnTransformer to apply OneHotEncoder to the categorical column
# Note: When using the loaded pipeline for prediction, the preprocessor is
# already part of the pipeline and fitted. We redefine it here primarily
# for clarity on the preprocessing steps involved if one were to build
# the prediction function from scratch or use the preprocessor separately.
# However, for prediction using the loaded pipeline, this separate preprocessor
# definition is not strictly necessary as the pipeline handles it internally.
# We keep it here to show the transformations applied.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', loaded_scaler, numerical_features_to_scale)
    ],
    remainder='passthrough' # Keep other columns ('Year' in this case)
)


def predict_tuberculosis_cases(new_data_df):
  """
  Predicts the number of new tuberculosis cases using the trained Random Forest model.

  Args:
    new_data_df: A pandas DataFrame containing the new data for prediction.
                   It should have the same columns as the features used during
                   training (excluding the target), including 'Entity', 'Year',
                   and the numerical features listed in numerical_features_to_scale.

  Returns:
    A numpy array containing the predicted number of tuberculosis cases.
  """
  # The loaded_rf_model is a pipeline that includes the preprocessor.
  # We can directly call predict on the pipeline with the raw new data.
  predictions = loaded_rf_model.predict(new_data_df)

  return predictions

# Example Usage:
if __name__ == "__main__":
    # Create sample new data
    sample_new_data_1 = pd.DataFrame({
        'Entity': ['Afghanistan', 'World'],
        'Year': [2024, 2024],
        'Total (estimated) polio cases': [10.0, 5000.0],
        'Reported cases of guinea worm disease in humans': [0.0, 100.0],
        'Number of new cases of rabies, in both sexes aged all ages': [5.0, 500.0],
        'Number of new cases of malaria, in both sexes aged all ages': [1000.0, 1000000.0],
        'Number of new cases of hiv/aids, in both sexes aged all ages': [50.0, 50000.0]
    })

    sample_new_data_2 = pd.DataFrame({
        'Entity': ['India', 'United States'],
        'Year': [2023, 2025],
        'Total (estimated) polio cases': [0.0, 0.0],
        'Reported cases of guinea worm disease in humans': [0.0, 0.0],
        'Number of new cases of rabies, in both sexes aged all ages': [10.0, 2.0],
        'Number of new cases of malaria, in both sexes aged all ages': [50000.0, 100.0],
        'Number of new cases of hiv/aids, in both sexes aged all ages': [10000.0, 500.0]
    })


    # Make predictions using the prediction function for sample_new_data_1
    sample_predictions_1 = predict_tuberculosis_cases(sample_new_data_1)

    # Print the predictions for sample_new_data_1
    print("Predictions for sample new data 1:")
    print(sample_predictions_1)

    print("-" * 30) # Separator

    # Make predictions using the prediction function for sample_new_data_2
    sample_predictions_2 = predict_tuberculosis_cases(sample_new_data_2)

    # Print the predictions for sample_new_data_2
    print("Predictions for sample new data 2:")
    print(sample_predictions_2)
