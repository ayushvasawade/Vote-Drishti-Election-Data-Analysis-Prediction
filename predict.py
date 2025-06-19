import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from json import dumps as jsonify

def predict_winner(data):
    data_raw = pd.DataFrame(data)
    data_preprocessed = data_raw.copy()
    final_results_dict = {}
    
    MODEL_FILENAME = 'model/final_model.joblib'
    SCALER_FILENAME = 'model/min_max_scaler.joblib'
    ENCODER_FILENAME = 'model/ordinal_encoder.joblib'

    FEATURE_COLUMNS = ['STATE', 'PARTY', 'GENDER', 'CRIMINALCASES', 'AGE', 'CATEGORY', 'EDUCATION', 'ASSETS', 'LIABILITIES', 'TOTAL ELECTORS']
    CATEGORICAL_FEATURES = ['STATE', 'PARTY', 'GENDER', 'CATEGORY', 'EDUCATION']
    NUMERICAL_FEATURES = ['CRIMINALCASES', 'AGE', 'ASSETS', 'LIABILITIES', 'TOTAL ELECTORS']


    try:
        print(f"Loading model from {MODEL_FILENAME}...")
        model = joblib.load(MODEL_FILENAME)
        print("Model loaded successfully.")

        print(f"Loading scaler from {SCALER_FILENAME}...")
        scaler = joblib.load(SCALER_FILENAME)
        print("Scaler loaded successfully.")

        print(f"Loading encoder from {ENCODER_FILENAME}...")
        encoder = joblib.load(ENCODER_FILENAME)
        print("Encoder loaded successfully.")

    except FileNotFoundError as e:
        print(f"\nError loading file: {e}")
        print("Please ensure the model, scaler, and encoder files (.joblib) exist in the same directory as this script.")
        exit()
    except Exception as e:
        print(f"\nAn error occurred during loading: {e}")
        exit()


    try:
        print("\nPreprocessing new data...")
        
        for cat in CATEGORICAL_FEATURES:
            data_preprocessed[cat] = data_preprocessed[cat].astype('category')


        data_preprocessed[CATEGORICAL_FEATURES] = encoder.transform(data_preprocessed[CATEGORICAL_FEATURES])
        print("Categorical encoding applied.")


        data_preprocessed[NUMERICAL_FEATURES] = scaler.transform(data_preprocessed[NUMERICAL_FEATURES])
        print("Numerical scaling applied.")


        data_preprocessed = data_preprocessed[FEATURE_COLUMNS]

        print("\n--- Preprocessed New Data (Ready for Prediction) ---")
        print(data_preprocessed)


        print("\nMaking predictions...")
        new_predictions = model.predict(data_preprocessed)
        print("Predictions completed.")




        data_raw['PREDICTED_WINNER (0=Loss, 1=Win)'] = new_predictions

        if hasattr(model, "predict_proba"):
            print("\nCalculating prediction probabilities...")
            prediction_probabilities = model.predict_proba(data_preprocessed)
            print("Probabilities calculated.")
            data_raw['Probability_Loss (Class 0)'] = prediction_probabilities[:, 0]
            data_raw['Probability_Win (Class 1)'] = prediction_probabilities[:, 1]
        else:
            data_raw['Probability_Loss (Class 0)'] = np.nan
            data_raw['Probability_Win (Class 1)'] = np.nan


        print("\nBuilding final results dictionary...")
        for index, row in data_raw.iterrows():
            candidate_data = row.to_dict()
            final_results_dict[index] = candidate_data

        print("Final results dictionary created.")

        print("\n--- Final Results Dictionary ---")
        print(jsonify(final_results_dict, indent=4))


    except Exception as e:
        print(f"\nAn error occurred during preprocessing or prediction: {e}")
        print("Ensure the new data has the correct columns and data types, and that categories/values are consistent with the training data.")

    return final_results_dict
