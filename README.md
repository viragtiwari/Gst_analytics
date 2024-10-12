# Binary Classification Pipeline with XGBoost, LightGBM, and Stacking Ensemble

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Training the Models](#training-the-models)
  - [1. Running the Training Pipeline](#1-running-the-training-pipeline)
- [Inference](#inference)
  - [2. Using the Inference Function](#2-using-the-inference-function)
- [Model Evaluation](#model-evaluation)
- [Saving and Loading Models](#saving-and-loading-models)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project implements a comprehensive machine learning pipeline for binary classification tasks. It leverages powerful algorithms like **XGBoost** and **LightGBM**, optimizes their hyperparameters using **Optuna**, and combines them into a **Stacking Ensemble** to enhance predictive performance. Additionally, an inference function is provided to make predictions on new data using the trained ensemble model.

---

## Features

- **Data Loading & Merging**: Seamlessly integrates feature and target datasets based on unique identifiers.
- **Data Cleaning**: Handles missing values through mean imputation.
- **Feature Selection**: Removes low-importance features to reduce noise and improve model efficiency.
- **Class Balancing**: Addresses class imbalance via undersampling to prevent biased predictions.
- **Hyperparameter Optimization**: Utilizes Optuna for tuning XGBoost and LightGBM hyperparameters to maximize ROC AUC.
- **Model Training**: Trains optimized XGBoost and LightGBM models on the resampled dataset.
- **Ensemble Learning**: Combines individual models into a Stacking Ensemble using Logistic Regression as the meta-model.
- **Model Evaluation**: Provides comprehensive metrics and visualizations to assess model performance.
- **Model Persistence**: Saves trained models for future deployment and inference.
- **Inference Function**: Offers a straightforward method to generate predictions on new datasets.

---

## Prerequisites

Before getting started, ensure you have the following installed:

- **Python**: Version 3.7 or higher
- **pip**: Python package installer

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/binary-classification-pipeline.git
   cd binary-classification-pipeline
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**

   Ensure you have the latest versions of the necessary libraries:

   ```bash
   pip install --upgrade pandas numpy xgboost lightgbm optuna joblib matplotlib scikit-learn
   ```

---

## Project Structure

```
binary-classification-pipeline/
├── models/
│   ├── best_xgb_model.pkl
│   ├── best_lgb_model.pkl
│   └── stacking_ensemble_model.pkl
├── scripts/
│   ├── training_pipeline.py
│   └── inference.py
├── README.md
└── requirements.txt
```

- **data/**: Contains the training and testing datasets.
- **models/**: Stores the serialized trained models.
- **scripts/**: Includes the training pipeline and inference function scripts.
- **README.md**: This documentation file.
- **requirements.txt**: Lists all Python dependencies.

---

## Data Preparation

1. **Organize Your Data**

   Ensure your data is structured as follows:

   - **Training Features**: `X_Train_Data_Input.csv`
   - **Training Targets**: `Y_Train_Data_Target.csv`
   - **Testing Features**: `X_Test_Data_Input.csv`
   - **Testing Targets**: `Y_Test_Data_Target.csv`

   Each features CSV should contain an `'ID'` column to uniquely identify each record, which will be used to merge with the target CSVs.

2. **CSV Structure**

   - **Features CSV (`X_Train_Data_Input.csv` & `X_Test_Data_Input.csv`)**
     - Columns: `'ID'`, `'Column0'`, `'Column1'`, ..., `'Column21'` (Adjust as per your dataset)
   
   - **Targets CSV (`Y_Train_Data_Target.csv` & `Y_Test_Data_Target.csv`)**
     - Columns: `'ID'`, `'target'` (Ensure the target column is named `'target'` or adjust accordingly in the scripts)

3. **Place Data Files**

   Store all CSV files in the `data/` directory as shown in the project structure.

---

## Training the Models

The training pipeline encompasses data loading, preprocessing, hyperparameter optimization, model training, evaluation, and saving the trained models.

### 1. Running the Training Pipeline

1. **Navigate to the Scripts Directory**

   ```bash
   cd scripts
   ```

2. **Run the Training Script**

   Ensure that the `training_pipeline.py` script contains the comprehensive pipeline code provided earlier. Execute the script:

   ```bash
   python training_pipeline.py
   ```

   **Note**: The script performs the following operations:

   - **Data Loading & Merging**: Reads and merges the training and testing datasets based on the `'ID'` column.
   - **Handling Missing Values**: Imputes missing values using mean strategy.
   - **Feature Selection**: Drops predefined low-importance columns.
   - **Class Balancing**: Undersamples the majority class to achieve a desired balance.
   - **Hyperparameter Optimization**: Utilizes Optuna to tune hyperparameters for XGBoost and LightGBM.
   - **Model Training**: Trains the optimized XGBoost and LightGBM models on the resampled dataset.
   - **Ensemble Creation**: Combines the individual models into a Stacking Ensemble using Logistic Regression as the meta-model.
   - **Model Evaluation**: Outputs performance metrics and ROC curves.
   - **Model Saving**: Serializes and saves the trained models into the `models/` directory.

3. **Monitor the Output**

   The script will print progress messages, including shapes of datasets, class distributions, best hyperparameters found by Optuna, and evaluation metrics for each model.

4. **Review Saved Models**

   Upon successful execution, the following models will be saved in the `models/` directory:

   - `best_xgb_model.pkl`: Optimized XGBoost model
   - `best_lgb_model.pkl`: Optimized LightGBM model
   - `stacking_ensemble_model.pkl`: Stacking Ensemble model

---

## Inference

The inference function allows you to make predictions on new, unseen data using the trained Stacking Ensemble model.

### 2. Using the Inference Function

1. **Prepare New Data**

   Ensure your new data is in CSV format with the following structure:

   - **Columns**: `'ID'`, `'Column0'`, `'Column1'`, ..., `'Column21'` (Adjust as per your dataset)
   - **Note**: The CSV must contain an `'ID'` column to uniquely identify each record.

2. **Place the Inference Script**

   Ensure that the `inference.py` script contains the inference function provided earlier. The script should be located in the `scripts/` directory.

3. **Run the Inference Function**

   You can utilize the inference function within a Python script or interactively. Here's an example of how to use it:

   ```python
   # Navigate to the scripts directory
   cd scripts

   # Open a Python interpreter or create a new script
   python
   ```

   ```python
   from inference import inference

   # Specify the path to your new data CSV
   new_data_path = '../data/New_Data_Input.csv'

   # Make predictions
   predictions = inference(new_data_path)

   # Display the predictions
   print(predictions)

   # Optionally, save the predictions to a CSV file
   predictions.to_csv('../data/Predictions_Output.csv', index=False)
   ```

   **Alternatively**, you can create a separate Python script (e.g., `run_inference.py`) with the following content:

   ```python
   import sys
   from inference import inference

   def main():
       if len(sys.argv) != 2:
           print("Usage: python run_inference.py <path_to_new_data_csv>")
           sys.exit(1)

       csv_file_path = sys.argv[1]
       predictions = inference(csv_file_path)
       output_path = 'Predictions_Output.csv'
       predictions.to_csv(output_path, index=False)
       print(f"Predictions saved to {output_path}")

   if __name__ == "__main__":
       main()
   ```

   **Execute the Inference Script**

   ```bash
   python run_inference.py ../data/New_Data_Input.csv
   ```

4. **Review the Predictions**

   The predictions will be saved in `Predictions_Output.csv` containing two columns:

   - `'ID'`: Unique identifier from the input data
   - `'result'`: Predicted class label (e.g., 0 or 1)

---

## Model Evaluation

During the training phase, each model's performance is evaluated using various metrics:

- **Accuracy**: Measures the proportion of correct predictions.
- **ROC AUC**: Evaluates the model's ability to distinguish between classes.
- **Classification Report**: Provides precision, recall, F1-score, and support for each class.
- **Confusion Matrix**: Displays the number of true positives, true negatives, false positives, and false negatives.
- **ROC Curve**: Visual representation of the trade-off between true positive rate and false positive rate.

These metrics help in assessing the effectiveness of individual models and the ensemble.

---

## Saving and Loading Models

- **Saving Models**: The training pipeline automatically saves the trained models in the `models/` directory using `joblib`.
  
  ```python
  # Example from the training script
  joblib.dump(best_xgb, 'best_xgb_model.pkl')
  joblib.dump(best_lgb, 'best_lgb_model.pkl')
  joblib.dump(stacking_ensemble, 'stacking_ensemble_model.pkl')
  ```

- **Loading Models**: The inference function demonstrates how to load the saved Stacking Ensemble model.
  
  ```python
  import joblib

  # Load the saved stacking ensemble model
  stacking_ensemble = joblib.load('stacking_ensemble_model.pkl')
  ```

**Important**: Ensure that the environment where you load the models has the same library versions as those used during training to avoid compatibility issues.

---

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any inquiries or support, please contact [your.email@example.com](mailto:your.email@example.com).

---

# Appendix

## Inference Function Code

Below is the `inference.py` script used for making predictions on new data.

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

def inference(csv_file_path):
    """
    Function to make predictions using the trained stacking ensemble model.
    
    Args:
        csv_file_path (str): Path to the CSV file containing the data to predict on.
    
    Returns:
        output (pd.DataFrame): DataFrame containing 'ID' and 'result' columns.
    """
    # Load the saved stacking ensemble model
    stacking_ensemble = joblib.load('models/stacking_ensemble_model.pkl')

    # Define the columns to drop (as per your training code)
    columns_to_drop = [
        'Column17', 'Column3', 'Column6', 'Column4', 'Column8', 'Column14',
        'Column2', 'Column5', 'Column0', 'Column19', 'Column15', 'Column12',
        'Column20', 'Column11', 'Column10', 'Column9', 'Column13', 'Column16', 'Column21'
    ]

    # Read the new data
    data = pd.read_csv(csv_file_path)

    # Check for 'ID' column
    if 'ID' not in data.columns:
        raise ValueError("Input data must contain an 'ID' column.")

    # Keep the IDs for output
    IDs = data['ID']

    # Drop 'ID' column
    X_inference = data.drop(columns=['ID'])

    # Drop the specified columns
    columns_present_to_drop = [col for col in columns_to_drop if col in X_inference.columns]
    X_inference = X_inference.drop(columns=columns_present_to_drop)

    # Handle missing values by imputing with mean (using inference data)
    imputer = SimpleImputer(strategy='mean')
    X_inference_imputed = pd.DataFrame(imputer.fit_transform(X_inference), columns=X_inference.columns)

    # Ensure the feature columns are in the same order as during training
    features = X_inference_imputed.columns.tolist()
    X_inference_imputed = X_inference_imputed[features]

    # Make predictions
    y_pred = stacking_ensemble.predict(X_inference_imputed)

    # Output predictions with IDs
    output = pd.DataFrame({'ID': IDs, 'result': y_pred})

    return output
```

**Notes**:

- **Model Path**: Ensure that the path `'models/stacking_ensemble_model.pkl'` correctly points to the saved ensemble model. Adjust the path if necessary.
- **Imputation Strategy**: The inference function uses a new `SimpleImputer` fitted on the inference data. For consistency, it's recommended to use the same imputer fitted on the training data. Consider saving the trained imputer during the training phase and loading it during inference.
- **Feature Order**: The function assumes that the feature columns in the inference data are the same as those used during training. Ensure that the input data aligns with the training feature set.

---

# Acknowledgements

- [XGBoost](https://github.com/dmlc/xgboost)
- [LightGBM](https://github.com/microsoft/LightGBM)
- [Optuna](https://optuna.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Joblib](https://joblib.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)

---

# Disclaimer

This project is intended for educational purposes. Ensure that you comply with all applicable data privacy laws and regulations when using and deploying machine learning models.
