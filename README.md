# IMDb Movies India - Data Preprocessing and Modeling

This repository contains a Jupyter Notebook focused on exploring and modeling a dataset of Indian movies, sourced from IMDb. 
The project involves data cleaning, feature engineering, and predictive modeling using machine learning techniques.

## Files in this Repository
- **IMDb__Movies__India_Model.ipynb**: The main Jupyter Notebook containing the code for data preprocessing and model development.
- **IMDb Movies India.csv**: Dataset used in the project (please ensure to place it in the same directory if running the notebook).

## Dataset Source
The dataset used in this project is available on Kaggle:  
[IMDb India Movies Dataset](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)

## Workflow Overview
1. **Library Imports & Data Loading**:
   - Essential libraries such as `pandas`, `numpy`, and `scikit-learn` are imported.
   - The dataset is loaded, and initial exploratory steps include checking for missing values and understanding column structures.

2. **Data Cleaning**:
   - Missing values are handled by dropping or imputing based on column types.
   - Non-numeric fields like `Duration` and `Year` are cleaned and converted to numeric formats.
   - Genres are expanded into binary columns for better analysis.

3. **Feature Engineering**:
   - Categorical variables such as `Director` and `Actors` are encoded using label encoding and one-hot encoding as required.

4. **Modeling**:
   - A Random Forest Regressor is used to predict target variables (e.g., `Rating`).
   - The model is evaluated using metrics such as Mean Squared Error (MSE) and R-squared.

## How to Run the Notebook
1. Ensure Python 3.7+ is installed with the following libraries:
   - pandas
   - numpy
   - scikit-learn

2. Download the dataset file from Kaggle ([link here](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)) and place it in the working directory.

3. Open the Jupyter Notebook and run all cells sequentially.

## Results
- The notebook demonstrates the end-to-end pipeline of handling a real-world dataset, from preprocessing to model evaluation.

## Future Work
- Fine-tuning hyperparameters using GridSearch or RandomizedSearch for improved model performance.
- Exploring additional models for comparison, such as XGBoost or neural networks.
