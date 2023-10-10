# Boston-Houses-Prices-prediction
This Python script is designed to predict house prices in Boston using the Boston Housing Dataset. 
It employs the XGBoost algorithm for regression, preprocessing with StandardScaler, and evaluation with Mean Absolute Error (MAE), R-squared (R^2) metrics, and has achieved an accuracy of 88%.

## Usage
1. Clone the repository or download the `boston_house_price_predictor.py` file to your local machine.
2. Ensure that you have the Boston Housing Dataset (boston.csv) in the same directory as the script. You can obtain this dataset from various sources online.
3. Open a terminal or command prompt and navigate to the directory containing the script and dataset.
4. Run the script using the following command: python boston_house_price_predictor.py
5. The script will load the dataset, perform data preprocessing, train an XGBoost regression model, and evaluate its performance using MAE and R^2 scores.
6. The script has achieved an accuracy of 88% in predicting house prices.

## Dataset Information
The dataset used in this script contains information about various features related to Boston neighborhoods and the median value of owner-occupied homes (MEDV).
- Features: The script uses several features like crime rates, average number of rooms, accessibility to highways, etc., to predict the MEDV.
- Target Variable: The target variable (what we want to predict) is the median value of owner-occupied homes (MEDV).

## Results
The script will print out the MAE and R^2 scores for both the training and testing datasets.
Additionally, it will display scatter plots comparing the actual prices to the predicted prices for both training and testing data, With an accuracy of 88%, this model provides reliable predictions for house prices in Boston.

