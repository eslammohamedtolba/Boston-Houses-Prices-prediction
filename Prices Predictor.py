# import required modules
import numpy as np, pandas as pd , matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# uploading house price dataset
House_price_data = pd.read_csv("boston.csv")
# find number of rows and columns
House_price_data.shape
# check if there is any null values(missing values)
House_price_data.isnull().sum()
# show the data
House_price_data.head()
# descripe the data by some statistical info
House_price_data.describe()
# find correlation between various features in the dataset and constructing the heatmap
correlation = House_price_data.corr()
plt.figure(figsize=(10,10))
#              dataset   makecbar    shape   floatNums feature_names/valuesNum size        color
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap = 'Blues')



# split house price data into input and output(label) data
X = House_price_data.drop(columns='MEDV',axis=1)
Y = House_price_data['MEDV']
# scalling the data input
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)
# split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(standardized_data,Y,train_size=0.7,random_state=2)




# create model and train it
XGBRegressorModel = XGBRegressor()
XGBRegressorModel.fit(x_train,y_train)
# finding the predicted values for train data
predicted_train_prices = XGBRegressorModel.predict(x_train)
# finding the predicted values for test data
predicted_test_prices = XGBRegressorModel.predict(x_test)

# measure the value error for training and testing data by mean_absolute error
error_train = mean_absolute_error(predicted_train_prices,y_train)
error_test = mean_absolute_error(predicted_test_prices,y_test)
print(error_train,error_test)
# measure the value error for training and testing data by R squared error
error_train = r2_score(predicted_train_prices,y_train)
error_test = r2_score(predicted_test_prices,y_test)
print(error_train,error_test)



# visualized the output and predicted prices
plt.title("actual prices vs predicted prices in train")
plt.xlabel("actual prices")
plt.ylabel("predicted prices")
plt.scatter(y_train,predicted_train_prices,color='blue',marker='X')
plt.show()
plt.title("actual prices vs predicted prices in test")
plt.xlabel("actual prices")
plt.ylabel("predicted prices")
plt.scatter(y_test,predicted_test_prices,color='red',marker='*')
plt.show()

