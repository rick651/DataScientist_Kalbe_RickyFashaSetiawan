import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

# Or, only disable specific warnings based on category
# Example: Disabling DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
df = pd.read_csv('C:/Users/hp/Downloads/PortoBuilder/Kalbe Internship/Final Project/Case Study Data Scientist/merged_data.csv')
df.head()
df.info()
df_totalquantity = df.groupby('Date')["Qty"].sum().reset_index()
df_totalquantity.info()
df_totalquantity.set_index('Date', inplace=True)
from pylab import rcParams
rcParams['figure.figsize'] = 20, 7
df_totalquantity.plot(marker='o')
plt.show()
# Assuming df_totalquantity is your DataFrame with 'Date' as the index
df_totalquantity.boxplot()
plt.title('Boxplot of Total Quantity')
plt.ylabel('Total Quantity')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")
adfuller_test(df_totalquantity['Qty'])
# Create ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df_totalquantity['Qty'], lags=30, ax=ax1)
plot_pacf(df_totalquantity['Qty'], lags=30, ax=ax2)

plt.tight_layout()
plt.show()
aic_scores = []
# Fit the ARIMA model
model = ARIMA(df_totalquantity['Qty'], order=(4,0,4))
model_fit = model.fit()
# Add AIC score to the list
aic_scores.append({'par': '(4,0,4)', 'aic': model_fit.aic})
aic_scores
from itertools import product

# Define ranges for p, d, and q
p = range(0, 5)  # 0 to 7
d = range(0, 3)  # 0 to 2
q = range(0, 5)  # 0 to 7

# Use the product function from itertools
# to create combinations of p, d, and q
pdq = list(product(p, d, q))
print(pdq)

# Splitting data into training and testing with ratio 8 : 2
data_train = df_totalquantity[:292]["Qty"]
data_test = df_totalquantity[292:]['Qty']

# Creating a list to store AIC scores
aic_scores = []

# Performing manual grid search to find optimal p, d, q
for param in pdq:
    # Fitting the ARIMA model
    model = ARIMA(data_train, order=param)
    model_fit = model.fit()
    # Adding AIC score to the list
    aic_scores.append({'par': param, 'aic': model_fit.aic})

# Finding the smallest AIC score
best_aic = min(aic_scores, key=lambda x: x['aic'])

print(best_aic)

# Creating an ARIMA model with the best p, d, and q from grid search
model = ARIMA(data_train, order=(best_aic['par']))
model_fit = model.fit()

# Making predictions for the next 73 days (testing data)
preds = model_fit.forecast(73)
preds.plot()
import pmdarima as pm

auto_arima = pm.auto_arima(data_train,stepwise=False, seasonal=False)
forecast = auto_arima.predict(n_periods=73)

auto_arima.summary()
# ploting
forecast.plot(label='auto arima')
preds.plot(label='grid search')

data_train.plot(label='train')
data_test.plot(label='test')
plt.legend()
# Calculate RMSE for training data
train_predictions = model_fit.predict(start=data_train.index[0], end=data_train.index[-1])
train_rmse = np.sqrt(mean_squared_error(data_train, train_predictions))

# Calculate RMSE for testing data
test_rmse = np.sqrt(mean_squared_error(data_test, preds))

print(f"RMSE for Training Data: {train_rmse:.2f}")
print(f"RMSE for Testing Data: {test_rmse:.2f}")
df_totalquantity.plot(figsize=(15, 6), alpha=0.5, marker="o")
preds.plot(linewidth=2, marker="o", legend=True)
from pandas.tseries.offsets import DateOffset

future_dates=[df_totalquantity.index[-1]+ DateOffset(days=x)for x in range(0,31)]
future_dates_df=pd.DataFrame(index=future_dates[1:],columns=df_totalquantity.columns)

future_df = pd.concat([df_totalquantity,future_dates_df])

model=ARIMA(df_totalquantity['Qty'], order=(2,1,3))
model_fit=model.fit()

future_df['forecast'] = model_fit.predict(start = 0, end = 395, dynamic = False)
future_df[['Qty', 'forecast']].plot(figsize=(12, 8))
future_df.tail(30).mean()
import pickle

# Creating an ARIMA model with the best p, d, and q from grid search
model = ARIMA(df_totalquantity['Qty'], order=best_aic['par'])
model_fit = model.fit()

# Save the model to a file using pickle
model_filename = 'arima_model.pkl'
with open('C:/Users/hp/Downloads/PortoBuilder/Kalbe Internship/Final Project/Case Study Data Scientist' + model_filename, 'wb') as model_file:
    pickle.dump(model_fit, model_file)

print("Model saved successfully!")
import pickle
# Load the ARIMA model from the file
model_filename = 'arima_model.pkl'
with open('C:/Users/hp/Downloads/PortoBuilder/Kalbe Internship/Final Project/Case Study Data Scientist' + model_filename, 'rb') as model_file:
    loaded_model_fit = pickle.load(model_file)

# Number of days for prediction
num_days = 30

# Forecast the next 30 days
forecast = loaded_model_fit.forecast(steps=num_days)

print("Forecasted quantities for the next", num_days, "days:")
print(forecast)

average_quantity_forecast = forecast.mean()
print("Average quantity forecast for the next", num_days, "days:", average_quantity_forecast)
forecast.plot()

import nbformat

def ipynb_to_py(input_notebook, output_script):
    with open(input_notebook, 'r', encoding='utf-8') as nb_file:
        notebook = nbformat.read(nb_file, as_version=4)
    
    code_cells = []
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            code_cells.append(cell.source)
    
    with open(output_script, 'w', encoding='utf-8') as py_file:
        py_file.write('\n'.join(code_cells))

# Replace 'input_notebook.ipynb' with the name of your input Jupyter Notebook file
# Replace 'output_script.py' with the desired name for the output Python script file
input_notebook = 'DataForecasting_Kalbe_Ricky.ipynb'
output_script = 'DataForecasting_Kalbe_Ricky.py'

ipynb_to_py(input_notebook, output_script)
