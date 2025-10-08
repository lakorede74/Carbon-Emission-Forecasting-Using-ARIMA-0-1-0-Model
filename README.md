# 🌍 Carbon Emissions Forecasting (ARIMA Time Series Analysis)
This project applies the ARIMA (Autoregressive Integrated Moving Average) model to historical carbon emission data to forecast future trends and quantify prediction accuracy. The final model is used to project emissions over a 6-year timeframe, providing crucial insights for environmental planning and policy-making.

## Dataset

**Source:** [Energy Institute](https://www.energyinst.org)
- **Description:** 2025 Energy Institute Statistical Review of World Energy
- **File name:** `EI-Stats-Review-ALL-data.xlsx
`
 
## 🔬 Methodology Overview
The analysis follows a standard time series workflow:

## Data Preparation: 

- The raw dataset comparises of numerous information raging from energy production, consunptions, emissions etc 

- Historical Carbon emissions data was extracted for  year 1994 - 2024 from the large dataset which was in Million Tons for standardized reporting

-  The data was then transformed to logarithmic values to improve its linearity.

## Code

```

import pandas as pd
import statsmodels as sm
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error


file = 'EI-Stats-Review-ALL-data.xlsx'

Enfile = pd.ExcelFile(file)
print(Enfile.sheet_names)


def extract_data(sheet, Country, column_name):
    df = pd.read_excel(Enfile, sheet_name=sheet, skiprows=2)
    df.rename(columns={df.columns[0]: "Country"}, inplace=True)
    df = df.groupby("Country").sum(numeric_only=True).reset_index()
    df = df.set_index("Country")
    df.replace({"_":0}, inplace=True)
    df = df[df.columns[:-3]]
    df = pd.DataFrame(df.loc[Country])
    df.rename(columns={Country: column_name}, inplace=True)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    return df

Carborn_df = extract_data("CO2e Emissions ", "Total World", "Carborn")

#CHOOSING MY SAMPLE SIZE (FROM 1994 TO 2024)
Carborn_df = Carborn_df.reset_index()
Carborn_df.rename(columns={Carborn_df.columns[0]: "Year"}, inplace= True)
Carborn_df = Carborn_df.query("Year >=1994  and Year <= 2024")
Carborn_df = Carborn_df[["Year", "Carborn"]].set_index("Year")


```




## Stationarity Testing: 

The Augmented Dickey-Fuller (ADF) test confirmed that the log-transformed series required first-order differencing to achieve stationarity.

Model Selection: Based on low AIC/BIC scores and residual analysis, the optimal model was determined to be ARIMA(0,1,0) with drift (trend='t').

p=0 (No Autoregressive component)

d=1 (Integrated component/First difference, handled internally by the model)

q=0 (No Moving Average component)


## The Code 
---
      CO2_series = Carborn_df.squeeze()
      
      if CO2_series.min() <= 0:
          CO2_series += abs(CO2_series.min()) +1
          
          print("WARNING: Data shifted to positivity to ensure log transformation")
      
      CO2_log = np.log(CO2_series)
      
      result = adfuller(CO2_log)
      
      
      print('ADF %f:' %result[0],
             'P-value %f:' %result[1])
             
      # splitting data for Testing and Training
      log_train = CO2_log[:-3]
      log_test = CO2_log[-3:]

      #Ploting to check for autocorrelation and partial autocorrelation
      fig, axes = plt.subplots(1, 2, figsize=(16,4))
      plot_acf(log_train, ax=axes[0], lags= 7)
      plot_pacf(log_train, ax=axes[1], lags= 7)
      plt.show()

---

## ACF & PACF
![](acfAndPacf.png)



## Training/Testing: 

I splited the data for training and testing, 27 data points for trainoing and the last 3 data points was reserved for testing the accuracy of my model's prediction as shown in the code blocks above




## Forecasting: 

The model was trained on the full historical dataset and used to forecast emission levels and their confidence intervals(95%).

---
    forecast_steps = 6
    
    log_train_forecast = ARIMA(log_train, order=(0, 1, 0), trend='t').fit()
    
    log_final_forecast = log_train_forecast.get_forecast(steps=forecast_steps)
    
    final_forecast = np.exp(log_final_forecast.predicted_mean)
    confidence_interval_forecast = np.exp(log_final_forecast.conf_int(alpha=0.05))

---

## 📊 Key Results

### forcast Result
The prediction for test (2021 - 2024) and for  2025 - 2027

### Prediction

![](https://github.com/lakorede74/Carbon-Emission-Forecasting-Using-ARIMA-0-1-0-Model/blob/9e05fb6c2b9d061d4fbcdc163e26c532e3feea74/PREDICTION.png)


## The PLot 
The plot of Carbon emission trend from 1984 - 2024 (green), predicted emissions from 2021 - 2027 (red)
shows how close the predictions are to the actual values of the test period

---
      plt.plot(CO2_series.index, CO2_series.values, label="Carbon emission(million-tone)", color="green")
      
      plt.plot(PREDICTION.index, PREDICTION['Predicted emission'], label="predicted emission(million-tone)", color="red", linestyle='--')
       
      plt.fill_between(PREDICTION.index, PREDICTION["Lower_ci (95%)"], PREDICTION['Upper_ci (95%)'], label="confidence interval (million-tone)", color="grey")
      
      plt.title(f'Carbon emission forecast with ARIMA(0, 1, 0) Model')
      plt.ylabel('Emission(million-tone)')
      plt.xlabel('Year')
      plt.grid(True, alpha= 0.5)
      plt.legend()
      plt.show()
---
## Forecast plot
![FORECAST PLOT](FINALFORECASTPLOT.png)

## Validation (Mean Error Metrics)
The Mean Absolute Error (MAE), Root Mean Square and the Mean Absolute Percnetage were calculated to measure how far off are the predicted values to the actual values.

---
     test_forecast_steps = 3
     forecast_log_test = log_train_forecast.get_forecast(steps=test_forecast_steps)
     
     forecast_test_value = np.exp(forecast_log_test.predicted_mean)
     
     actual_test_value = np.exp(log_test)
     
     
     
     
     MAE_CO2 = mean_absolute_error(actual_test_value, forecast_test_value)
     
     RMSE_CO2 = np.sqrt(mean_squared_error(actual_test_value, forecast_test_value))
     
     #MEAN ABSOLUTE PERCENTAGE
     forecast_test_value.index = actual_test_value.index
     
     MAPE_CO2 = np.mean(np.abs((actual_test_value - forecast_test_value)/actual_test_value)) * 100
     
     
     Mean_errors_Metrics = pd.DataFrame({
         'MAE': [MAE_CO2],
         'RMSE': [RMSE_CO2],
         'MAPE': [MAPE_CO2]
         },index=['Error metrics'])
---


## Test forecast values VS actual values
![](TEST_FORECASTandACTUAL.png)

Mean Result shows that the predicted values did not really deviate from it's actual values, which means forecast are reliable 
![ERROR METRICS](ERROR_METRICS.png)


---
# Conclusion/Recommendation
Year to year continuous positive growth rate of Carbon emissions indicates:

- Poor adoption of clean and renewable energy policy
- Economic  expansion
- Heavy reliance on fossil fuels

The constraints for the poor transitioning to clean and renewable energy should be investigated and subtly address to avoid adverse climatic impacts of increasing growth of carbon emission.


## 🛠️ Tools and Libraries

| Library | Purpose |
|----------|----------|
| `pandas` | Data manipulation and cleaning |
| `numpy` | Numerical computations |
| `matplotlib` | Visualization |
| `statsmodels` | ARIMA model implementation |

---

