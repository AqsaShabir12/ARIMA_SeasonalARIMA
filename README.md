# ARIMA and Seasonal ARIMA Models for Time Series Forecasting

This project demonstrates how to model and forecast monthly champagne sales using ARIMA and Seasonal ARIMA (SARIMA) models. It includes data preprocessing, stationarity testing, differencing, ACF/PACF analysis, and model building using Python's `statsmodels` library.

---

## Dataset

- **File**: `sales_data.csv`
- **Description**: Monthly champagne sales (in millions) from Jan 1964 to Sep 1972.
- **Source**: Perrin Freres champagne sales data.

---

##  Steps and Methodology

### 1. **Data Cleaning and Preprocessing**
- Loaded the dataset using `pandas`.
- Renamed confusing column headers.
- Dropped rows with NaN values at the end of the dataset.
- Converted the 'Month' column to `datetime` format.
- Set 'Month' as the index to enable time series analysis.

### 2. **Initial Visualization**
- Plotted raw sales data over time.
- Observation: Clear seasonality and trend components → **non-stationary**.

### 3. **Stationarity Check**
- Used **Augmented Dickey-Fuller (ADF)** test via `adfuller` to check for unit root.
- **Result**: High p-value → fail to reject null → data is **non-stationary**.

### 4. **Differencing to Achieve Stationarity**
- Applied:
  - **First-order differencing**: `df['Sales'] - df['Sales'].shift(1)`
  - **Seasonal differencing (lag=12)**: `df['Sales'] - df['Sales'].shift(12)`
- Re-ran ADF test → significant p-value → **stationary** series confirmed.

### 5. **ACF and PACF Plots**
- Used to determine optimal `p` and `q` parameters.
- Visualized using `plot_acf` and `plot_pacf` from `statsmodels.graphics.tsaplots`.

### 6. **ARIMA Model Building**
- Fitted an **ARIMA(1,1,1)** model using:
  ```python
  from statsmodels.tsa.arima.model import ARIMA
  model = ARIMA(df['Sales'], order=(1, 1, 1))
  model_fit = model.fit()
- Inspected summary for coefficient significance and diagnostics.
### 6. **SARIMA Model Building**
- Added seasonal order (1,1,1,12) to capture yearly patterns.
  ```python
  from statsmodels.tsa.statespace.SARIMAX import SARIMAX
  model = SARIMAX(df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
  results = model.fit()
- Forecasted future sales using results.predict() and visualized predictions.

## Requirements
 ```pip install
 pandas numpy matplotlib seaborn statsmodels
