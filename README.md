# AI-ML Recruitment 2026

## 1. Candidate Details
- **Name:** Krishna Agarwal
- **Academic Year:** Second Year B.Tech
- **Track:** AI-ML Recruitment Task 2026
- **Task Option:** Option 1 — Air Quality Forecasting
- **GitHub Repository:** [AIML-Recruitment-2026-KrishnaAgarwal](https://github.com/krishnanew-byte/AIML-Recruitment-2026-KrishnaAgarwal.git)

---

## 2. Task Completed
- **Selected Task:** Option 1 — Air Quality Forecasting using the UCI Air Quality Dataset.
- **Deliverables:**
  - `Air_Quality_Forecasting.ipynb`: Complete, end-to-end executed Jupyter Notebook containing all 23 structured sections, beginner-friendly code, comprehensive markdown explanations, and real embedded output figures.
  - `README.md`: In-depth documentation covering all 17 required sections, methodological explanations, empirical findings, and instructions.
  - `results/`: Directory storing all high-resolution generated plots, model evaluation metrics (`metrics_summary.json`), and summary tables (`metrics_summary.csv`).

---

## 3. Problem Statement
Urban air pollution causes severe respiratory and cardiovascular health complications. Ground-level volatile organic compounds (VOCs) such as **Benzene ($C_6H_6$)** and combustion byproducts like **Carbon Monoxide ($CO$)** and **Nitrogen Oxides ($NO_x$)** fluctuate rapidly due to traffic cycles and changing atmospheric dispersion conditions.

The objective of this project is to develop a robust, explainable, and leakage-free Machine Learning regression system to forecast **future Benzene concentration ($C_6H_6$) 1 hour ahead ($t+1$)** using historical hourly readings from an array of chemical metal-oxide sensors, meteorological variables (Temperature, Relative Humidity, Absolute Humidity), temporal lags, rolling statistics, and diurnal calendar features.

---

## 4. Dataset
- **Name:** Air Quality Dataset
- **Source:** UCI Machine Learning Repository
- **Collection Setting:** Deployed in an Italian city at road level in a heavily polluted zone from **March 2004 to April 2005** (~13 months).
- **Size:** 9,358 hourly recorded observations across 15 attributes.

### Feature Description:
| Column | Description | Unit / Measurement Type |
| :--- | :--- | :--- |
| `Date` | Date of measurement | DD/MM/YYYY |
| `Time` | Time of measurement | HH.MM.SS |
| `CO(GT)` | True hourly averaged Carbon Monoxide | $mg/m^3$ (Reference Analyzer) |
| `PT08.S1(CO)` | Tin oxide sensor resistance targeted to CO | Sensor resistance ($\Omega$) |
| `NMHC(GT)` | True Non-Metanic HydroCarbons | $\mu g/m^3$ (Reference Analyzer) |
| `C6H6(GT)` | True Benzene concentration (**Target**) | $\mu g/m^3$ (Reference Analyzer) |
| `PT08.S2(NMHC)` | Titania sensor resistance targeted to NMHC | Sensor resistance ($\Omega$) |
| `NOx(GT)` | True Nitrogen Oxides concentration | $ppb$ (Reference Analyzer) |
| `PT08.S3(NOx)` | Tungsten oxide sensor targeted to $NO_x$ | Sensor resistance ($\Omega$) |
| `NO2(GT)` | True Nitrogen Dioxide concentration | $\mu g/m^3$ (Reference Analyzer) |
| `PT08.S4(NO2)` | Tungsten oxide sensor targeted to $NO_2$ | Sensor resistance ($\Omega$) |
| `PT08.S5(O3)` | Indium oxide sensor targeted to Ozone ($O_3$) | Sensor resistance ($\Omega$) |
| `T` | Ambient Temperature | $^\circ C$ |
| `RH` | Relative Humidity | $\%$ |
| `AH` | Absolute Humidity | $g/m^3$ |

### Dataset Idiosyncrasies & Challenges:
1. **Invalid Value Sentinel:** Sensor malfunctions and telemetry dropouts are coded as **`-200`**.
2. **European Formatting:** Semicolon (`;`) column separators, European decimal commas (e.g. `2,6` instead of `2.6`), and trailing empty columns (`Unnamed: 15`, `Unnamed: 16`).
3. **Severe Missingness in NMHC:** `NMHC(GT)` has 8,443 missing values out of 9,357 (**90.23% missing**).

---

## 5. Approach
The solution follows a disciplined, step-by-step engineering workflow:
```
Raw UCI CSV
    │
    ▼
1. Data Cleaning & Inspection (Drop trailing blanks, parse European decimals, map -200 -> NaN)
    │
    ▼
2. Domain Decision on Missing Data (Drop 90% missing NMHC, apply time interpolation for continuous sensors)
    │
    ▼
3. Datetime Engineering (Unified datetime index, chronological sorting)
    │
    ▼
4. Exploratory Data Analysis (Distributions, 13-month trends, correlation matrix, diurnal rush hours)
    │
    ▼
5. Feature Engineering (Lags t-1..t-24, rolling means 3h/6h/24h, rolling std 6h, hour, day, weekend, month)
    │
    ▼
6. Future Target Formulation (Target = C6H6 shifted to t+1; strictly past features predicting future)
    │
    ▼
7. Time-Order Chronological Split (80% Train: Mar 2004 - Jan 2005; 20% Test: Jan 2005 - Apr 2005)
    │
    ▼
8. Model Training & Benchmarking (Random Forest Regressor vs Linear Regression Baseline)
    │
    ▼
9. Out-of-Sample Evaluation & Error Analysis (MAE, MSE, RMSE, R², residual distributions, feature importances)
    │
    ▼
10. Actionable Findings & Interview Explanations
```

---

## 6. Data Preprocessing
1. **Parsing:** Loaded using `pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')`. Removed trailing blank rows and redundant `Unnamed` columns.
2. **Handling `-200` Missing Sentinel:**
   - Replaced all `-200` occurrences with `np.nan`.
   - Identified that `NMHC(GT)` was 90.23% missing. Rather than hallucinating values via artificial imputation, `NMHC(GT)` was dropped from modeling.
   - For other sensor columns (missing only 3.9% to 17.9%), continuous physical changes make **time-series linear interpolation (`df.interpolate(method='time')`)** appropriate. Edge boundary values were finalized with `bfill()` / `ffill()`.
3. **Temporal Processing:**
   - Combined `Date` and `Time` into a unified `datetime` format (`%d/%m/%Y %H.%M.%S`).
   - Indexed the dataframe by timestamp and verified chronological monotonicity (`df.index.is_monotonic_increasing == True`).
4. **Final Cleaned Dataset Shape:** 9,357 rows $\times$ 12 clean continuous numerical sensor features, 0 remaining NaNs.

---

## 7. Feature Engineering
To predict future air quality without lookahead bias, we engineered 15+ causal features based on atmospheric physics and human behavior:

| Feature Name | Category | Formula / Definition | Predictive & Physical Intuition |
| :--- | :--- | :--- | :--- |
| `c6h6_lag_1` | Target Lag | $C_6H_6(t-1)$ | Immediate memory; air quality has high physical inertia. |
| `c6h6_lag_2`, `lag_3` | Target Lag | $C_6H_6(t-2), C_6H_6(t-3)$ | Short-term momentum and recent trajectory of pollution. |
| `c6h6_lag_6`, `lag_12` | Target Lag | $C_6H_6(t-6), C_6H_6(t-12)$ | Half-day baseline concentration shift. |
| `c6h6_lag_24` | Target Lag | $C_6H_6(t-24)$ | 24-hour seasonal lag capturing yesterday's pollution at the exact same hour. |
| `s2_nmhc_lag_1` | Sensor Lag | $PT08.S2(t-1)$ | Titania sensor reading 1 hour prior. |
| `co_gt_lag_1` | Sensor Lag | $CO(GT)(t-1)$ | Previous hour's Carbon Monoxide combustion level. |
| `nox_gt_lag_1` | Sensor Lag | $NO_x(GT)(t-1)$ | Previous hour's vehicular exhaust marker. |
| `c6h6_rolling_mean_3` | Rolling Stat | Moving average ($w=3$ hrs) | Smooths out high-frequency sensor noise and captures short-term trend. |
| `c6h6_rolling_mean_6` | Rolling Stat | Moving average ($w=6$ hrs) | Medium-term concentration level. |
| `c6h6_rolling_mean_24`| Rolling Stat | Moving average ($w=24$ hrs)| Daily baseline level; eliminates diurnal volatility. |
| `c6h6_rolling_std_6` | Rolling Stat | Moving standard dev ($w=6$ hrs) | Measures atmospheric volatility and sudden concentration spikes. |
| `hour` | Temporal | Hour of day ($0 - 23$) | Informs model of morning (8-9 AM) and evening (7-8 PM) commuter rush hours. |
| `day_of_week` | Temporal | Day of week ($0 = \text{Mon}, 6 = \text{Sun}$) | Captures weekday industrial activity vs weekend reductions. |
| `is_weekend` | Temporal | Binary flag ($1$ if Sat/Sun, else $0$) | Distinct weekend traffic signature. |
| `month` | Temporal | Month ($1 - 12$) | Captures winter temperature inversions vs summer dispersion. |

---

## 8. Machine Learning Model
### Primary Model: Random Forest Regressor
- **Architecture:** Bagging ensemble of 100 decorrelated decision trees (`RandomForestRegressor`).
- **Hyperparameters:**
  - `n_estimators`: 100
  - `max_depth`: 12 (controls tree depth to prevent overfitting on leaf nodes)
  - `min_samples_split`: 6
  - `min_samples_leaf`: 3
  - `random_state`: 42 (ensures deterministic reproducibility)
  - `n_jobs`: -1 (parallel CPU utilization)
- **Why Random Forest was chosen:**
  1. **Non-Linear Dynamics:** Solid-state metal oxide chemical sensors respond non-linearly (logarithmic/power-law) to gas concentrations. Random Forests naturally model these complex non-linear response curves.
  2. **Collinearity Resilience:** Multiple chemical sensors respond concurrently to reducing combustion gases. Tree ensembles perform split-based feature selection without matrix singularity issues.
  3. **Bagging Variance Reduction:** Averaging 100 independently bootstrapped trees smooths individual noisy predictions.
  4. **Interpretability:** Provides transparent Gini/Variance feature importance metrics.

### Benchmark Model: Linear Regression
- Standard Ordinary Least Squares (OLS) trained on the identical feature matrix to objectively benchmark the non-linear advantage of the Random Forest.

---

## 9. Evaluation Metrics
We evaluated models using four standard regression metrics on both training and held-out test data:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   - Measures average magnitude of errors in original physical units ($\mu g/m^3$). Intuitive and robust to outliers.

2. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
   - Penalizes large deviations quadratically, highlighting extreme forecast failures.

3. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
   - Standard deviation of unexplained residuals expressed in $\mu g/m^3$.

4. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   - Quantifies the proportion of variance in Benzene concentrations explained by the model ($1.0$ is perfect; $0.0$ equals predicting the simple average).

---

## 10. Results

### Performance Summary Table:
| Model | Dataset Partition | MAE ($\mu g/m^3$) | MSE | RMSE ($\mu g/m^3$) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **Test (Unseen Future)** | **1.8074** | **7.6714** | **2.7697** | **0.8161** |
| Random Forest Regressor | Train (Historical) | 1.1560 | 2.9655 | 1.7221 | 0.9493 |
| Linear Regression Baseline | Test (Unseen Future) | 2.1155 | 9.5308 | 3.0872 | 0.7715 |

> **Key Performance Takeaway:** The Random Forest achieved an **$R^2$ of 0.8161** on completely unseen future data, explaining **81.6% of the variance** in 1-hour ahead Benzene concentrations and outperforming the linear baseline by a significant margin.

### Generated Results Visualizations (Stored in `results/`):
- `results/eda_pollutant_distributions.png`: Right-skewed distribution plots of key pollutants.
- `results/eda_time_series_overview.png`: 13-month multi-panel overview showing winter thermal inversions.
- `results/eda_correlation_heatmap.png`: Sensor vs. pollutant correlation matrix showing strong titania sensor coupling (+0.98).
- `results/eda_hourly_daily_patterns.png`: Bimodal rush-hour peaks (8-9 AM, 7-8 PM) and weekly trends.
- `results/actual_vs_predicted.png`: Full out-of-sample timeline comparison over the 2.5-month unseen test period.
- `results/actual_vs_predicted_zoom.png`: High-resolution 7-day (168-hour) zoom demonstrating close diurnal peak tracking.
- `results/feature_importance.png`: Feature importance breakdown showing dominance of contemporaneous sensor readings, lags, and rolling averages.
- `results/residuals_analysis.png`: Residual distribution centered at zero and actual vs. predicted scatter with 1:1 ideal line.

---

## 11. Key Findings
1. **Bimodal Diurnal Traffic Cycles:** Ground-level Benzene and Carbon Monoxide exhibit two daily rush-hour peaks: morning (8:00 AM – 9:00 AM) and evening (7:00 PM – 8:00 PM), matching urban commuter traffic.
2. **The "Weekend Effect":** Benzene concentrations drop by ~20% on Saturdays and ~35% on Sundays compared to mid-week workdays due to commercial transport reductions.
3. **Exceptional Sensor Coupling:** Chemical sensor `PT08.S2(NMHC)` demonstrates a **+0.98 Pearson correlation** with Benzene reference readings, validating the capability of low-cost metal-oxide sensors to track VOCs.
4. **Winter Inversion Accumulation:** Systematically higher pollution spikes occur from November through February because thermal inversions trap vehicular exhaust near ground level.
5. **Short-Term Temporal Persistence:** Feature importance analysis revealed that current sensor readings and immediate 1-to-3 hour lags account for over 60% of predictive power.
6. **Sensor Quality Screening is Critical:** Over 90% of `NMHC(GT)` was missing (-200 sentinel). Removing it prevented severe synthetic imputation bias.
7. **Tree Ensembles Outperform Linear Baselines:** Random Forest improved out-of-sample $R^2$ from 0.7715 (linear) to **0.8161**, successfully capturing non-linear sensor response dynamics.

---

## 12. Limitations
1. **Underestimation of Extreme Pollution Spikes:** Decision tree ensembles predict within the bounding range of historical tree leaf averages. During extreme pollution episodes ($>25\ \mu g/m^3$), the model underpredicts the true peak (MAE increases from 1.26 to 4.92 $\mu g/m^3$).
2. **Absence of Critical Meteorological Covariates:** The dataset lacks wind speed, wind direction, precipitation, and boundary layer mixing height, which are primary drivers of atmospheric pollutant advection and dispersion.
3. **Single-Step Forecast Horizon:** The system forecasts 1 hour into the future ($t+1$). Expanding to multi-hour horizons (e.g. 12 to 24 hours ahead) requires recursive forecasting or sequence modeling, which introduces error compounding over time.

---

## 13. Future Improvements
1. **External Meteorological Data Integration:** Ingest live wind speed, wind direction, and atmospheric pressure gradients from municipal weather stations.
2. **Direct Multi-Horizon Forecasting:** Implement multi-output models to forecast $[t+1, t+3, t+6, t+12, t+24]$ hours simultaneously for day-ahead municipal alerts.
3. **Quantile Regression for Health Advisories:** Train Quantile Random Forests to predict $90^{\text{th}}$ and $95^{\text{th}}$ percentile upper bounds, ensuring alerts are issued even during unprecedented spikes.

---

## 14. Key Learnings
- **Time-Series Integrity:** Learned why standard random shuffling destroys time-series validity and causes catastrophic lookahead data leakage.
- **Sentinel Missing Values:** Learned to check beyond standard `NaN` values and recognize domain-specific sentinels like `-200` in raw sensor streams.
- **Physical Feature Engineering:** Gained hands-on experience constructing lagged features, rolling averages, and rolling volatility that mirror physical atmospheric inertia.
- **Model Evaluation Realism:** Understood the difference between in-sample fit ($R^2 = 0.95$) and true out-of-sample generalization ($R^2 = 0.82$) on future unseen time horizons.

---

## 15. Challenges & Solutions
1. **European Delimiters and Decimal Formats:**
   - *Challenge:* The raw CSV used semicolons (`;`) and decimal commas (`,`), causing numerical columns to load as strings.
   - *Solution:* Parsed using `pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')` and stripped trailing empty `Unnamed` columns.
2. **The 90% Missing NMHC Sensor:**
   - *Challenge:* Column `NMHC(GT)` had 8,443 missing values (-200).
   - *Solution:* Recognized that imputing 90% of a column would inject synthetic bias; dropped the column while retaining other sensor channels.
3. **Zero-Leakage Target Construction:**
   - *Challenge:* Formulating a true forecasting target without leaking future information into input features.
   - *Solution:* Shifted target $y$ by $-1$ ($t+1$) while strictly computing input features $X$ using historical data up to time $t$.

---

## 16. Project Structure
```
AIML-Recruitment-2026-KrishnaAgarwal/
│
├── README.md                          # Comprehensive project documentation
├── Air_Quality_Forecasting.ipynb       # Fully executed end-to-end Jupyter Notebook
├── AirQualityUCI.csv                  # Cleaned / local copy of the UCI dataset
│
└── results/                           # Generated output plots and evaluation summaries
    ├── actual_vs_predicted.png        # Full test timeline actual vs predicted plot
    ├── actual_vs_predicted_zoom.png   # 7-day (168-hour) zoomed-in diurnal tracking
    ├── eda_correlation_heatmap.png    # Sensor vs pollutant correlation matrix
    ├── eda_hourly_daily_patterns.png  # Diurnal rush hours and weekly pattern profiles
    ├── eda_pollutant_distributions.png# Histograms and KDE distribution curves
    ├── eda_time_series_overview.png   # 13-month multi-panel time-series overview
    ├── feature_importance.png         # Top 12 Random Forest feature importances
    ├── residuals_analysis.png         # Residual distribution and actual vs predicted scatter
    ├── metrics_summary.json           # Evaluation metrics in JSON format
    └── metrics_summary.csv            # Evaluation metrics in CSV format
```

---

## 17. How to Run

### Step 1: Clone the Repository
```bash
git clone https://github.com/krishnanew-byte/AIML-Recruitment-2026-KrishnaAgarwal.git
cd AIML-Recruitment-2026-KrishnaAgarwal
```

### Step 2: Install Dependencies
Ensure Python 3.9+ is installed, then run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nbformat nbclient ipykernel
```

### Step 3: Run the Jupyter Notebook
Open the notebook in Jupyter Notebook, JupyterLab, or VS Code:
```bash
jupyter notebook Air_Quality_Forecasting.ipynb
```
Select **Kernel $\rightarrow$ Restart & Run All**. The entire notebook will execute sequentially from top to bottom without errors in under 30 seconds.

---

### Candidate Sign-off:
**Krishna Agarwal**  
Second Year B.Tech Student  
Coding Ninjas 10X AI-ML Recruitment 2026
