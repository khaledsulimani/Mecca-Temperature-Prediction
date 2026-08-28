# 🌡️ Mecca Temperature Prediction & Machine Learning Analyzer

A machine learning project that analyzes historical weather data from **Mecca, Saudi Arabia**, and compares regression algorithms based on prediction accuracy, training speed, and memory efficiency.

The project implements a complete machine learning pipeline using real meteorological data from **Meteostat**, including data acquisition, preprocessing, model training, evaluation, and computational performance analysis.

![Python](https://img.shields.io/badge/Python-Machine%20Learning-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple.svg)
![Meteostat](https://img.shields.io/badge/Meteostat-Weather%20Data-lightblue.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

---

## 🌟 Features

### 🤖 **Dual Machine Learning Model Support**
- **Linear Regression**: Lightweight regression model used as an efficient baseline
- **Random Forest Regressor**: Ensemble learning model capable of capturing nonlinear relationships
- **Model Comparison**: Side-by-side evaluation of predictive and computational performance

### 🌦️ **Real Meteorological Data**
- Uses historical weather observations from **Meteostat**
- Retrieves data from weather station **41030**
- Covers approximately five years of historical observations
- Automatically caches downloaded weather data locally

### 🧹 **Machine Learning Preprocessing Pipeline**
- Missing target value removal
- Median-based missing value imputation
- Automatic feature availability detection
- 80/20 train-test split
- Feature standardization using `StandardScaler`

### 📊 **Advanced Performance Analysis**
- Mean Absolute Error (**MAE**)
- Coefficient of Determination (**R² Score**)
- Training execution time
- Memory usage profiling
- Automatic model comparison table

---

# 🚀 Quick Start

## Installation

```bash
# Clone the repository
git clone https://github.com/khaledsulimani/Mecca-Temperature-Prediction.git

# Navigate to the project directory
cd Mecca-Temperature-Prediction

# Create a virtual environment
python -m venv venv

# Activate the environment on Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python mecca_temperature_prediction.py
```

---

## 📦 Dependencies

- **Python 3.x**
- **Pandas** - Data loading and manipulation
- **NumPy** - Numerical processing
- **Meteostat** - Historical meteorological data
- **Scikit-learn** - Machine learning and preprocessing

Current project dependencies are defined in:

```text
requirements.txt
```

---

# 🌍 Weather Dataset

The project retrieves historical weather observations from the **Meteostat** data service.

The implementation uses weather station:

```text
41030
```

for Mecca, Saudi Arabia.

The dataset contains approximately five years of historical weather observations between **2021 and 2026**.

## 📋 Weather Features

Depending on availability from Meteostat, the pipeline can use:

| Feature | Description |
|---|---|
| `tmin` | Minimum daily temperature |
| `tmax` | Maximum daily temperature |
| `rhum` | Relative humidity |
| `prcp` | Daily precipitation |
| `wspd` | Average wind speed |
| `pres` | Atmospheric pressure |
| `cldc` | Cloud cover |

### 🎯 Target Variable

```text
temp
```

represents the average daily temperature used as the regression target.

---

# 🔄 Hybrid Data Loading System

The project includes a hybrid data-loading mechanism designed to reduce unnecessary network requests.

### 1. **Local Dataset Detection**

The application first checks for:

```text
mecca_weather_data.csv
```

### 2. **Local Loading**

If the dataset exists, the application loads the weather observations directly from the local CSV file.

### 3. **Automatic Data Retrieval**

If the dataset is unavailable, the application retrieves the required historical observations from Meteostat.

### 4. **Automatic Caching**

The retrieved dataset is automatically saved locally for future executions.

This approach provides:

- Faster repeated execution
- Reduced network dependency
- Reproducible experiments
- Offline dataset availability

---

# 🧹 Data Preprocessing Pipeline

Real-world meteorological datasets can contain incomplete or missing observations.

The project applies several preprocessing techniques before model training.

## 1. **Feature Selection**

The system automatically checks which expected weather variables are available:

```python
FEATURES = [
    "tmin",
    "tmax",
    "rhum",
    "prcp",
    "wspd",
    "pres",
    "cldc"
]
```

Unavailable columns are automatically ignored.

---

## 2. **Missing Target Removal**

Rows where the target temperature is unavailable are removed before training.

```python
df = df.dropna(subset=[TARGET])
```

---

## 3. **Train-Test Split**

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

using a fixed random state for reproducibility.

---

## 4. **Missing Value Imputation**

Missing feature values are handled using:

```python
SimpleImputer(strategy="median")
```

Median imputation reduces the influence of extreme values compared with mean-based replacement.

---

## 5. **Feature Scaling**

The input features are standardized using:

```python
StandardScaler()
```

This transforms the features to approximately:

```text
Mean = 0
Standard Deviation = 1
```

---

# 🧠 Machine Learning Models

## 📈 Linear Regression

Linear Regression is used as the baseline machine learning model.

It estimates a linear relationship between meteorological features and the target temperature.

### Advantages

- Extremely fast training
- Low memory consumption
- Simple implementation
- Highly interpretable
- Effective for strongly correlated variables

---

## 🌲 Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

The model configuration used in this project is:

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
```

### Advantages

- Captures nonlinear relationships
- Handles complex interactions between features
- Robust ensemble architecture
- Less dependent on linear assumptions

---

# 🧠 Model Comparison

| Model | Strengths | Best For |
|---|---|---|
| **Linear Regression** | Fast, lightweight, interpretable | Efficient baseline modeling |
| **Random Forest** | Nonlinear, robust, ensemble-based | Complex feature relationships |

The project compares both models not only based on predictive performance, but also on their **computational efficiency**.

---

# 📊 Evaluation Metrics

## 🎯 Mean Absolute Error — MAE

MAE measures the average absolute difference between predicted and actual temperatures.

```text
Lower MAE = Better prediction accuracy
```

---

## 📐 R² Score

The R² score measures how much of the variation in the target variable is explained by the model.

```text
R² closer to 1.0 = Stronger model fit
```

---

## ⏱️ Training Time

Model training duration is measured using Python's high-resolution:

```python
time.perf_counter()
```

This allows direct comparison of computational speed.

---

## 💾 Memory Usage

Memory consumption during model training is measured using:

```python
tracemalloc
```

This makes it possible to evaluate models from both a **machine learning** and **software engineering** perspective.

---

# 📊 Performance Benchmarks

The project produced the following evaluation results during the documented experiment:

| Model | MAE | R² Score | Training Time | Memory Usage |
|---|---:|---:|---:|---:|
| **Linear Regression** | **0.2641 °C** | **0.9949** | **0.0084 s** | **0.0140 MB** |
| **Random Forest Regressor** | **0.2510 °C** | **0.9944** | **0.3792 s** | **0.2344 MB** |

---

# 🔬 Performance Analysis

Both machine learning models achieved strong results during the experiment.

### 🎯 Prediction Accuracy

Random Forest achieved the lower MAE:

```text
Random Forest: 0.2510 °C
Linear Regression: 0.2641 °C
```

The difference in absolute prediction error was relatively small.

---

### 📐 Model Fit

Linear Regression achieved:

```text
R² = 0.9949
```

while Random Forest achieved:

```text
R² = 0.9944
```

Both models therefore produced very high R² scores in the tested setup.

---

### ⚡ Training Performance

Linear Regression required:

```text
0.0084 seconds
```

while Random Forest required:

```text
0.3792 seconds
```

This demonstrates the additional computational cost associated with the ensemble model.

---

### 💾 Memory Efficiency

Linear Regression consumed approximately:

```text
0.0140 MB
```

compared with:

```text
0.2344 MB
```

for Random Forest during the documented experiment.

---

# ⚖️ Engineering Trade-Off

One of the important findings of this project is that higher model complexity does not automatically guarantee significantly better performance.

Random Forest achieved a slightly lower MAE.

However, Linear Regression provided:

- Comparable predictive performance
- Faster training
- Lower memory usage
- Lower computational complexity

This demonstrates the importance of selecting machine learning models based on the requirements of the application rather than model complexity alone.

---

# 🔧 Machine Learning Pipeline

```text
Historical Weather Data
          │
          ▼
      Meteostat
          │
          ▼
 Local CSV Cache
          │
          ▼
   Feature Selection
          │
          ▼
 Missing Value Handling
          │
          ▼
   Train-Test Split
          │
          ▼
   Feature Scaling
          │
          ▼
   ┌───────────────────────┐
   │                       │
   ▼                       ▼
Linear Regression    Random Forest
   │                       │
   └───────────┬───────────┘
               ▼
          Prediction
               │
               ▼
     Performance Evaluation
               │
               ▼
     MAE / R² / Time / RAM
```

---

# 📁 Project Structure

```text
Mecca-Temperature-Prediction/
│
├── mecca_temperature_prediction.py
│   └── Main machine learning pipeline
│
├── mecca_weather_data.csv
│   └── Historical weather dataset cache
│
├── requirements.txt
│   └── Python project dependencies
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Git ignored files
│
└── report/
    └── Mecca_Temperature_Prediction_Report.pdf
```

---

# 🎯 Code Example

## Training the Models

```python
models = {
    "Linear Regression":
        LinearRegression(),

    "Random Forest Regressor":
        RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),
}
```

---

## Evaluating the Models

```python
mae = mean_absolute_error(
    y_test,
    fitted.predict(X_test)
)

r2 = r2_score(
    y_test,
    fitted.predict(X_test)
)
```

---

# 🖥️ Example Output

The application automatically prints model evaluation results in a comparison table.

```text
===========================================================================
Model                         MAE (C)   R2 Score   Time (s)   Mem (MB)
---------------------------------------------------------------------------
Linear Regression             ...
Random Forest Regressor       ...
===========================================================================
```

Exact runtime results may vary depending on:

- Hardware
- Operating system
- Python version
- Library versions
- Dataset state

---

# 🧪 Reproducibility

The project uses fixed random states where applicable:

```python
random_state=42
```

This improves reproducibility between experiments.

The preprocessing pipeline also fits imputation and scaling operations only on the training dataset before transforming the testing dataset.

---

# 🔧 Configuration

Several project parameters can be modified directly from the Python source.

## Random Forest Configuration

```python
n_estimators=100
max_depth=12
random_state=42
n_jobs=-1
```

## Dataset Period

The historical data period can be modified using:

```python
start_date = datetime(...)
end_date = datetime(...)
```

## Feature Selection

Additional Meteostat variables can be added to:

```python
FEATURES = [...]
```

---

# 🐛 Current Limitations & Future Improvements

## Current Limitations

- Current evaluation uses a standard train-test split rather than a chronological time-series split
- Prediction performance depends on the available meteorological features
- The project currently focuses on regression comparison rather than deployed real-time forecasting
- Model performance has been evaluated on a single geographical area
- No graphical dashboard is currently implemented

---

## 🚀 Planned Improvements

- [ ] Implement chronological time-series validation
- [ ] Add true future-horizon temperature forecasting
- [ ] Implement LSTM neural networks
- [ ] Add lagged temperature and weather features
- [ ] Add rolling-window statistical features
- [ ] Integrate real-time weather observations
- [ ] Integrate local ESP32 / IoT weather sensors
- [ ] Add graphical data visualization
- [ ] Build an interactive analytics dashboard
- [ ] Implement automated model retraining
- [ ] Export trained models for deployment
- [ ] Develop a REST API for model inference
- [ ] Compare additional regression algorithms
- [ ] Expand the system to additional Saudi cities

---

# 🔮 Future Architecture

A future version of the system could integrate local IoT weather stations with the machine learning pipeline:

```text
ESP32 Weather Sensors
        │
        ▼
Environmental Measurements
        │
        ▼
   Data Collection API
        │
        ▼
Historical + Real-Time Data
        │
        ▼
 Machine Learning Models
        │
        ▼
 Temperature Forecast
        │
        ▼
 Dashboard / REST API
```

This architecture could allow the model to incorporate highly localized environmental observations from different areas of Mecca.

---

# 📄 Project Report

A detailed university report is included with the project.

The report covers:

- Project motivation
- Machine learning methodology
- Data acquisition
- Data preprocessing
- Linear Regression
- Random Forest
- Model evaluation
- Computational performance
- Results and analysis
- Future improvements

### 📘 Full Report

[View Mecca Temperature Prediction Project Report](report/Mecca_Temperature_Prediction_Report.pdf)

---

# 📚 References & Technologies

## 🤖 Machine Learning

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
  Official documentation for machine learning algorithms, preprocessing, metrics, and model evaluation.

- [LinearRegression – Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
  Documentation for the Linear Regression implementation.

- [RandomForestRegressor – Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
  Documentation for the Random Forest regression implementation.

---

## 🌦️ Meteorological Data

- [Meteostat](https://meteostat.net/)
  Historical weather and climate data platform.

- [Meteostat Python Library](https://dev.meteostat.net/python/)
  Python interface used for retrieving historical weather observations.

---

## 📊 Data Processing

- [Pandas Documentation](https://pandas.pydata.org/docs/)
  Python data manipulation and analysis library.

- [NumPy Documentation](https://numpy.org/doc/)
  Numerical computing library used throughout the project.

---

## 🧹 Machine Learning Preprocessing

- [SimpleImputer – Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
  Missing value imputation functionality.

- [StandardScaler – Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
  Feature standardization functionality.

---

## 📏 Model Evaluation

- [Mean Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
  Regression metric used to calculate average absolute prediction error.

- [R² Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
  Regression metric used to evaluate model fit.

---

# 🧑‍💻 Author

- **Khaled Mahmoud Sulaimani** – [@khaledsulimani](https://github.com/khaledsulimani)

Computer Science  
Umm Al-Qura University

---

⭐ **If you find this project helpful, please give it a star!** ⭐
