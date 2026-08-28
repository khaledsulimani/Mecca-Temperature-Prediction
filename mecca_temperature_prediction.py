# Mecca Temperature Prediction — University ML Project
# Compares LinearRegression vs RandomForestRegressor on real weather data.

import os, time, tracemalloc
from datetime import datetime
import numpy as np
import pandas as pd
from meteostat import daily as meteostat_daily
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -- Trains a model and returns (fitted_model, time_sec, memory_mb) ----------
def train_and_profile(model, X_train, y_train):
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    mem_bytes = sum(s.size_diff for s in snap_after.compare_to(snap_before, "lineno") if s.size_diff > 0)
    return model, round(elapsed, 4), round(mem_bytes / 1024**2, 4)

# -- 1. HYBRID DATA LOADING ---------------------------------------------------
# Use local CSV cache if it exists; otherwise fetch from meteostat and save it.
CACHE = "mecca_weather_data.csv"

if os.path.exists(CACHE):
    print(f"Loading data from local cache: {CACHE}")
    df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
else:
    print("Cache not found. Fetching from meteostat API (station 41030 — Makkah)...")
    start_date = datetime(2021, 5, 10)
    end_date   = datetime(2026, 5, 10)
    df = meteostat_daily("41030", start=start_date, end=end_date).fetch()
    # Normalise any Parameter enum column names to plain strings
    df.columns = [str(c).split(".")[-1].lower() for c in df.columns]
    df.to_csv(CACHE)
    print(f"Data fetched and saved to '{CACHE}' for future runs.")

print(f"Dataset: {len(df)} rows | Columns: {list(df.columns)}\n")

# -- 2. PREPROCESSING ---------------------------------------------------------
# Target: 'temp' (daily avg C). Features: tmin, tmax, rhum, prcp, wspd, pres, cldc.
# Drop rows with no target, impute missing feature values with the column median,
# split 80/20, then scale features with StandardScaler.
FEATURES = ["tmin", "tmax", "rhum", "prcp", "wspd", "pres", "cldc"]
TARGET   = "temp"

FEATURES = [c for c in FEATURES if c in df.columns]  # skip any absent columns
df = df.dropna(subset=[TARGET])

X = df[FEATURES].to_numpy(dtype=float, na_value=np.nan)
y = df[TARGET].to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="median")     # fill NaNs with column medians
X_train = imputer.fit_transform(X_train)       # fit only on training data
X_test  = imputer.transform(X_test)

scaler  = StandardScaler()                     # zero-mean, unit-variance scaling
X_train = scaler.fit_transform(X_train)        # fit only on training data
X_test  = scaler.transform(X_test)

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}\n")

# -- 3. TRAIN & PROFILE BOTH MODELS -------------------------------------------
models = {
    "Linear Regression":       LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
}

results = []
for name, model in models.items():
    fitted, t, mem = train_and_profile(model, X_train, y_train)
    mae = round(mean_absolute_error(y_test, fitted.predict(X_test)), 4)
    r2  = round(r2_score(y_test, fitted.predict(X_test)), 4)
    results.append({"Model": name, "MAE (C)": mae, "R2 Score": r2, "Time (s)": t, "Memory (MB)": mem})
    print(f"  {name}: MAE={mae} | R2={r2} | Time={t}s | Memory={mem}MB")

# -- 4. PRINT COMPARISON TABLE -------------------------------------------------
print("\n" + "="*75)
print(f"  {'Model':<28} {'MAE (C)':>10} {'R2 Score':>10} {'Time (s)':>10} {'Mem (MB)':>10}")
print("-"*75)
for r in results:
    print(f"  {r['Model']:<28} {r['MAE (C)']:>10} {r['R2 Score']:>10} {r['Time (s)']:>10} {r['Memory (MB)']:>10}")
print("="*75)