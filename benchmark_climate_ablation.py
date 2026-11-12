
import os
import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "outputs/monthly_supply_climate_features.csv"
OUTPUT_DIR = "outputs/climate_ablation"

TRAIN_START = pd.Timestamp("2020-02-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-01")

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 0, 12)

RAINFALL_COL = "rainfall_mm_month_lag1"

FULL_CLIMATE_COLS = [
    "rainfall_mm_month_lag1",
    "temp_avg_c_lag1",
    "humidity_pct_lag1",
    "solar_mj_lag1"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)
df["period"] = pd.to_datetime(df["period"])

possible_region_cols = [
    "region_key",
    "region_standard",
    "region"
]

region_col = next(
    (c for c in possible_region_cols if c in df.columns),
    None
)

if region_col is None:
    raise ValueError("Kolom wilayah tidak ditemukan.")

required_cols = [
    "period",
    "production_ton",
    RAINFALL_COL
] + FULL_CLIMATE_COLS

missing = [
    c for c in required_cols if c not in df.columns
]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

df = df.sort_values([region_col, "period"])

print("Jumlah wilayah:", df[region_col].nunique())
print("Jumlah data:", len(df))


# ============================================================
# 3. MODEL FUNCTIONS
# ============================================================

def fit_sarima(train):

    model = SARIMAX(
        endog=train["production_ton"],
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    return model.fit(
        disp=False,
        maxiter=100
    )


def fit_sarimax(train, climate_cols):

    model = SARIMAX(
        endog=train["production_ton"],
        exog=train[climate_cols],
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    return model.fit(
        disp=False,
        maxiter=100
    )


# ============================================================
# 4. FORECASTING EXPERIMENT
# ============================================================

results = []
failed = []

for region, group in df.groupby(region_col):

    print(f"\nProcessing: {region}")

    try:

        group = (
            group.sort_values("period")
            .set_index("period")
            .asfreq("MS")
        )

        test = group.loc[TEST_START:TEST_END]

        if len(test) != 12:
            raise ValueError(
                f"Testing data tidak lengkap: {len(test)}"
            )

        for target_date in test.index:

            origin = (
                target_date - pd.DateOffset(months=1)
            )

            # Data yang tersedia sampai bulan sebelumnya
            train = group.loc[
                TRAIN_START:origin
            ].copy()

            if len(train) < 24:
                raise ValueError(
                    "Training data kurang dari 24 bulan"
                )

            actual = group.loc[
                target_date, "production_ton"
            ]

            if pd.isna(actual):
                continue

            # Pastikan seluruh model menggunakan
            # observasi target yang sama
            seasonal_date = (
                target_date - pd.DateOffset(months=12)
            )

            naive_value = group.loc[
                seasonal_date, "production_ton"
            ]

            if pd.isna(naive_value):
                continue

            if train["production_ton"].isna().any():
                raise ValueError("Missing production")

            # ------------------------------------------------
            # MODEL 1: SEASONAL NAIVE
            # ------------------------------------------------

            forecast_naive = float(naive_value)

            # ------------------------------------------------
            # MODEL 2: SARIMA
            # ------------------------------------------------

            sarima_fitted = fit_sarima(train)

            forecast_sarima = float(
                sarima_fitted.forecast(steps=1).iloc[0]
            )

            # ------------------------------------------------
            # MODEL 3: SARIMAX RAINFALL ONLY
            # ------------------------------------------------

            if train[RAINFALL_COL].isna().any():
                raise ValueError(
                    "Missing rainfall pada training"
                )

            rainfall_target = group.loc[
                [target_date], [RAINFALL_COL]
            ]

            if rainfall_target.isna().any().any():
                raise ValueError(
                    "Missing rainfall pada target"
                )

            rainfall_fitted = fit_sarimax(
                train,
                [RAINFALL_COL]
            )

            forecast_rainfall = float(
                rainfall_fitted.get_forecast(
                    steps=1,
                    exog=rainfall_target
                ).predicted_mean.iloc[0]
            )

            # ------------------------------------------------
            # MODEL 4: SARIMAX FULL CLIMATE
            # ------------------------------------------------

            if train[FULL_CLIMATE_COLS].isna().any().any():
                raise ValueError(
                    "Missing climate pada training"
                )

            full_target = group.loc[
                [target_date], FULL_CLIMATE_COLS
            ]

            if full_target.isna().any().any():
                raise ValueError(
                    "Missing climate pada target"
                )

            full_fitted = fit_sarimax(
                train,
                FULL_CLIMATE_COLS
            )

            forecast_full = float(
                full_fitted.get_forecast(
                    steps=1,
                    exog=full_target
                ).predicted_mean.iloc[0]
            )

            # ------------------------------------------------
            # SAVE PAIRED PREDICTIONS
            # ------------------------------------------------

            forecasts = {
                "SeasonalNaive": forecast_naive,
                "SARIMA": forecast_sarima,
                "SARIMAX_Rainfall": forecast_rainfall,
                "SARIMAX_Full": forecast_full
            }

            for model_name, forecast_value in forecasts.items():

                results.append({
                    "region": region,
                    "origin": origin,
                    "target": target_date,
                    "horizon": 1,
                    "model": model_name,
                    "actual_ton": float(actual),
                    "forecast_ton": forecast_value,
                    "abs_error": abs(
                        float(actual) - forecast_value
                    ),
                    "squared_error": (
                        float(actual) - forecast_value
                    ) ** 2
                })

    except Exception as e:

        failed.append({
            "region": region,
            "error": str(e)
        })

        print("FAILED:", str(e))


# ============================================================
# 5. SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame(results)

if predictions.empty:
    raise ValueError("Tidak ada prediksi yang berhasil.")

predictions.to_csv(
    f"{OUTPUT_DIR}/predictions.csv",
    index=False
)

print("\nTotal predictions:", len(predictions))
print(
    "Wilayah berhasil:",
    predictions["region"].nunique()
)


# ============================================================
# 6. OVERALL METRICS
# ============================================================

metrics = (
    predictions
    .groupby("model")
    .agg(
        MAE=("abs_error", "mean"),
        MSE=("squared_error", "mean"),
        n_predictions=("abs_error", "count")
    )
    .reset_index()
)

metrics["RMSE"] = np.sqrt(metrics["MSE"])

metrics = metrics[
    ["model", "MAE", "RMSE", "n_predictions"]
]

metrics.to_csv(
    f"{OUTPUT_DIR}/metrics_overall.csv",
    index=False
)

print("\n===== OVERALL METRICS =====")
print(metrics.to_string(index=False))


# ============================================================
# 7. METRICS PER REGION
# ============================================================

region_metrics = (
    predictions
    .groupby(["region", "model"])
    .agg(
        MAE=("abs_error", "mean"),
        MSE=("squared_error", "mean"),
        n_predictions=("abs_error", "count")
    )
    .reset_index()
)

region_metrics["RMSE"] = np.sqrt(
    region_metrics["MSE"]
)

region_metrics.to_csv(
    f"{OUTPUT_DIR}/metrics_per_region.csv",
    index=False
)


# ============================================================
# 8. COMPARE IMPROVEMENT AGAINST SARIMA
# ============================================================

mae_table = metrics.set_index("model")["MAE"]

comparison = pd.DataFrame({
    "model": mae_table.index,
    "MAE": mae_table.values
})

sarima_mae = mae_table["SARIMA"]

comparison["MAE_change_vs_SARIMA_pct"] = (
    (comparison["MAE"] - sarima_mae)
    / sarima_mae
) * 100

comparison.to_csv(
    f"{OUTPUT_DIR}/comparison_vs_sarima.csv",
    index=False
)

print("\n===== MAE COMPARISON VS SARIMA =====")
print(comparison.to_string(index=False))


# ============================================================
# 9. FAILED REGIONS
# ============================================================

if failed:

    pd.DataFrame(failed).to_csv(
        f"{OUTPUT_DIR}/failed_regions.csv",
        index=False
    )

    print("\nFailed regions:")
    print(pd.DataFrame(failed).to_string(index=False))

else:

    print("\nAll regions processed successfully.")


print("\n====================================")
print("CLIMATE ABLATION EXPERIMENT FINISHED")
print("Output:", OUTPUT_DIR)
print("====================================")

