
import os
import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ============================================================
# 1. KONFIGURASI
# ============================================================

INPUT_FILE = "outputs/monthly_supply_climate_features.csv"
OUTPUT_DIR = "outputs/forecasting_benchmark_final"

TRAIN_START = pd.Timestamp("2020-02-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-01")

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 0, 12)

CLIMATE_LAGS = [
    "rainfall_mm_month_lag1",
    "temp_avg_c_lag1",
    "humidity_pct_lag1",
    "solar_mj_lag1"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DAN PEMERIKSAAN DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["period"] = pd.to_datetime(df["period"])

# Deteksi nama kolom wilayah
possible_region_cols = [
    "region_key",
    "region_standard",
    "region"
]

region_col = next(
    (col for col in possible_region_cols if col in df.columns),
    None
)

if region_col is None:
    raise ValueError("Kolom wilayah tidak ditemukan.")

required_cols = [
    "period",
    "production_ton"
] + CLIMATE_LAGS

missing_cols = [
    col for col in required_cols if col not in df.columns
]

if missing_cols:
    raise ValueError(f"Kolom tidak ditemukan: {missing_cols}")

df = df.sort_values([region_col, "period"])

print("Jumlah wilayah:", df[region_col].nunique())
print("Jumlah baris:", len(df))
print("Kolom wilayah:", region_col)


# ============================================================
# 3. FUNGSI FITTING SARIMA DAN SARIMAX
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


def fit_sarimax(train):
    model = SARIMAX(
        endog=train["production_ton"],
        exog=train[CLIMATE_LAGS],
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
# 4. EKSPERIMEN FORECASTING
# ============================================================

results = []
failed = []

for region, group in df.groupby(region_col):

    print(f"\nMemproses: {region}")

    try:
        group = (
            group.sort_values("period")
            .set_index("period")
            .asfreq("MS")
        )

        test = group.loc[TEST_START:TEST_END]

        if len(test) != 12:
            raise ValueError(
                f"Data testing tidak lengkap: {len(test)}"
            )

        for target_date in test.index:

            origin = target_date - pd.DateOffset(months=1)

            # Data training hanya sampai bulan sebelum target
            train = group.loc[
                TRAIN_START:origin
            ].copy()

            if len(train) < 24:
                raise ValueError(
                    f"Training terlalu sedikit: {len(train)}"
                )

            actual = group.loc[
                target_date, "production_ton"
            ]

            if pd.isna(actual):
                continue

            # ------------------------------------------------
            # MODEL 1: SEASONAL NAIVE
            # ------------------------------------------------

            seasonal_date = (
                target_date - pd.DateOffset(months=12)
            )

            naive_forecast = group.loc[
                seasonal_date, "production_ton"
            ]

            if pd.notna(naive_forecast):

                results.append({
                    "region": region,
                    "origin": origin,
                    "target": target_date,
                    "horizon": 1,
                    "model": "SeasonalNaive",
                    "actual_ton": float(actual),
                    "forecast_ton": float(naive_forecast)
                })

            # ------------------------------------------------
            # MODEL 2: SARIMA
            # Refit setiap forecast origin
            # ------------------------------------------------

            sarima_train = train["production_ton"]

            if sarima_train.isna().any():
                raise ValueError(
                    "Missing production pada training SARIMA"
                )

            sarima_fitted = fit_sarima(train)

            sarima_forecast = sarima_fitted.forecast(
                steps=1
            )

            sarima_value = float(
                sarima_forecast.iloc[0]
            )

            results.append({
                "region": region,
                "origin": origin,
                "target": target_date,
                "horizon": 1,
                "model": "SARIMA",
                "actual_ton": float(actual),
                "forecast_ton": sarima_value
            })

            # ------------------------------------------------
            # MODEL 3: SARIMAX + CLIMATE
            # Refit setiap forecast origin
            # ------------------------------------------------

            if train[CLIMATE_LAGS].isna().any().any():
                raise ValueError(
                    "Missing climate pada training SARIMAX"
                )

            exog_target = group.loc[
                [target_date], CLIMATE_LAGS
            ]

            if exog_target.isna().any().any():
                raise ValueError(
                    "Missing climate pada target SARIMAX"
                )

            sarimax_fitted = fit_sarimax(train)

            sarimax_forecast = sarimax_fitted.get_forecast(
                steps=1,
                exog=exog_target
            )

            sarimax_value = float(
                sarimax_forecast.predicted_mean.iloc[0]
            )

            results.append({
                "region": region,
                "origin": origin,
                "target": target_date,
                "horizon": 1,
                "model": "SARIMAX_Climate",
                "actual_ton": float(actual),
                "forecast_ton": sarimax_value
            })

    except Exception as e:

        failed.append({
            "region": region,
            "error": str(e)
        })

        print("Gagal:", str(e))


# ============================================================
# 5. MENYIMPAN HASIL PREDIKSI
# ============================================================

predictions = pd.DataFrame(results)

if predictions.empty:
    raise ValueError(
        "Tidak ada prediksi yang berhasil dihasilkan."
    )

predictions["abs_error"] = (
    predictions["actual_ton"]
    - predictions["forecast_ton"]
).abs()

predictions["squared_error"] = (
    predictions["actual_ton"]
    - predictions["forecast_ton"]
) ** 2

predictions.to_csv(
    f"{OUTPUT_DIR}/predictions.csv",
    index=False
)

print("\nJumlah prediksi:", len(predictions))
print("Jumlah wilayah berhasil:",
      predictions["region"].nunique())


# ============================================================
# 6. EVALUASI MAE DAN RMSE
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

print("\n===== HASIL EVALUASI KESELURUHAN =====")
print(metrics.to_string(index=False))


# ============================================================
# 7. EVALUASI PER WILAYAH
# ============================================================

metrics_region = (
    predictions
    .groupby(["region", "model"])
    .agg(
        MAE=("abs_error", "mean"),
        MSE=("squared_error", "mean"),
        n_predictions=("abs_error", "count")
    )
    .reset_index()
)

metrics_region["RMSE"] = np.sqrt(
    metrics_region["MSE"]
)

metrics_region.to_csv(
    f"{OUTPUT_DIR}/metrics_per_region.csv",
    index=False
)



# ============================================================
# 8. PERBANDINGAN MODEL
# ============================================================

comparison = (
    metrics
    .set_index("model")[["MAE", "RMSE"]]
    .T
)

print("\n===== PERBANDINGAN MODEL =====")
print(comparison.to_string())


# ============================================================
# 9. SIMPAN INFORMASI KEGAGALAN
# ============================================================

if failed:

    failed_df = pd.DataFrame(failed)

    failed_df.to_csv(
        f"{OUTPUT_DIR}/failed_regions.csv",
        index=False
    )

    print("\nWilayah gagal:")
    print(failed_df.to_string(index=False))

else:

    print("\nSemua wilayah berhasil diproses.")


print("\n======================================")
print("EKSPERIMEN SELESAI")
print("Hasil tersimpan di:", OUTPUT_DIR)
print("======================================")
