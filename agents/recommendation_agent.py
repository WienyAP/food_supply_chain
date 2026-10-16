
import pandas as pd
from .base_agent import BaseAgent


class RecommendationAgent(BaseAgent):
    """Menyusun kandidat berdasarkan neraca produksi dan konsumsi."""

    def __init__(self):
        super().__init__("RecommendationAgent")

    def run(self, data: dict) -> pd.DataFrame:
        supply = data["supply"].copy()
        demand = data.get("demand")
        distribution = data.get("distribution")

        required = {
            "year", "region", "region_key", "produksi_beras"
        }
        missing = required - set(supply.columns)
        if missing:
            raise ValueError(f"Kolom supply tidak tersedia: {sorted(missing)}")

        if supply.duplicated(["year", "region_key"]).any():
            raise ValueError("Supply memiliki duplikasi tahun-wilayah.")

        # Gabungkan estimasi konsumsi jika tersedia.
        if demand is not None and not demand.empty:
            demand_cols = [
                "year", "region_key", "estimasi_konsumsi_ton_tahun"
            ]
            missing_demand = set(demand_cols) - set(demand.columns)
            if missing_demand:
                raise ValueError(
                    f"Kolom demand tidak tersedia: {sorted(missing_demand)}"
                )

            demand = demand[demand_cols].copy()
            if demand.duplicated(["year", "region_key"]).any():
                raise ValueError("Demand memiliki duplikasi tahun-wilayah.")

            supply = supply.merge(
                demand,
                on=["year", "region_key"],
                how="left",
                validate="one_to_one",
            )

            supply["neraca_indikatif_ton"] = (
                supply["produksi_beras"]
                - supply["estimasi_konsumsi_ton_tahun"]
            )
            supply["dasar_analisis"] = "produksi_dikurangi_estimasi_konsumsi"
        else:
            supply["estimasi_konsumsi_ton_tahun"] = float("nan")
            supply["neraca_indikatif_ton"] = float("nan")
            supply["dasar_analisis"] = "konsumsi_belum_tersedia"

        latest_year = supply["year"].max()
        latest = supply[supply["year"] == latest_year].copy()

        # Gunakan neraca jika data konsumsi tersedia.
        if latest["neraca_indikatif_ton"].notna().all():
            sources = latest[latest["neraca_indikatif_ton"] > 0]
            targets = latest[latest["neraca_indikatif_ton"] < 0]
        else:
            # Fallback sementara: status relatif produksi.
            if "supply_status" not in latest.columns:
                raise ValueError("Kolom supply_status tidak tersedia untuk fallback.")

            sources = latest[latest["supply_status"] == "high"]
            targets = latest[latest["supply_status"] == "low"]

        candidates = []

        for _, source in sources.iterrows():
            for _, target in targets.iterrows():
                if source["region_key"] == target["region_key"]:
                    continue

                if pd.notna(source["neraca_indikatif_ton"]) and source["neraca_indikatif_ton"] <= 0:
                    continue
                if pd.notna(target["neraca_indikatif_ton"]) and target["neraca_indikatif_ton"] >= 0:
                    continue

                candidates.append({
                    "year": int(latest_year),
                    "wilayah_asal": source["region"],
                    "wilayah_tujuan": target["region"],
                    "produksi_asal_ton": source["produksi_beras"],
                    "kebutuhan_asal_ton": source["estimasi_konsumsi_ton_tahun"],
                    "neraca_asal_ton": source["neraca_indikatif_ton"],
                    "produksi_tujuan_ton": target["produksi_beras"],
                    "kebutuhan_tujuan_ton": target["estimasi_konsumsi_ton_tahun"],
                    "neraca_tujuan_ton": target["neraca_indikatif_ton"],
                    "dasar_analisis": source["dasar_analisis"],
                    "status_rekomendasi": "kandidat_prioritas_analitis",
                    "dukungan_catatan_rute": "belum_tervalidasi",
                })

        result = pd.DataFrame(candidates)

        if result.empty:
            return pd.DataFrame(columns=[
                "year", "wilayah_asal", "wilayah_tujuan",
                "neraca_asal_ton", "neraca_tujuan_ton",
                "dasar_analisis", "status_rekomendasi",
                "dukungan_catatan_rute",
            ])
  
        # Prioritaskan 10 kandidat dengan defisit tujuan terbesar.
        # Jika defisit sama, dahulukan wilayah asal dengan neraca positif terbesar.
        result = result.sort_values(
            ["neraca_tujuan_ton", "neraca_asal_ton"],
            ascending=[True, False]
        ).head(10).reset_index(drop=True)

        # Periksa catatan rute secara persis; tidak ditemukan bukan berarti
        # rute tidak pernah ada di dunia nyata.
        if distribution is not None and not distribution.empty:
            route_cols = {
                "Lokasi Asal Pasokan", "Lokasi Tujuan Pengiriman"
            }
            if route_cols.issubset(distribution.columns):
                route_pairs = set(zip(
                    distribution["Lokasi Asal Pasokan"].astype(str).str.strip().str.casefold(),
                    distribution["Lokasi Tujuan Pengiriman"].astype(str).str.strip().str.casefold(),
                ))

                for idx, row in result.iterrows():
                    pair = (
                        str(row["wilayah_asal"]).strip().casefold(),
                        str(row["wilayah_tujuan"]).strip().casefold(),
                    )
                    result.loc[idx, "dukungan_catatan_rute"] = (
                        "terdapat_catatan_rute"
                        if pair in route_pairs
                        else "belum_tervalidasi"
                    )

        return result.reset_index(drop=True)
