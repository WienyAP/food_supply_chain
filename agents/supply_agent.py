
import pandas as pd
from agents.base_agent import BaseAgent


class SupplyAgent(BaseAgent):

    def __init__(self):
        super().__init__("SupplyAgent")

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {
            "year",
            "month",
            "region",
            "region_key",
            "produksi_beras"
        }

        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"SupplyAgent kekurangan kolom: {sorted(missing)}"
            )

        df = data.copy()

        annual = (
            df.groupby(
                ["year", "region", "region_key"],
                as_index=False
            )
            .agg(
                produksi_beras=(
                    "produksi_beras", "sum"
                ),
                jumlah_bulan_tercatat=(
                    "month", "nunique"
                )
            )
        )

        annual["cakupan_bulan_lengkap"] = (
            annual["jumlah_bulan_tercatat"] == 12
        )

        annual = annual.sort_values(
            ["region_key", "year"]
        ).reset_index(drop=True)

        annual["pertumbuhan_produksi_pct"] = (
            annual.groupby("region_key")["produksi_beras"]
            .pct_change(fill_method=None)
            * 100
        )

        annual["rata_produksi_historis"] = (
            annual.groupby("region_key")["produksi_beras"]
            .transform("mean")
        )

        annual["rasio_produksi_historis"] = (
            annual["produksi_beras"]
            / annual["rata_produksi_historis"].replace(
                0, float("nan")
            )
        )

        annual["supply_status"] = "normal"

        annual.loc[
            annual["rasio_produksi_historis"] >= 1.20,
            "supply_status"
        ] = "high"

        annual.loc[
            annual["rasio_produksi_historis"] < 0.80,
            "supply_status"
        ] = "low"

        return annual
