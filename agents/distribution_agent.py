
import pandas as pd

from .base_agent import BaseAgent


class DistributionAgent(BaseAgent):
    """Mengolah catatan distribusi dan merangkum aliran beras."""

    REQUIRED_COLUMNS = [
        "Tahun",
        "Bulan",
        "Komoditas",
        "Lokasi Asal Pasokan",
        "Provinsi Asal Pasokan",
        "Lokasi Tujuan Pengiriman",
        "Provinsi Tujuan Pengiriman",
        "Volume",
        "Satuan",
    ]

    def __init__(self):
        super().__init__("DistributionAgent")

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            raise ValueError("Data distribusi kosong.")

        missing = [
            col for col in self.REQUIRED_COLUMNS
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Kolom wajib distribusi tidak ditemukan: {missing}"
            )

        df = data[self.REQUIRED_COLUMNS].copy()

        # Rapikan teks dan pertahankan nilai kosong sebagai NA.
        text_columns = [
            "Bulan",
            "Komoditas",
            "Lokasi Asal Pasokan",
            "Provinsi Asal Pasokan",
            "Lokasi Tujuan Pengiriman",
            "Provinsi Tujuan Pengiriman",
            "Satuan",
        ]

        for col in text_columns:
            df[col] = df[col].astype("string").str.strip()

        # Filter komoditas beras.
        df = df[
            df["Komoditas"].str.casefold().eq("beras")
        ].copy()

        if df.empty:
            raise ValueError(
                "Tidak ditemukan catatan distribusi beras."
            )

        # Validasi tahun dan volume setelah filter beras.
        df["Tahun"] = pd.to_numeric(
            df["Tahun"], errors="coerce"
        )

        df["Volume"] = pd.to_numeric(
            df["Volume"], errors="coerce"
        )

        invalid = df["Tahun"].isna() | df["Volume"].isna()

        if invalid.any():
            sample = df.loc[
                invalid, self.REQUIRED_COLUMNS
            ].head(10)

            raise ValueError(
                "Terdapat Tahun atau Volume yang tidak valid "
                "pada catatan beras.\n"
                + sample.to_string(index=False)
            )

        df["Tahun"] = df["Tahun"].astype(int)

        if (df["Volume"] < 0).any():
            raise ValueError(
                "Ditemukan volume distribusi negatif."
            )

        # Untuk sementara, hanya menerima satuan kg.
        unit = df["Satuan"].str.casefold()

        if not unit.eq("kg").all():
            invalid_units = sorted(
                unit[~unit.eq("kg")].dropna().unique().tolist()
            )

            raise ValueError(
                "Volume harus menggunakan satuan kg. "
                f"Satuan lain atau kosong ditemukan: {invalid_units}"
            )

        # Konversi volume.
        df["volume_kg"] = df["Volume"]
        df["volume_ton"] = df["volume_kg"] / 1000

        # Klasifikasi arus berdasarkan provinsi.
        same_province = (
            df["Provinsi Asal Pasokan"].str.casefold()
            == df["Provinsi Tujuan Pengiriman"].str.casefold()
        )

        province_complete = (
            df["Provinsi Asal Pasokan"].notna()
            & df["Provinsi Tujuan Pengiriman"].notna()
        )

        df["jenis_arus"] = "provinsi_tidak_diketahui"

        df.loc[
            province_complete & same_province,
            "jenis_arus"
        ] = "dalam_provinsi"

        df.loc[
            province_complete & ~same_province,
            "jenis_arus"
        ] = "antarprovinsi"

        df["arus_provinsi"] = (
            df["Provinsi Asal Pasokan"].fillna("Tidak diketahui")
            + " -> "
            + df["Provinsi Tujuan Pengiriman"].fillna(
                "Tidak diketahui"
            )
        )

        return df.reset_index(drop=True)

    @staticmethod
    def summarize_by_route(
        distribution: pd.DataFrame
    ) -> pd.DataFrame:
        """Ringkasan arus berdasarkan provinsi asal dan tujuan."""

        return (
            distribution.groupby(
                [
                    "Provinsi Asal Pasokan",
                    "Provinsi Tujuan Pengiriman",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                jumlah_catatan=("volume_kg", "size"),
                total_volume_kg=("volume_kg", "sum"),
                total_volume_ton=("volume_ton", "sum"),
            )
            .sort_values(
                "total_volume_kg", ascending=False
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def summarize_monthly(
        distribution: pd.DataFrame
    ) -> pd.DataFrame:
        """Ringkasan volume beras per tahun dan bulan."""

        return (
            distribution.groupby(
                ["Tahun", "Bulan"],
                as_index=False,
                dropna=False,
            )
            .agg(
                jumlah_catatan=("volume_kg", "size"),
                total_volume_kg=("volume_kg", "sum"),
                total_volume_ton=("volume_ton", "sum"),
            )
            .sort_values(["Tahun", "Bulan"])
            .reset_index(drop=True)
        )

    @staticmethod
    def summarize_by_location(
        distribution: pd.DataFrame
    ) -> pd.DataFrame:
        """Ringkasan volume menurut lokasi asal dan tujuan."""

        return (
            distribution.groupby(
                [
                    "Lokasi Asal Pasokan",
                    "Provinsi Asal Pasokan",
                    "Lokasi Tujuan Pengiriman",
                    "Provinsi Tujuan Pengiriman",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                jumlah_catatan=("volume_kg", "size"),
                total_volume_kg=("volume_kg", "sum"),
                total_volume_ton=("volume_ton", "sum"),
            )
            .sort_values(
                "total_volume_kg", ascending=False
            )
            .reset_index(drop=True)
        )
