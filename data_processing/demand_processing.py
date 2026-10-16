
import pandas as pd

from data_processing.preprocessing import prepare_consumption


def preprocess_konsumsi(df: pd.DataFrame):
    """
    Membersihkan konsumsi beras dan memisahkan data agregat provinsi
    dari data kabupaten/kota.

    Kolom Konsumsi_Pangan dipertahankan apa adanya.
    Satuannya harus diverifikasi dari metadata sumber.
    """
    result = prepare_consumption(df)

    province_mask = (
        result["region"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("jawa timur")
    )

    df_provinsi = result.loc[province_mask].copy()
    df_wilayah = result.loc[~province_mask].copy()

    # Validasi duplikasi dilakukan per wilayah dan tahun,
    # bukan hanya per wilayah.
    if df_wilayah.duplicated(
        ["year", "region_key"]
    ).any():
        raise ValueError(
            "Ditemukan duplikasi wilayah-tahun pada konsumsi."
        )

    return (
        df_wilayah.reset_index(drop=True),
        df_provinsi.reset_index(drop=True)
    )
