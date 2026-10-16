
from data_processing.loader import (
    load_produksi,
    load_konsumsi,
    load_penduduk
)
from data_processing.preprocessing import (
    prepare_production,
    prepare_consumption,
    prepare_population
)
from data_processing.demand_processing import preprocess_konsumsi


def main():
    produksi = prepare_production(load_produksi())
    konsumsi = prepare_consumption(load_konsumsi())
    penduduk = prepare_population(load_penduduk())

    konsumsi_wilayah, konsumsi_provinsi = preprocess_konsumsi(
        load_konsumsi()
    )

    print("\n=== VALIDASI DATA ===")
    print("Produksi:", produksi.shape)
    print("Konsumsi beras:", konsumsi.shape)
    print("Penduduk:", penduduk.shape)
    print("Konsumsi kabupaten/kota:", konsumsi_wilayah.shape)
    print("Konsumsi agregat provinsi:", konsumsi_provinsi.shape)

    print("\nKolom produksi:", produksi.columns.tolist())
    print("Kolom konsumsi:", konsumsi.columns.tolist())
    print("Kolom penduduk:", penduduk.columns.tolist())

    print("\nPeriode produksi:", produksi["year"].min(),
          "-", produksi["year"].max())
    print("Periode konsumsi:", konsumsi["year"].min(),
          "-", konsumsi["year"].max())
    print("Periode penduduk:", penduduk["year"].min(),
          "-", penduduk["year"].max())

    print("\nJumlah wilayah produksi:",
          produksi["region_key"].nunique())
    print("Jumlah wilayah konsumsi:",
          konsumsi_wilayah["region_key"].nunique())
    print("Jumlah wilayah penduduk:",
          penduduk["region_key"].nunique())

    # Audit apakah wilayah konsumsi memiliki data penduduk
    # pada tahun yang sama.
    check = konsumsi_wilayah.merge(
        penduduk[["year", "region_key"]],
        on=["year", "region_key"],
        how="left",
        indicator=True,
        validate="one_to_one"
    )

    unmatched = check[check["_merge"] == "left_only"]

    print("\nBaris konsumsi tanpa pasangan penduduk:",
          len(unmatched))

    if not unmatched.empty:
        print(
            unmatched[["year", "region"]]
            .drop_duplicates()
            .to_string(index=False)
        )

    print("\nVALIDASI SELESAI")


if __name__ == "__main__":
    main()
