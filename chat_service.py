"""
Shared chatbot service for Streamlit and FastAPI.
Business/data handlers are reused from the existing app.py implementation.
"""
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

def load_project_data():
    files = {
        "supply": [OUTPUT_DIR / "supply_result.csv", BASE_DIR / "data" / "Produksi Beras all 2020-2025.csv"],
        "distribution": [OUTPUT_DIR / "distribution_processed.csv", BASE_DIR / "data" / "distribusi_per_wilayah_tujuan.csv"],
        "recommendations": [OUTPUT_DIR / "recommendation_candidates.csv", BASE_DIR / "data" / "recommendation_candidates.csv"],
        "forecast": [OUTPUT_DIR / "forecast_supply_demand_balance_2026.csv"],
    }

    data = {}
    for name, candidates in files.items():
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        data[name] = pd.read_csv(path) if path else pd.DataFrame()
    return data


def build_data_context(data):
    context_parts = []

    supply = data.get("supply", pd.DataFrame())
    distribution = data.get("distribution", pd.DataFrame())
    recommendations = data.get("recommendations", pd.DataFrame())
    forecast = data.get("forecast", pd.DataFrame())

    if not supply.empty:
        years = (
            sorted(supply["year"].dropna().unique().tolist())
            if "year" in supply.columns
            else "tidak diketahui"
        )

        context_parts.append(
            "DATA PRODUKSI BERAS\n"
            f"Kolom: {list(supply.columns)}\n"
            f"Jumlah baris: {len(supply)}\n"
            f"Tahun tersedia: {years}\n"
            f"Contoh data:\n{supply.head(10).to_string(index=False)}"
        )

    if not distribution.empty:
        context_parts.append(
            "DATA DISTRIBUSI BERAS\n"
            f"Kolom: {list(distribution.columns)}\n"
            f"Jumlah baris: {len(distribution)}\n"
            f"Contoh data:\n"
            f"{distribution.head(10).to_string(index=False)}"
        )

    if not recommendations.empty:
        context_parts.append(
            "KANDIDAT REKOMENDASI DISTRIBUSI\n"
            f"Kolom: {list(recommendations.columns)}\n"
            f"Jumlah baris: {len(recommendations)}\n"
            f"Contoh data:\n"
            f"{recommendations.head(10).to_string(index=False)}"
        )

    if not forecast.empty:
        context_parts.append(
            "PROYEKSI SUPPLY-DEMAND 2026 (SEASONAL NAIVE)\n"
            f"Jumlah baris: {len(forecast)}\n"
            f"Kolom: {list(forecast.columns)}\n"
            f"Contoh data: {forecast.head(8).to_string(index=False)}\n"
            "Catatan: balance merupakan indikator forecast supply dikurangi estimasi demand, bukan stok aktual."
        )

    if not context_parts:
        return "Belum ada hasil analisis yang dapat dibaca."

    return "\n\n".join(context_parts)


def search_production(question, supply):
    """Mencari data produksi atau tren historis dari seluruh dataset."""
    import re

    if supply.empty:
        return None

    q = question.lower()

    # Cari nama wilayah, prioritaskan nama terpanjang
    regions = supply["region"].dropna().unique().tolist()
    matched_region = None

    for region in sorted(regions, key=len, reverse=True):
        if region.lower() in q:
            matched_region = region
            break

    # Kenali tahun dalam pertanyaan
    years = sorted({
        int(y) for y in re.findall(r"\b20\d{2}\b", q)
    })

    if matched_region is None:
        return None

    # Filter data wilayah
    region_data = supply[
        supply["region"].str.lower() == matched_region.lower()
    ].copy()

    if region_data.empty:
        return f"Data untuk {matched_region} tidak ditemukan."

    # Analisis tren jika pengguna menyebut rentang dua tahun
    trend_keywords = [
        "tren", "trend", "perkembangan", "perbandingan",
        "dari", "sampai", "hingga", "antara"
    ]

    is_trend_question = (
        len(years) >= 2
        and any(keyword in q for keyword in trend_keywords)
    )

    if is_trend_question:
        start_year = min(years)
        end_year = max(years)

        result = region_data[
            region_data["year"].between(start_year, end_year)
        ].sort_values("year")

        if result.empty:
            return (
                f"Tidak ada data untuk {matched_region} "
                f"pada rentang {start_year}–{end_year}."
            )

        lines = [
            "### Tren Produksi Beras",
            "",
            f"**Wilayah:** {matched_region}",
            f"**Periode:** {start_year}–{end_year}",
            "",
            "| Tahun | Produksi beras | Status relatif historis |",
            "|---:|---:|---|",
        ]

        for _, row in result.iterrows():
            production = float(row["produksi_beras"])
            status = row.get("supply_status", "tidak tersedia")

            lines.append(
                f"| {int(row['year'])} | {production:,.2f} | {status} |"
            )

        first = result.iloc[0]
        last = result.iloc[-1]

        if start_year in result["year"].values and end_year in result["year"].values:
            first_value = float(
                result.loc[result["year"] == start_year, "produksi_beras"].iloc[0]
            )
            last_value = float(
                result.loc[result["year"] == end_year, "produksi_beras"].iloc[0]
            )

            if first_value != 0:
                change_pct = (
                    (last_value - first_value) / abs(first_value)
                ) * 100

                lines.extend([
                    "",
                    f"**Perubahan {start_year}–{end_year}:** "
                    f"{change_pct:+.2f}%",
                ])

        lines.extend([
            "",
            "*Catatan: satuan produksi belum diverifikasi. "
            "Perubahan persentase dihitung dari nilai awal dan akhir "
            "yang tersedia, bukan dari penjumlahan seluruh tahun.*",
        ])

        return "\n".join(lines)

    # Pencarian satu tahun
    if len(years) == 1:
        requested_year = years[0]

        result = region_data[
            region_data["year"] == requested_year
        ]

        if result.empty:
            return (
                f"Data produksi {matched_region} tahun "
                f"{requested_year} tidak ditemukan."
            )

        row = result.iloc[0]

        return (
            "### Hasil Pencarian Produksi Beras\n\n"
            f"- **Wilayah:** {matched_region}\n"
            f"- **Tahun:** {requested_year}\n"
            f"- **Produksi beras:** "
            f"{float(row['produksi_beras']):,.2f}\n"
            f"- **Status relatif historis:** "
            f"{row.get('supply_status', 'tidak tersedia')}\n"
            f"- **Bulan tercatat:** "
            f"{row.get('jumlah_bulan_tercatat', 'tidak tersedia')}\n"
            f"- **Cakupan bulan lengkap:** "
            f"{row.get('cakupan_bulan_lengkap', 'tidak tersedia')}\n\n"
            "*Catatan: satuan produksi belum diverifikasi.*"
        )

    return None


def compare_production(question, supply):
    """Membandingkan produksi dua wilayah pada tahun tertentu."""
    import re

    if supply.empty:
        return None

    q = question.lower()

    comparison_keywords = [
        "bandingkan", "perbandingan", "selisih",
        "lebih tinggi", "lebih rendah", "perbedaan"
    ]

    if not any(word in q for word in comparison_keywords):
        return None

    # Identifikasi wilayah yang disebutkan
    regions = supply["region"].dropna().unique().tolist()
    matched_regions = [
        region for region in sorted(regions, key=len, reverse=True)
        if region.lower() in q
    ]

    # Hilangkan duplikasi
    matched_regions = list(dict.fromkeys(matched_regions))

    if len(matched_regions) < 2:
        return None

    # Ambil dua wilayah pertama yang ditemukan
    region_a, region_b = matched_regions[:2]

    # Cari tahun
    years = re.findall(r"\b20\d{2}\b", q)

    if not years:
        return None

    year = int(years[0])

    rows = supply[
        (supply["region"].isin([region_a, region_b]))
        & (supply["year"] == year)
    ]

    values = {
        row["region"]: float(row["produksi_beras"])
        for _, row in rows.iterrows()
    }

    if region_a not in values or region_b not in values:
        return (
            f"Data produksi tahun {year} belum tersedia "
            f"untuk kedua wilayah: {region_a} dan {region_b}."
        )

    value_a = values[region_a]
    value_b = values[region_b]

    difference = abs(value_a - value_b)

    if value_a > value_b:
        higher = region_a
        lower = region_b
        higher_value = value_a
        lower_value = value_b
    elif value_b > value_a:
        higher = region_b
        lower = region_a
        higher_value = value_b
        lower_value = value_a
    else:
        higher = None
        lower = None
        higher_value = value_a
        lower_value = value_b

    # Persentase dihitung relatif terhadap nilai yang lebih rendah
    if lower_value != 0:
        percentage = difference / abs(lower_value) * 100
    else:
        percentage = None

    if higher is None:
        comparison = "Kedua wilayah memiliki nilai produksi yang sama."
    else:
        comparison = (
            f"{higher} memiliki produksi lebih tinggi daripada "
            f"{lower}."
        )

    percentage_text = (
        f"{percentage:.2f}%"
        if percentage is not None
        else "tidak dapat dihitung karena nilai pembanding nol"
    )

    return (
        "### Perbandingan Produksi Beras\n\n"
        f"- **Tahun:** {year}\n"
        f"- **{region_a}:** {value_a:,.2f}\n"
        f"- **{region_b}:** {value_b:,.2f}\n"
        f"- **Selisih absolut:** {difference:,.2f}\n"
        f"- **Perbandingan:** {comparison}\n"
        f"- **Persentase selisih terhadap nilai yang lebih rendah:** "
        f"{percentage_text}\n\n"
        "*Catatan: satuan produksi belum diverifikasi. "
        "Persentase menggunakan nilai produksi yang lebih rendah "
        "sebagai pembagi.*"
    )


#================================
# INISIALISASI MODEL
#================================


def load_model():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY tidak ditemukan. Periksa file .env"
        )
    return genai.Client(api_key=api_key)

def handle_supply(question):
    data = load_project_data()
    forecast = data.get("forecast", pd.DataFrame())
    q = question.lower()
    forecast_terms = ["forecast", "proyeksi", "prediksi", "surplus", "defisit", "balance", "keseimbangan"]
    if not forecast.empty and any(term in q for term in forecast_terms):
        import re
        region_col = "region_standard"
        regions = forecast[region_col].dropna().astype(str).unique().tolist()
        matched = next((r for r in sorted(regions, key=len, reverse=True) if r.lower() in q), None)
        year_match = re.search(r"\b20\d{2}\b", q)
        month_match = re.search(r"(?:bulan\s*)?(0?[1-9]|1[0-2])\b", q)
        view = forecast.copy()
        if matched:
            view = view[view[region_col].str.lower() == matched.lower()]
        if month_match:
            view = view[view["month"] == int(month_match.group(1))]
        if view.empty:
            return "Data proyeksi tidak ditemukan untuk filter wilayah/bulan tersebut."
        if matched:
            row = view.iloc[0]
            return (
                "### Proyeksi Supply-Demand 2026\n\n"
                f"- **Wilayah:** {row[region_col]}\n"
                f"- **Periode:** {row['period']}\n"
                f"- **Forecast supply:** {row['forecast_supply_ton']:,.2f} ton\n"
                f"- **Estimasi demand:** {row['estimated_demand_ton']:,.2f} ton\n"
                f"- **Balance indicator:** {row['balance_indicator_ton']:+,.2f} ton\n"
                f"- **Status:** {row['balance_status']}\n\n"
                "Metode supply: Seasonal Naive (bulan yang sama tahun 2025). Demand 2026 dihitung dari penduduk 2026 dan konsumsi per kapita 2025 yang dialokasikan merata per bulan.\n\n"
                "*Balance bukan stok aktual atau volume yang pasti dapat dikirim.*"
            )
        monthly = view.groupby("month", as_index=False)[["forecast_supply_ton", "estimated_demand_ton", "balance_indicator_ton"]].sum()
        return (
            "### Ringkasan Proyeksi Jawa Timur 2026\n\n"
            f"- **Forecast supply tahunan:** {monthly.forecast_supply_ton.sum():,.2f} ton\n"
            f"- **Estimasi demand tahunan:** {monthly.estimated_demand_ton.sum():,.2f} ton\n"
            f"- **Balance indicator tahunan:** {monthly.balance_indicator_ton.sum():+,.2f} ton\n"
            f"- **Jumlah wilayah:** {view[region_col].nunique()}\n\n"
            "*Angka ini adalah proyeksi dan estimasi, bukan realisasi atau stok fisik.*"
        )
    supply = data["supply"]

    if supply.empty:
        return "Data produksi belum tersedia"

    # 1. Perbandingan produksi
    answer = compare_production(question, supply)

    if answer is not None:
        return answer

    # 2. Cari jawaban dari pola pertanyaan produksi
    answer = search_production(question, supply)

    if answer is not None:
        return answer

    production_keywords = [
        "produksi", "padi", "beras", "panen", "produktivitas", "konndisi produksi",
    ]

    if any(word in question.lower() for word in production_keywords):
        return summarize_production(supply, question)

    return (
        "Saya belum dapat mengenali permintaan analisis produksi ini"
        "COba sebutkan wilayah atau tahun yang diinginkan"
    )

def summarize_production(supply, question=""):
    import re

    df = supply.copy()
    q = question.lower()

    # Pastikan kolom utama tersedia.
    required_columns = ["year", "region", "produksi_beras"]
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        return (
            f"Kolom yang diperlukan belum tersedia: {missing}. "
            "Periksa struktur supply_result.csv."
        )

    # Konversi tipe data.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["produksi_beras"] = pd.to_numeric(
        df["produksi_beras"], errors="coerce"
    )

    df = df.dropna(subset=["year", "region", "produksi_beras"])

    if df.empty:
        return "Tidak ada data produksi beras yang valid."

    # Filter tahun jika pengguna menyebutkannya.
    years = [int(y) for y in re.findall(r"\b20\d{2}\b", q)]

    if years:
        df = df[df["year"].isin(years)]

        if df.empty:
            return (
                f"Tidak ditemukan data produksi untuk tahun "
                f"{', '.join(map(str, years))}."
            )

    # Ringkasan produksi pada tahun yang dipilih.
    yearly = (
        df.groupby("year", as_index=False)["produksi_beras"]
        .sum()
        .sort_values("year")
    )

    result = ["### Ringkasan Produksi Beras Jawa Timur", ""]

    for _, row in yearly.iterrows():
        result.append(
            f"- **Tahun {int(row['year'])}:** "
            f"{row['produksi_beras']:,.2f} ton"
        )

    # Peringkat wilayah berdasarkan tahun terbaru yang dipilih.
    latest_year = int(df["year"].max())
    latest_df = df[df["year"] == latest_year].copy()

    ranking = (
        latest_df.groupby("region", as_index=False)["produksi_beras"]
        .sum()
        .sort_values("produksi_beras", ascending=False)
    )

    result.extend([
        "",
        f"### Peringkat Produksi Kabupaten/Kota Tahun {latest_year}",
        "",
        ranking.head(5).to_markdown(index=False),
    ])

    # Informasi pertumbuhan jika tersedia.
    if "pertumbuhan_produksi_pct" in latest_df.columns:
        growth = latest_df[
            ["region", "pertumbuhan_produksi_pct"]
        ].copy()

        growth["pertumbuhan_produksi_pct"] = pd.to_numeric(
            growth["pertumbuhan_produksi_pct"],
            errors="coerce",
        )

        growth = growth.dropna(
            subset=["pertumbuhan_produksi_pct"]
        )

        if not growth.empty:
            result.extend([
                "",
                f"### Pertumbuhan Produksi Tahun {latest_year}",
                "",
                growth.sort_values(
                    "pertumbuhan_produksi_pct",
                    ascending=False,
                ).head(5).to_markdown(index=False),
            ])

    result.extend([
        "",
        "**Catatan:** Total provinsi dihitung dengan menjumlahkan "
        "produksi beras seluruh wilayah pada tahun yang sama. "
        "Pastikan setiap wilayah hanya memiliki satu catatan per tahun "
        "agar tidak terjadi penghitungan ganda.",
    ])

    return "\n".join(result)

def handle_distribution(question):

    data = load_project_data()
    df = data["distribution"]

    if df.empty:
        return "Data distribusi belum tersedia."

    q = question.lower()

    # Cari kolom berdasarkan nama yang tersedia.
    columns = {str(c).lower() : c for c in df.columns}

    def find_column(*names):
        for name in names:
            if name in columns:
                return columns[name]
        return None

    year_col = find_column("tahun", "year")
    origin_col = find_column(
        "lokasi asal pasokan", "lokasi_asal_pasokan", "origin", "wilayah_asal"
    )

    destination_col = find_column(
        "lokasi tujuan pengiriman", "lokasi_tujuan_pengiriman", "destination", "wilayah_tujuan"
    )

    volume_col = find_column("volume_ton", "volume_kg", "volume")
    unit_col = find_column("satuan", "unit")

    result = df.copy()

    # Filter tahun bila disebutkan.
    import re
    years = re.findall(r"\b20\d{2}\b", q)

    if years and year_col:
        result = result[
            pd.to_numeric(result[year_col], errors="coerce")
            == int(years[0])
        ]

    # Filter lokasi asal dan tujuan
    if origin_col:
        origins = result[origin_col].dropna().astype(str).unique()
        for origin in sorted(origins, key=len, reverse=True):
            if origin.lower() in q:
                result = result[
                    result[origin_col].astype(str).str.lower()
                    == origin.lower()
                ]
                break

    if destination_col:
        destinations = (
            result[destination_col].dropna().astype(str).unique()
        )

        for destination in sorted(destinations, key=len, reverse=True):
            if destination.lower() in q:
                result = result[
                    result[destination_col].astype(str).str.lower()
                    == destination.lower()
                ]
                break

    if result.empty:
        return(
            "Tidak ditemukan catatan distribusi yang cocok"
            "dengan filter pertanyaan. Ini tidak membuktikan"
            "bahwa rute tersebut tidak pernah terjadi"
        )

    # Tampilkan maksimal 10 catatan aktual yang cocok.
    display_cols = [
        c for c in [
            year_col, origin_col, destination_col, volume_col, unit_col
        ]
        if c is not None
    ]

    if not display_cols:
        return (
            "Data distribusi tersedia, tetapi nama kolomnya"
            "belum dikenali. Perlu pemeriksaan struktur CSV"
        )

    return(
        "### Catatan Distribusi Beras\n\n"
        "Berikut catatan yang cocok dari file distribusi yang diproses:\n\n"
        + result[display_cols].head(10).to_markdown(index=False)
        + "\n\nCatatan: hasil ini adalah catatan pada dataset,"
        "bukan bukti bahwa seluruh distribusi di lapangan telah tercatat."
    )


def handle_recommendation(question):
    data = load_project_data()
    df = data["recommendations"].copy()

    if df.empty:
        return "Data kandidat rekomendasi belum tersedia."

    # Pilih kolom ringkas yang memang tersedia.
    display_columns = [
        "year",
        "wilayah_asal",
        "wilayah_tujuan",
        "status_asal",
        "status_tujuan",
        "dukungan_catatan_rute",
        "status_rekomendasi",
    ]

    available_columns = [
        col for col in display_columns
        if col in df.columns
    ]

    if not available_columns:
        return (
            "Data kandidat tersedia, tetapi struktur kolom "
            "belum sesuai untuk ditampilkan."
        )

    # Batasi jumlah baris agar jawaban mudah dibaca.
    table = df[available_columns].head(10)

    return (
        "### Kandidat Rekomendasi Distribusi\n\n"
        "Berikut maksimal 10 kandidat dari hasil analisis:\n\n"
        + table.to_markdown(index=False)
        + "\n\n**Catatan:** Kandidat ini belum merupakan "
        "keputusan pengiriman yang tervalidasi. "
        "Status produksi relatif tinggi atau rendah tidak "
        "secara otomatis membuktikan surplus atau defisit beras."
    )


def handle_general(question):
    model = load_model()
    data = load_project_data()
    context = build_data_context(data)

    prompt = (
        "Kamu adalah asisten Agri-Food Supply Chain di Jawa Timur.\n"
        "Jawab pertanyaan konseptual dengan jelas.\n"
        "Jangan mengarang angka atau mengklaim analisis yang tidak dilakukan.\n"
        "Bedakan fakta, interpretasi, dan rekomendasi.\n\n"
        f"Konteks proyek:\n{context}\n\n"
        f"Pertanyaan pengguna:\n{question}"
    )

    response = model.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text or "Gemini tidak menghasilkan jawaban"


def handle_monthly_supply(question: str) -> str:
    """
    Menangani pertanyaan produksi bulanan:
    - 2020-2025: mengambil data historis.
    - 2026: mengambil data forecast Seasonal Naive.
    - Di luar periode: memberi informasi bahwa data belum tersedia.
    """
    import re

    q = question.lower()

    month_names = {
        "januari": 1, "februari": 2, "maret": 3,
        "april": 4, "mei": 5, "juni": 6,
        "juli": 7, "agustus": 8, "september": 9,
        "oktober": 10, "november": 11, "desember": 12
    }

    selected_month = next(
        (number for name, number in month_names.items()
         if re.search(rf"\b{name}\b", q)),
        None
    )

    year_match = re.search(r"\b(20\d{2})\b", q)
    requested_year = int(year_match.group(1)) if year_match else None

    if selected_month is None:
        return "Silakan sebutkan bulan yang ingin dianalisis."

    if requested_year is None:
        return "Silakan sebutkan tahun yang ingin dianalisis."

    # --------------------------------------------------
    # A. DATA HISTORIS 2020-2025
    # --------------------------------------------------
    if 2020 <= requested_year <= 2025:

        historical_path = (
            BASE_DIR / "data" /
            "Produksi Beras all 2020-2025.csv"
        )

        if historical_path.exists():
            historical = pd.read_csv(historical_path)
        else:
            data = load_project_data()
            historical = data.get("supply", pd.DataFrame())

        if historical.empty:
            return "Dataset produksi historis belum tersedia."

        required = ["year", "month", "region", "produksi_beras"]

        missing = [
            col for col in required
            if col not in historical.columns
        ]

        if missing:
            return (
                f"Kolom data historis belum lengkap: {missing}. "
                "Periksa dataset produksi bulanan."
            )

        historical["year"] = pd.to_numeric(
            historical["year"], errors="coerce"
        )
        historical["month"] = pd.to_numeric(
            historical["month"], errors="coerce"
        )
        historical["produksi_beras"] = pd.to_numeric(
            historical["produksi_beras"], errors="coerce"
        )

        # Cari wilayah yang disebut pengguna.
        regions = historical["region"].dropna().astype(str).unique()

        matched_region = next(
            (
                region for region in sorted(regions, key=len, reverse=True)
                if region.casefold() in q
            ),
            None
        )

        # Jika wilayah disebut tetapi tidak ada di dataset.
        if matched_region is None:
            region_hint = re.search(
                r"(?:wilayah|kabupaten|kota|daerah)\s+([a-z][a-z-]*)",
                q
            )

            if region_hint:
                requested_region = region_hint.group(1).title()
                available_regions = ", ".join(
                    sorted(regions)[:5]
                )

                return (
                    f"Wilayah **{requested_region}** tidak ditemukan "
                    "dalam dataset proyek. "
                    "Cakupan data produksi saat ini adalah "
                    "kabupaten/kota di Jawa Timur. "
                    f"Contoh wilayah yang tersedia: {available_regions}."
                )

        view = historical[
            (historical["year"] == requested_year) &
            (historical["month"] == selected_month)
        ].copy()

        if matched_region:
            view = view[
                view["region"].str.casefold() ==
                matched_region.casefold()
            ]

        if view.empty:
            return (
                f"Data produksi bulan {selected_month}, "
                f"tahun {requested_year} tidak ditemukan."
            )

        # Tampilkan hasil historis.
        if matched_region:
            row = view.iloc[0]

            return (
                "### Data Historis Produksi Beras\n\n"
                f"- **Wilayah:** {matched_region}\n"
                f"- **Periode:** {selected_month:02d}/{requested_year}\n"
                f"- **Produksi beras:** "
                f"{row['produksi_beras']:,.2f}\n\n"
                "Sumber: dataset produksi beras bulanan historis. "
                "Angka ini merupakan data historis, bukan forecast."
            )

        # Jika tidak ada wilayah spesifik, jumlahkan seluruh wilayah.
        total = view["produksi_beras"].sum()

        return (
            "### Produksi Beras Jawa Timur\n\n"
            f"- **Periode:** {selected_month:02d}/{requested_year}\n"
            f"- **Total produksi:** {total:,.2f}\n"
            f"- **Jumlah wilayah tercatat:** {view['region'].nunique()}\n\n"
            "Sumber: dataset produksi beras bulanan historis."
        )

    # --------------------------------------------------
    # B. FORECAST 2026
    # --------------------------------------------------
    elif requested_year == 2026:

        data = load_project_data()
        forecast = data.get("forecast", pd.DataFrame())

        if forecast.empty:
            return "Dataset forecast 2026 belum tersedia."

        view = forecast.copy()

        regions = (
            view["region_standard"]
            .dropna()
            .astype(str)
            .unique()
        )

        matched_region = next(
            (
                region for region in sorted(regions, key=len, reverse=True)
                if region.casefold() in q
            ),
            None
        )

        if matched_region:
            view = view[
                view["region_standard"].str.casefold() ==
                matched_region.casefold()
            ]
        else:
            region_hint = re.search(
                r"(?:wilayah|kabupaten|kota|daerah)\s+([a-z][a-z-]*)",
                q
            )

            if region_hint:
                requested_region = region_hint.group(1).title()
                return (
                    f"Wilayah **{requested_region}** tidak ditemukan "
                    "dalam dataset forecast Jawa Timur."
                )

        view = view[view["month"] == selected_month]

        if view.empty:
            return "Forecast tidak ditemukan untuk periode tersebut."

        if matched_region:
            row = view.iloc[0]

            return (
                "### Forecast Produksi Beras 2026\n\n"
                f"- **Wilayah:** {matched_region}\n"
                f"- **Periode:** {row['period']}\n"
                f"- **Forecast supply:** "
                f"{row['forecast_supply_ton']:,.2f} ton\n\n"
                "Metode: Seasonal Naive, menggunakan produksi "
                "bulan yang sama pada tahun 2025.\n\n"
                "*Angka ini merupakan prediksi, bukan realisasi aktual.*"
            )

        total = view["forecast_supply_ton"].sum()

        return (
            "### Forecast Produksi Beras Jawa Timur 2026\n\n"
            f"- **Periode:** {selected_month:02d}/2026\n"
            f"- **Forecast supply:** {total:,.2f} ton\n"
            f"- **Jumlah wilayah:** {view['region_standard'].nunique()}\n\n"
            "Metode: Seasonal Naive."
        )

    # --------------------------------------------------
    # C. TAHUN DI LUAR DATASET
    # --------------------------------------------------
    else:
        return (
            f"Data produksi untuk tahun {requested_year} "
            "belum tersedia. Dataset historis mencakup 2020-2025 "
            "dan dataset forecast mencakup tahun 2026."
        )


def handle_monthly_demand(question: str) -> str:
    """Handle monthly estimated demand questions using forecast dataset."""
    data = load_project_data()
    forecast = data.get("forecast", pd.DataFrame())
    if forecast.empty:
        return "Data estimasi demand 2026 belum tersedia."
    view = forecast.copy()
    import re
    month_names = {
        "januari": 1, "februari": 2, "maret": 3, "april": 4,
        "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
        "september": 9, "oktober": 10, "november": 11, "desember": 12,
    }
    q = question.lower()
    selected_month = next((v for k, v in month_names.items() if k in q), None)
    if selected_month:
        view = view[view["month"] == selected_month]
    if view.empty:
        return "Data estimasi demand tidak ditemukan untuk periode tersebut."
    grouped = view.groupby(["month", "period"], as_index=False)["estimated_demand_ton"].sum()
    return (
        "### Estimasi Demand Beras Bulanan 2026\\n\\n"
        + grouped.to_markdown(index=False)
        + "\\n\\nCatatan: demand bulanan merupakan alokasi merata dari estimasi tahunan, bukan pengukuran konsumsi bulanan aktual."
    )


def handle_balance(question: str) -> str:
    """Handle supply-demand balance questions using the 2026 forecast."""
    data = load_project_data()
    forecast = data.get("forecast", pd.DataFrame())
    if forecast.empty:
        return "Data balance forecast 2026 belum tersedia."
    view = forecast.copy()
    q = question.lower()
    regions = view["region_standard"].dropna().astype(str).unique().tolist()
    matched = next((r for r in sorted(regions, key=len, reverse=True) if r.casefold() in q), None)
    if matched:
        view = view[view["region_standard"].str.casefold() == matched.casefold()]
    import re
    month_names = {
        "januari": 1, "februari": 2, "maret": 3, "april": 4,
        "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
        "september": 9, "oktober": 10, "november": 11, "desember": 12,
    }
    selected_month = next((v for k, v in month_names.items() if k in q), None)
    if selected_month:
        view = view[view["month"] == selected_month]
    if view.empty:
        return "Data balance tidak ditemukan untuk filter tersebut."
    if matched or selected_month:
        cols = ["region_standard", "period", "forecast_supply_ton",
                "estimated_demand_ton", "balance_indicator_ton", "balance_status"]
        return "### Balance Indicator\\n\\n" + view[cols].to_markdown(index=False) + (
            "\\n\\nBalance adalah indikator produksi dikurangi estimasi konsumsi, bukan stok aktual."
        )
    summary = view.groupby("period", as_index=False)[
        ["forecast_supply_ton", "estimated_demand_ton", "balance_indicator_ton"]
    ].sum()
    return (
        "### Ringkasan Balance Supply-Demand Jawa Timur 2026\\n\\n"
        + summary.to_markdown(index=False)
        + "\\n\\nBalance bukan stok aktual atau volume yang pasti dapat didistribusikan."
    )


def build_handlers():
    return {
        "supply": handle_supply,
        "monthly_supply": handle_monthly_supply,
        "monthly_demand": handle_monthly_demand,
        "balance": handle_balance,
        "distribution": handle_distribution,
        "recommendation": handle_recommendation,
        "general": handle_general,
    }
