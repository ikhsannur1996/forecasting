# Forecasting Kebutuhan Uang Tunai ATM: End-to-End

Proyek forecasting perbankan dari notebook sampai API: meramal **penarikan tunai harian per ATM untuk 14 hari ke depan**, lengkap dengan **interval prediksi** dan **rekomendasi jumlah pengisian kas**. Tujuannya menyeimbangkan dua risiko:
- isi terlalu sedikit → ATM **kehabisan uang** (*cash-out*),
- isi terlalu banyak → uang **menganggur** dan menanggung biaya dana.

Dua model dibandingkan, yaitu **Holt-Winters (ETS)** per ATM dan **Gradient Boosting global**, dengan seasonal naive sebagai patokan. Model terbaik disimpan sebagai **JSON**, disajikan lewat **FastAPI**, dan dijalankan di **Docker lokal**.

**Highlight:**
- **Model statistik** sebelum backtesting: efek gajian, Lebaran, akhir pekan, libur dalam % + p-value per tipe lokasi (regresi panel Driscoll-Kraay) dan SARIMAX
- 20 ATM di 6 tipe lokasi (mall, kantor, perumahan, transport hub, kampus, pasar), harian Jan 2022 – Des 2025
- Pola nyata Indonesia: mingguan per lokasi, **tanggal gajian**, **Lebaran** (THR, mudik, arus balik), libur nasional, tren turun karena pembayaran digital
- Fitur kalender dengan **binning** (jendela gajian, fase Lebaran) → **one-hot**, plus lag & rolling. Satu fungsi fitur dipakai bersama oleh training, backtest, dan API
- **Rolling-origin cross-validation**: tuning di 2023, pemilihan model di 2024, **uji akhir 2025 dikunci** (26 origin × 14 hari × 20 ATM)
- Statistik: ADF/KPSS, Kruskal-Wallis, **Diebold-Mariano (Newey-West + HLN)**, **cluster bootstrap CI**, Ljung-Box, uji cakupan interval
- **Simulasi bisnis** pengisian kas + analisis sensitivitas biaya
- API dapat menerima **data aktual terbaru** (`recent_actuals`) tanpa retrain

## Hasil (uji akhir 2025)

| Metode | WAPE | MASE | Bias |
|---|---|---|---|
| **Gradient Boosting global** ✅ terpilih | **11,6%** (CI 11,3–11,9%) | **0,53** | −0,15% |
| Holt-Winters (ETS) per ATM | 19,9% | 0,92 | −1,0% |
| Seasonal naive (minggu lalu) | 22,1% | 1,01 | −1,0% |

- Error berkurang **48%** dibanding naive dan **42%** dibanding Holt-Winters. Diebold-Mariano p < 10⁻¹⁵, dan GBR lebih baik di 84% lintasan ATM × origin.
- **Lebaran**: WAPE GBR 12% (H-7…H-1) dan 22% (H0…H+3), sedangkan ETS 29% dan **52%**. ETS tidak tahu kapan Lebaran.
- Stabil sepanjang 2025 (WAPE per origin 9,9–14,7%, tanpa tren memburuk), dan tidak ada segmen dengan bias material (> 2%).
- Interval: cakupan P10–P90 **82%** (target 80%), P05–P95 **92%** (target 90%). Sedikit konservatif, dan aman untuk kas.
- **Pengisian kas**: kebijakan berbasis ramalan menurunkan hari cash-out dari **19%** (manual “minggu lalu × 1,3”) ke **0,6%**, dengan total biaya **86% lebih rendah**. Penghematannya 64–92% di semua skenario penalti yang diuji.
- Ringkasan evaluasi: **10 ✅ · 3 ⚠️ · 0 ❌**. ⚠️ = interval sedikit lebar (2 baris) dan autokorelasi residual kecil di 5 ATM.

## Struktur Proyek

```
forecasting/
├── data/atm_withdrawals.csv, atm_master.csv   # dibuat oleh scripts/generate_data.py
├── scripts/generate_data.py                   # generator: pola per lokasi, gajian, Lebaran, libur, tren, noise AR(1)
├── notebooks/atm_cash_forecasting_end_to_end.ipynb
├── src/
│   ├── calendar_id.py      # libur nasional, Lebaran, binning gajian & fase Lebaran (pandas saja)
│   ├── features.py         # lag, rolling, one-hot, forecast rekursif (pandas saja; dipakai juga oleh API)
│   ├── models.py           # SeasonalNaive, ETSForecaster (statsmodels), GBRForecaster (scikit-learn)
│   ├── backtest.py         # rolling-origin backtest paralel + skala MASE
│   ├── evaluation.py       # WAPE/MASE, Diebold-Mariano, cluster bootstrap, interval, Ljung-Box, bias segmen
│   └── json_model.py       # ekspor JSON + inferensi (pohon GBR / rumus Holt-Winters) tanpa sklearn/statsmodels
├── models/model.json       # 600 pohon + riwayat 35 hari + kuantil interval + kuantil pengisian + evaluasi
├── app/main.py, schemas.py # FastAPI
├── tests/                  # test_api.py, test_evaluation.py (19 test)
├── sample_request.json, sample_request_with_actuals.json
├── requirements.txt        # API (tanpa scikit-learn/statsmodels)
├── requirements-dev.txt
├── Dockerfile, docker-compose.yml, .dockerignore
```

---

## 1. Data & Notebook

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/generate_data.py --seed 42      # opsional, data sudah tersedia
jupyter lab notebooks/atm_cash_forecasting_end_to_end.ipynb
```

Jalankan semua cell (*Run All*, ±8 menit di laptop 10 core). Bagian terberat adalah rolling-origin backtest, yang melatih ulang model di 60 titik waktu. Hasilnya menimpa `models/model.json` dan file `sample_*.json`.

Isi notebook (19 bagian; setiap cell kode didahului kotak **Alur data: Input → Proses → Output → Berikutnya**):

| Bagian | Isi |
|---|---|
| 0–3 | Peta alur, import, load & validasi data, EDA (tren, pola mingguan per lokasi, gajian, *event study* Lebaran, STL) |
| 4 | **Uji statistik time series**: ADF & KPSS, tren (Newey-West), Kruskal-Wallis + ε², Mann-Whitney, ACF/PACF |
| 5 | **Desain validasi rolling-origin** (tuning 2023 / seleksi 2024 / uji akhir 2025) |
| 6 | Fitur: lag & rolling, **binning** gajian & Lebaran → **one-hot**; **uji kebocoran otomatis** |
| 7 | 3 metode |
| **7.1** | **Model statistik (sebelum backtesting)**: regresi kalender **per tipe lokasi** dengan efek tetap ATM: **intercept, koefisien, efek %, p-value (Holm)**, CI, **GVIF**, SE **Driscoll-Kraay** (tahan autokorelasi & korelasi antar ATM); **SARIMAX** + kalender untuk satu ATM (koefisien AR, AIC, Ljung-Box) |
| 8 | Tuning di 2023 |
| 9 | Pemilihan model di 2024 + Diebold-Mariano & Wilcoxon |
| 10 | Kalibrasi interval prediksi per horizon (residual 2024) |
| 11 | **Backtest akhir 2025**: metrik + cluster bootstrap CI, FVA, DM, stabilitas, cakupan interval, bias per segmen, performa per fase Lebaran |
| 12 | Diagnostik residual 1 langkah (Ljung-Box per ATM) |
| 13 | **Simulasi pengisian kas** + kurva biaya + sensitivitas |
| 14 | Permutation importance per kelompok fitur |
| 15 | **Ringkasan evaluasi** & kesimpulan |
| 16–18 | Refit semua data → `model.json` + ramalan Jan 2026, validasi JSON, alur data di API |

## 2. Menjalankan API Lokal (tanpa Docker)

```bash
uvicorn app.main:app --reload --port 8003
```

Buka **http://localhost:8003/docs**. Untuk test:

```bash
pytest -q
```

## 3. Deploy ke Docker Lokal

### Prasyarat
- Docker Desktop **berjalan** (cek: `docker info`)
- `models/model.json` sudah ada

### Opsi A: Docker CLI

```bash
docker build -t atm-cash-forecast-api:latest .
docker run -d --name atm-cash-forecast-api -p 8003:8000 atm-cash-forecast-api:latest
docker ps                                   # tunggu STATUS = healthy
```

### Opsi B: Docker Compose

```bash
docker compose up -d --build
docker compose logs -f api
```

> Port host **8003**. Keempat API proyek bisa berjalan bersamaan: 8000 credit default, 8001 segmentasi, 8002 valuasi properti, 8003 forecasting ATM.

### Uji API

```bash
curl http://localhost:8003/health
curl http://localhost:8003/atms
curl http://localhost:8003/model-info

# ramalan 14 hari untuk 2 ATM
curl -X POST http://localhost:8003/forecast -H "Content-Type: application/json" -d @sample_request.json

# kirim data aktual terbaru (semua ATM, berurutan setelah tanggal data terakhir model) → ramalan dari tanggal terbaru
curl -X POST http://localhost:8003/forecast -H "Content-Type: application/json" -d @sample_request_with_actuals.json
```

Contoh response (dipotong):

```json
{
  "model_name": "Gradient Boosting",
  "data_until": "2025-12-31",
  "horizon": 3,
  "load_quantile": 0.995,
  "warnings": ["Data terakhir model 2025-12-31 (281 hari lalu). Kirim recent_actuals atau refit model."],
  "atms": [{
    "atm_id": "ATM09", "location_type": "residential", "cassette_capacity": 350.0,
    "total_forecast": 438.36, "total_p10": 360.36, "total_p90": 524.23, "total_recommended_load": 634.84,
    "days": [{
      "date": "2026-01-01", "h": 1, "forecast": 139.53, "p10": 116.71, "p90": 165.82, "p05": 110.22, "p95": 175.43,
      "recommended_load": 199.09, "capped_by_capacity": false, "is_holiday": true,
      "payday_window": "gajian_1-2", "lebaran_phase": "normal"
    }]
  }]
}
```

| Field | Arti |
|---|---|
| `forecast` | Ramalan median penarikan hari itu (juta Rp) |
| `p10`–`p90`, `p05`–`p95` | Interval 80% & 90% |
| `recommended_load` | Rekomendasi isi kas = kuantil optimal (0,995) dari distribusi ramalan, dibatasi kapasitas kaset |
| `capped_by_capacity` | `true` jika rekomendasi melebihi kapasitas. Pertimbangkan pengisian lebih sering |
| `payday_window`, `lebaran_phase`, `is_holiday` | Konteks kalender (hasil binning yang dipakai model) |
| `warnings` | Horizon > 14 hari, kalender kedaluwarsa, data model sudah lama, batas kapasitas |

### Stop & Bersihkan

```bash
docker stop atm-cash-forecast-api && docker rm atm-cash-forecast-api   # Opsi A
docker compose down                                                     # Opsi B
```

---

## Referensi API

| Method | Endpoint | Keterangan |
|---|---|---|
| GET | `/health` | Status & versi model |
| GET | `/model-info` | Info model, interval per horizon, kuantil pengisian, ringkasan evaluasi |
| GET | `/atms` | Daftar ATM, tipe lokasi, kapasitas kaset |
| POST | `/forecast` | `{atm_ids?, horizon 1–28 (default 14), recent_actuals?, include_daily?}` |
| GET | `/docs` | Swagger UI |

**Validasi (HTTP 422):**
- ATM harus dikenal, dan horizon 1–28.
- `recent_actuals` harus berisi **semua ATM** untuk setiap tanggal mulai hari setelah data terakhir model, berurutan tanpa lubang, dengan nilai ≥ 0.
- Field yang tidak dikenal ditolak.

## Operasional yang Disarankan

| Kegiatan | Frekuensi | Catatan |
|---|---|---|
| Kirim data aktual lewat `recent_actuals` | harian | lag selalu terbaru tanpa retrain |
| Refit model (jalankan notebook) | bulanan | menyerap tren & pola terbaru |
| Rekalibrasi interval & cek backtest | tahunan | cakupan 2025 sedikit konservatif (82%/92%) |
| Perbarui `src/calendar_id.py` | tahunan | tanggal Lebaran, Idul Adha, Imlek, cuti bersama; API memberi peringatan jika lewat batas |
| Ganti asumsi biaya (biaya simpan 8%/th, penalti Rp 2 jt) | sekali | pakai angka internal bank, lalu jalankan ulang Bagian 13 |

## Troubleshooting

| Masalah | Solusi |
|---|---|
| Peringatan “Data terakhir model … hari lalu” | Kirim `recent_actuals` atau refit model |
| 422 pada `recent_actuals` | Pastikan semua 20 ATM ada dan tanggal berurutan mulai `data_until + 1` |
| Banyak `capped_by_capacity` | Kapasitas kaset tidak cukup untuk siklus itu → tambah frekuensi pengisian |
| Port bentrok | `-p 8004:8000` |
| Notebook lama | Kurangi jumlah origin di Bagian 5 atau `n_estimators` di Bagian 7 untuk eksperimen |

> ⚠️ Data bersifat sintetis untuk pembelajaran. Di produksi gunakan data transaksi ATM riil, termasuk hari ATM offline dan data pengisian aktual. Hari ATM offline harus ditandai dan dikeluarkan dari target, karena penarikan nol saat ATM rusak bukan permintaan nol.
