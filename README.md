# Eksperimen SMSML - Dry Bean Classification (Yandiyan)

## Deskripsi Proyek
Repositori ini berisi eksperimen, EDA, dan otomatisasi preprocessing pada **Dry Bean Dataset** (UCI ML Repository) sebagai bagian dari submission kelas MLOps Dicoding.

- **Dataset:** [Dry Bean Dataset – UCI ML Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Sumber:** Koklu, M., & Ozkan, I. A. (2020).  
- **Task:** Klasifikasi multikelas 7 varietas kacang kering menggunakan data geometri morfologi biji.

## Struktur Direktori
Eksperimen_SML_Yandiyan/
├── data/
│ ├── raw/ # File dataset asli (.xlsx)
│ └── processed/ # Hasil preprocessing otomatis (csv, pkl)
├── preprocessing/
│ ├── Eksperimen_Yandiyan.ipynb # Notebook EDA dan baseline modeling
│ └── automate_Yandiyan.py # Script otomatisasi preprocessing
├── .github/
│ └── workflows/
│ └── preprocess.yml # (Opsional, workflow otomatis via Actions)
├── .gitignore
├── README.md


## Cara Menjalankan Preprocessing Otomatis

1. Pastikan dependencies terinstall:
    ```
    pip install pandas scikit-learn openpyxl joblib
    ```
2. Tempatkan file dataset `Dry_Bean_Dataset.xlsx` di folder `data/raw/`.
3. Jalankan:
    ```
    python preprocessing/automate_Yandiyan.py
    ```
4. Hasil preprocessing otomatis akan tersimpan di folder `data/processed/`:
    - X_train.csv, X_test.csv
    - y_train.csv, y_test.csv
    - scaler.pkl, label_encoder.pkl

## Notebook Eksplorasi

- Notebook eksperimen (`Eksperimen_Yandiyan.ipynb`) berisi:
  - Data loading, EDA lengkap (statistik, visualisasi)
  - Preprocessing manual
  - Baseline model Random Forest
  - Insight data dan evaluasi awal model

## Workflow Otomatisasi (Advanced)

> (Opsional untuk nilai advanced)  
File `.github/workflows/preprocess.yml` menjalankan script preprocessing otomatis setiap ada push ke repo.  
Contoh workflow bisa dilihat pada file tersebut.

## Credit Template

- Struktur notebook & pipeline mengacu pada template submission Dicoding SMSML.
- Kode & insight dikembangkan sendiri (no plagiarism).

---

## **4. (Optional Advanced) GitHub Actions Workflow Otomatis Preprocessing**

**Buat file** `.github/workflows/preprocess.yml`:

```yaml
name: Preprocess Data (Otomatis)

on:
  push:
    paths:
      - 'preprocessing/**'
      - 'data/raw/**'
      - '.github/workflows/preprocess.yml'
  workflow_dispatch:

jobs:
  preprocess:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pandas scikit-learn openpyxl joblib

    - name: Run preprocessing automation
      run: python preprocessing/automate_Yandiyan.py

    - name: Upload processed files as workflow artifact
      uses: actions/upload-artifact@v4
      with:
        name: processed_data
        path: data/processed/
