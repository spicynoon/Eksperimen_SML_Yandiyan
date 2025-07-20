# automate_Yandiyan.py

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def run(input_path="../loan_prediction_raw/train.csv", output_path="../preprocessing/loan_prediction_preprocessing/loan_cleaned.csv"):
    # Load dataset
    df = pd.read_csv(input_path)

    # Drop kolom tidak relevan
    df.drop('Loan_ID', axis=1, inplace=True)

    # Ubah '3+' di kolom Dependents → 3
    df['Dependents'] = df['Dependents'].replace('3+', '3')

    # Imputasi missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    # Encoding fitur kategorikal
    label_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    # Convert Dependents ke integer
    df['Dependents'] = df['Dependents'].astype(int)

    # Simpan dataset hasil preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset berhasil disimpan ke: {output_path}")

# Jalankan fungsi utama jika file ini dieksekusi langsung
if __name__ == "__main__":
    run()
