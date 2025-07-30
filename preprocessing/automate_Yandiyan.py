import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

RAW_PATH = "../data/raw/Dry_Bean_Dataset.xlsx"
OUT_DIR = "../data/processed/"

def load_raw(path):
    print(">>> Load raw data")
    df = pd.read_excel(path)
    print("Shape:", df.shape)
    return df

def preprocess(df):
    print(">>> Preprocessing data")
    # Hapus duplikat
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"Menghapus {n_dupes} duplikat...")
        df = df.drop_duplicates()
    # Label encoding
    le = LabelEncoder()
    df['Class_enc'] = le.fit_transform(df['Class'])
    # Pisahkan fitur & label
    X = df.drop(['Class', 'Class_enc'], axis=1)
    y = df['Class_enc']
    # Split train-test stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Simpan kolom & objek scaler/encoder
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler, le

def save_processed(X_train, X_test, y_train, y_test, columns, scaler, le):
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(X_train, columns=columns).to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
    pd.DataFrame(X_test, columns=columns).to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False, header=["Class_enc"])
    pd.DataFrame(y_test).to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False, header=["Class_enc"])
    # Simpan scaler & encoder
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))
    print(f"Saved processed data and objects to {OUT_DIR}")

if __name__ == "__main__":
    df = load_raw(RAW_PATH)
    X_train, X_test, y_train, y_test, feature_names, scaler, le = preprocess(df)
    save_processed(X_train, X_test, y_train, y_test, feature_names, scaler, le)
    print(">>> Preprocessing otomatis selesai.")
