import os
import pandas as pd
import kagglehub
import joblib
from sklearn.preprocessing import StandardScaler

def main(out_dir="namadataset_preprocessing"):
    path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")
    csv_path = os.path.join(path, "heart_cleveland_upload.csv")
    df = pd.read_csv(csv_path)

    # target: condition -> binary target
    df["target"] = (df["condition"] > 0).astype(int)
    df = df.drop(columns=["condition"])

    df2 = df.copy()

    for col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    for col in df2.columns:
        if df2[col].isna().sum() > 0:
            df2[col] = df2[col].fillna(df2[col].median())

    df2 = df2.drop_duplicates()

    feature_cols = [c for c in df2.columns if c != "target"]
    for c in feature_cols:
        q1 = df2[c].quantile(0.25)
        q3 = df2[c].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df2[c] = df2[c].clip(lower, upper)

    X = df2[feature_cols].astype(float)
    y = df2["target"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_processed = pd.DataFrame(X_scaled, columns=feature_cols)
    df_processed["target"] = y.values

    os.makedirs(out_dir, exist_ok=True)
    df_processed.to_csv(os.path.join(out_dir, "heart_disease_preprocessing.csv"), index=False)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    print("✅ Preprocessing selesai. Output ada di:", out_dir)

if __name__ == "__main__":
    main()
