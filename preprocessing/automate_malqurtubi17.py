import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
from imblearn.over_sampling import SMOTE

def preprocess_data(input_path, target_column, save_path, file_path, output_path):
    # Baca data
    df = pd.read_csv(input_path)
    print(f"Data berhasil dibaca dari: {input_path}")
    
    # Kolom numerik dan kategorikal
    num_col = [
        'BMI', 'GenHlth', 'MentHlth', 'PhysHlth',
        'Age', 'Education', 'Income'
    ]

    cat_col = [
        'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
        'DiffWalk', 'Sex'
    ]

    # Simpan nama kolom (tanpa target) ke file CSV
    df.drop(columns=[target_column]).head(0).to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    # Hapus duplikat & baris dengan nilai NaN pada target
    df = df.drop_duplicates()
    df = df.dropna(subset=[target_column])  # Pastikan target tidak mengandung NaN
    print(f"Data setelah menghapus baris dengan NaN di kolom target: {df.shape[0]} baris.")

    # Outlier removal (IQR)
    mask = pd.Series(True, index=df.index)
    for feature in num_col:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask &= (df[feature] >= lower) & (df[feature] <= upper)
    df = df[mask].reset_index(drop=True)

    # Pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_col),
        ('cat', cat_pipeline, cat_col)
    ])

    # Pisahkan fitur dan target
    X = df[num_col + cat_col]
    y = df[target_column]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Cek apakah ada nilai NaN setelah SMOTE pada kolom target
    if y_resampled.isnull().sum() > 0:
        print(f"Ada {y_resampled.isnull().sum()} nilai NaN setelah SMOTE pada target column.")
    else:
        print("Tidak ada NaN pada kolom target setelah SMOTE.")

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )

    # Fit & transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Simpan pipeline
    dump(preprocessor, save_path)

    # Simpan dataset yang sudah diproses
    processed_data = pd.DataFrame(X_train_processed, columns=num_col + cat_col)
    processed_data[target_column] = y_train.reset_index(drop=True)
    processed_data.to_csv(output_path, index=False)

    print("Preprocessing selesai dan data telah disimpan.")

    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(
        input_path="diabetes_raw.csv",
        target_column="Diabetes_binary",
        save_path="preprocessing/preprocessor_pipeline.joblib",
        file_path="preprocessing/data_header.csv",
        output_path="preprocessing/diabetes_preprocessing.csv"
    )