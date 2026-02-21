import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(data_path: str):
    """
    Load and preprocess the heart disease dataset.
    Returns X_train, X_test, y_train, y_test, scaler
    """
    df = pd.read_csv(data_path)

    # Drop rows with missing values if any
    df = df.dropna()

    # Features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Binary target: 0 = no disease, 1 = disease
    y = (y > 0).astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        list(df.drop("target", axis=1).columns),
    )
