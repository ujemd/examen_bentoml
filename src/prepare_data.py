"""Load the raw admission data, clean it and split it into train and test sets.

Outputs (CSV files in data/processed):
    X_train, X_test, y_train, y_test
"""
from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "admission.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# Raw headers are normalized (lowercase, only letters and digits) before being
# mapped, so stray spaces ("LOR ", "Chanceof Admit ") or variants such as
# "Chance of Admit" all end up with the same clean snake_case name.
COLUMN_NAMES = {
    "serialno": "serial_no",
    "grescore": "gre_score",
    "toeflscore": "toefl_score",
    "universityrating": "university_rating",
    "sop": "sop",
    "lor": "lor",
    "cgpa": "cgpa",
    "research": "research",
    "chanceofadmit": "chance_of_admit",
}
TARGET = "chance_of_admit"

def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())

def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    unknown = [c for c in df.columns if normalize(c) not in COLUMN_NAMES]
    if unknown:
        raise ValueError(f"Unexpected columns in the raw file: {unknown}")
    df = df.rename(columns={c: COLUMN_NAMES[normalize(c)] for c in df.columns})

    # The serial number is only a row identifier and carries no information.
    df = df.drop(columns=["serial_no"])

    print("Rows before cleaning:", len(df))
    df = df.dropna().drop_duplicates()
    print("Rows after cleaning: ", len(df))
    return df


def main() -> None:
    df = load_and_clean(RAW_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Features:", list(X.columns))
    print("Saved to", PROCESSED_DIR)


if __name__ == "__main__":
    main()