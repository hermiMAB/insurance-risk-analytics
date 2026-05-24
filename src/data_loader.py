import os
import pandas as pd
def load_data():
    project_root = os.path.abspath('../')
    txt_path = os.path.join(project_root, 'data', 'MachineLearningRating_v3.txt')

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")

    df = pd.read_csv(
        txt_path,
        sep="|",
        encoding="utf-8",
        na_values=["", " ", "  ", "   ", "None", "NULL", "NaN", "nan", "?"],
        on_bad_lines="warn"
    )

    # normalize empty strings
    df = df.replace(r'^\s*$', pd.NA, regex=True)

    return df