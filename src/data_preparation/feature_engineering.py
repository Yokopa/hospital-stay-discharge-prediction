"""
Utilities for feature engineering, including anemia classification,
kidney function staging, and liver fibrosis risk calculation.
"""

import pandas as pd
import numpy as np

# Function to infer pregnancy and classify anemia
def classify_anemia_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies anemia severity for each patient based on hemoglobin level (Hbn), sex, and inferred pregnancy status.

    If any required columns ('Hbn', 'sex', 'diagnosis') are missing, the 'anemia_level' column will be added with NaN.

    Args:
        df (pd.DataFrame): Input DataFrame with at least columns 'Hbn', 'sex', and 'diagnosis'.

    Returns:
        pd.DataFrame: DataFrame with a new column 'anemia_level' indicating anemia classification.
    """
    df = df.copy()
    required_cols = {'Hbn', 'sex', 'diagnosis'}
    if not required_cols.issubset(df.columns):
        df['anemia_level'] = np.nan
        return df

    df['sex'] = df['sex'].str.lower()

    df['pregnant'] = df.apply(
        lambda row: any(code.strip().startswith('O') for code in str(row['diagnosis']).split(',')) if row['sex'] == 'f' else False,
        axis=1
    )

    def classify_anemia(hb, sex, pregnant):
        if pd.isna(hb):
            return 'Unknown'
        if sex == 'm':
            return 'Severe Anemia' if hb < 8 else 'Moderate Anemia' if hb < 11 else 'Mild Anemia' if hb < 13 else 'Normal'
        if pregnant:
            return 'Severe Anemia' if hb < 7 else 'Moderate Anemia' if hb < 10 else 'Mild Anemia' if hb < 11 else 'Normal'
        return 'Severe Anemia' if hb < 7 else 'Moderate Anemia' if hb < 11 else 'Mild Anemia' if hb < 12 else 'Normal' # non-pregnant females

    df['anemia_level'] = df.apply(lambda row: classify_anemia(row['Hbn'], row['sex'], row['pregnant']), axis=1)
    df.drop(columns='pregnant', inplace=True)
    return df

def classify_kidney_function(df: pd.DataFrame, egfr_col: str = 'EPIGFR') -> pd.DataFrame:
    """
    Classifies kidney function based on eGFR values.

    Categories:
        - 'normal'   : eGFR ≥ 90
        - 'mild'     : 60 ≤ eGFR < 90
        - 'moderate' : 30 ≤ eGFR < 60
        - 'severe'   : eGFR < 30

    If the eGFR column is missing or values are missing/invalid,
    the 'kidney_function' column will be set to 'unknown'.

    Args:
        df (pd.DataFrame): Input DataFrame.
        egfr_col (str): Column name containing eGFR values. Default is 'EPIGFR'.

    Returns:
        pd.DataFrame: DataFrame with a new column 'kidney_function'.
    """
    df = df.copy()

    if egfr_col not in df.columns:
        df['kidney_function'] = 'unknown'
        return df

    conditions = [
        df[egfr_col] >= 90,
        (df[egfr_col] >= 60) & (df[egfr_col] < 90),
        (df[egfr_col] >= 30) & (df[egfr_col] < 60),
        df[egfr_col] < 30
    ]
    categories = ['normal', 'mild', 'moderate', 'severe']

    # Use np.select for known categories; default to 'unknown' for missing/invalid
    df['kidney_function'] = np.select(conditions, categories, default='unknown')

    return df

def compute_apri(df: pd.DataFrame, asat_col: str = 'ASAT', platelet_col: str = 'THZn', ast_uln: float = 40) -> pd.DataFrame:
    """
    Computes APRI (AST to Platelet Ratio Index) score and classifies liver fibrosis risk.

    Risk categories:
        - 'no_fibrosis'   : APRI < 0.5
        - 'moderate_risk' : 0.5 ≤ APRI ≤ 1.5
        - 'high_risk'     : APRI > 1.5

    If required columns are missing, 'APRI' and 'liver_fibrosis_risk' columns will be filled with NaN.

    Args:
        df (pd.DataFrame): Input DataFrame.
        asat_col (str): Column name for AST (ASAT). Default is 'ASAT'.
        platelet_col (str): Column name for platelet count. Default is 'THZn'.
        ast_uln (float): AST upper limit of normal. Default is 40.

    Returns:
        pd.DataFrame: DataFrame with 'APRI' and 'liver_fibrosis_risk' columns added.
    
    Example:
        >>> df = compute_apri(df, asat_col='ASAT', platelet_col='THZn', ast_uln=40)
    """
    df = df.copy()
    if asat_col not in df.columns or platelet_col not in df.columns:
        df['APRI'] = np.nan
        df['liver_fibrosis_risk'] = np.nan
        return df

    df['APRI'] = ((df[asat_col] / ast_uln) / df[platelet_col]) * 100

    conditions = [
        df['APRI'] < 0.5,
        (df['APRI'] >= 0.5) & (df['APRI'] <= 1.5),
        df['APRI'] > 1.5
    ]
    choices = ['no_fibrosis', 'moderate_risk', 'high_risk']
    df['liver_fibrosis_risk'] = np.select(conditions, choices, default='unknown')

    return df


