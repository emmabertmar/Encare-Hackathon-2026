import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load the ERAS dataset."""
    df = pd.read_csv(filepath, sep=',', low_memory=False)
    return df

def preprocess_for_synthesis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for jittering synthesis.
    Minimal processing to preserve original distributions.
    """
    df_subset = df.copy()
    return df_subset