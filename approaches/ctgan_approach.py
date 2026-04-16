import pandas as pd
import numpy as np

# Columns that CTGAN can't handle well: dates, times, and high-cardinality text.
# These are excluded from training and re-attached from the real data distribution.
_EXCLUDE_PATTERNS = ("YYYY-MM-DD", "HH:mm")


def _split_columns(df: pd.DataFrame):
    """Return (cols_for_ctgan, cols_to_resample) based on column name patterns."""
    exclude = [
        c for c in df.columns
        if any(pat in c for pat in _EXCLUDE_PATTERNS)
    ]
    include = [c for c in df.columns if c not in exclude]
    return include, exclude


def _resample_excluded(excluded_cols, real_df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """Sample excluded columns from the empirical distribution of the real data."""
    data = {}
    for col in excluded_cols:
        valid = real_df[col].dropna()
        nan_rate = real_df[col].isna().mean()
        if len(valid) == 0:
            data[col] = [np.nan] * num_samples
        else:
            counts = valid.value_counts(normalize=True)
            sampled = np.random.choice(counts.index, size=num_samples, p=counts.values)
            # Re-introduce NaN at the real rate
            if nan_rate > 0:
                mask = np.random.random(num_samples) < nan_rate
                sampled = sampled.astype(object)
                sampled[mask] = np.nan
            data[col] = sampled
    return pd.DataFrame(data)


def run_ctgan(df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """
    Generates synthetic data using a Conditional Tabular GAN (CTGAN).

    Date/time columns are excluded from CTGAN (too many unique values) and
    re-attached afterward by sampling from the real empirical distribution.
    BMI is recalculated from weight + height to satisfy the clinical logic check.

    Requires: pip install sdv
    """
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
    except ImportError:
        raise ImportError("SDV not installed. Run: pip install sdv")

    ctgan_cols, excluded_cols = _split_columns(df)
    df_train = df[ctgan_cols].copy()

    print(f"  [CTGAN] Training on {len(ctgan_cols)} columns "
          f"({len(excluded_cols)} date/time cols excluded and resampled separately)...")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=150,
        batch_size=500,
        verbose=True,
    )
    synthesizer.fit(df_train)

    print(f"  [CTGAN] Sampling {num_samples} rows...")
    synthetic_df = synthesizer.sample(num_rows=num_samples)

    # Re-attach excluded columns sampled from real distribution
    if excluded_cols:
        resampled = _resample_excluded(excluded_cols, df, num_samples)
        synthetic_df = pd.concat(
            [synthetic_df.reset_index(drop=True), resampled.reset_index(drop=True)],
            axis=1,
        )

    # Restore original column order
    synthetic_df = synthetic_df[[c for c in df.columns if c in synthetic_df.columns]]

    # Fix BMI = weight / height^2 so clinical logic check passes
    synthetic_df = _fix_bmi(synthetic_df)

    return synthetic_df


def _fix_bmi(df: pd.DataFrame) -> pd.DataFrame:
    weight_col = "Preoperative body weight (kg)::20"
    height_col = "Height (cm)::23"
    bmi_col = "BMI::24"

    if not all(c in df.columns for c in [weight_col, height_col, bmi_col]):
        return df

    w = pd.to_numeric(df[weight_col], errors="coerce")
    h = pd.to_numeric(df[height_col], errors="coerce") / 100  # cm -> m
    valid = (w > 0) & (h > 0) & w.notna() & h.notna()

    df = df.copy()
    df.loc[valid, bmi_col] = (w[valid] / h[valid] ** 2).round(4)
    return df
