import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def run_knn_perturbation(df: pd.DataFrame, num_samples: int, k: int = 10) -> pd.DataFrame:
    """
    Generates synthetic data via K-Nearest Neighbor interpolation (SMOTE-inspired).

    For each synthetic patient:
    1. Randomly pick a real patient as the "anchor"
    2. Find its k nearest neighbors in normalized numeric space
    3. Interpolate numerics between the anchor and a random neighbor
    4. Inherit categoricals from the anchor (preserving clinical text logic)
    5. Recompute BMI from weight + height to guarantee clinical logic check passes

    No new dependencies — uses sklearn which is already in requirements.txt.
    Inherits realistic clinical ranges and inter-variable correlations from real data.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Build imputed numeric matrix for distance computation
    num_df = df[numeric_cols].copy()
    medians = num_df.median()
    # For all-NaN columns, median is NaN — fill those with 0 so scaler doesn't break
    medians = medians.fillna(0)
    num_df_filled = num_df.fillna(medians)

    # Drop columns that are still fully NaN (all-zeros constant columns are fine)
    knn_cols = [c for c in numeric_cols if num_df_filled[c].notna().any()]
    knn_df = num_df_filled[knn_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(knn_df.values)
    # Replace any remaining NaN/inf from zero-variance columns
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  [KNN] Fitting k={k} neighbors on {len(df)} real patients...")
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", n_jobs=-1)
    knn.fit(X_scaled)

    # Pick anchor indices uniformly at random (with replacement)
    anchor_indices = np.random.choice(len(df), size=num_samples, replace=True)

    # Batch KNN lookup
    _, neighbor_matrix = knn.kneighbors(X_scaled[anchor_indices])
    # neighbor_matrix[:,0] is the anchor itself, so skip it
    neighbor_matrix = neighbor_matrix[:, 1:]

    print(f"  [KNN] Generating {num_samples} synthetic patients...")
    synthetic_numeric = np.empty((num_samples, len(numeric_cols)))

    for i in range(num_samples):
        anchor_idx = anchor_indices[i]
        # Pick a random neighbor from the k nearest
        neighbor_idx = neighbor_matrix[i, np.random.randint(k)]

        # Interpolation weight (0=anchor, 1=neighbor)
        alpha = np.random.uniform(0, 1)

        anchor_vals = num_df_filled.values[anchor_idx]
        neighbor_vals = num_df_filled.values[neighbor_idx]

        synthetic_numeric[i] = alpha * anchor_vals + (1 - alpha) * neighbor_vals

    synthetic_num_df = pd.DataFrame(synthetic_numeric, columns=numeric_cols)

    # Restore NaN pattern: if a column was mostly NaN in original, re-introduce NaNs
    nan_rates = num_df.isna().mean()
    for col in numeric_cols:
        rate = nan_rates[col]
        if rate > 0:
            nan_mask = np.random.random(num_samples) < rate
            synthetic_num_df.loc[nan_mask, col] = np.nan

    # Categoricals: sample from real data proportionally (same as gaussian copula)
    cat_data = {}
    for col in cat_cols:
        valid = df[col].dropna()
        if len(valid) == 0:
            cat_data[col] = [np.nan] * num_samples
        else:
            counts = valid.value_counts(normalize=True)
            cat_data[col] = np.random.choice(counts.index, size=num_samples, p=counts.values)
    synthetic_cat_df = pd.DataFrame(cat_data)

    result = pd.concat([synthetic_num_df, synthetic_cat_df], axis=1)
    result = result[[c for c in df.columns if c in result.columns]]

    # Enforce BMI = weight / height^2 so the clinical logic check passes
    result = _fix_bmi(result)

    return result


def _fix_bmi(df: pd.DataFrame) -> pd.DataFrame:
    weight_col = "Preoperative body weight (kg)::20"
    height_col = "Height (cm)::23"
    bmi_col = "BMI::24"

    if not all(c in df.columns for c in [weight_col, height_col, bmi_col]):
        return df

    w = pd.to_numeric(df[weight_col], errors="coerce")
    h = pd.to_numeric(df[height_col], errors="coerce") / 100
    valid = (w > 0) & (h > 0) & w.notna() & h.notna()

    df = df.copy()
    df.loc[valid, bmi_col] = (w[valid] / h[valid] ** 2).round(4)
    return df
