import pandas as pd
import numpy as np
from scipy import stats

def run_gaussian_copula(df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """
    Synthesizes data using a Gaussian Copula, which preserves 
    inter-column correlations present in the original data.
    
    Steps:
    1. Separate numeric and categorical columns
    2. Transform numeric columns to uniform [0,1] via their empirical CDF
    3. Transform uniforms to standard normal (probit transform)
    4. Fit a multivariate normal to the resulting normal space
    5. Sample from that multivariate normal
    6. Back-transform: normal -> uniform -> original scale (via inverse ECDF)
    7. Re-attach categoricals sampled proportionally
    """
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- Step 1: Build numeric matrix, impute NaNs with median ---
    num_df = df[numeric_cols].copy()
    medians = num_df.median()
    num_df = num_df.fillna(medians)

    # --- Step 2 & 3: ECDF -> Uniform -> Normal (Probit) transform ---
    # This maps each column to a standard normal marginal
    normal_matrix = np.zeros_like(num_df.values, dtype=float)
    
    for i, col in enumerate(numeric_cols):
        col_data = num_df[col].values
        # Empirical CDF: rank / (n+1) to avoid 0 and 1
        ranks = stats.rankdata(col_data)
        uniform = ranks / (len(ranks) + 1)
        # Probit transform: uniform -> standard normal
        normal_matrix[:, i] = stats.norm.ppf(uniform)

    # --- Step 4: Fit multivariate normal (just need the correlation matrix) ---
    corr_matrix = np.corrcoef(normal_matrix.T)
    # Regularize to ensure positive semi-definite (numerical stability)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)
    # Stronger regularization to handle near-singular matrix from small dataset
    corr_matrix += np.eye(len(numeric_cols)) * 0.1

    mean_vec = np.zeros(len(numeric_cols))

    # --- Step 5: Sample from the fitted multivariate normal ---
    sampled_normal = np.random.multivariate_normal(mean_vec, corr_matrix, size=num_samples)

    # --- Step 6: Back-transform: Normal -> Uniform -> Original scale ---
    synthetic_num = np.zeros_like(sampled_normal)
    
    for i, col in enumerate(numeric_cols):
        col_data = num_df[col].values
        col_data_sorted = np.sort(col_data)
        
        # Normal -> Uniform via normal CDF
        uniform_samples = stats.norm.cdf(sampled_normal[:, i])
        # Clip to avoid edge artifacts
        uniform_samples = np.clip(uniform_samples, 0, 1)
        # Uniform -> Original scale via empirical quantile (inverse ECDF)
        quantile_indices = (uniform_samples * (len(col_data_sorted) - 1)).astype(int)
        synthetic_num[:, i] = col_data_sorted[quantile_indices]

    synthetic_num_df = pd.DataFrame(synthetic_num, columns=numeric_cols)

    # --- Step 7: Categoricals sampled proportionally ---
    synthetic_cat_df = pd.DataFrame(index=range(num_samples))
    
    for col in cat_cols:
        valid = df[col].dropna()
        if len(valid) == 0:
            synthetic_cat_df[col] = np.nan
        else:
            # Sample according to observed frequency distribution
            value_counts = valid.value_counts(normalize=True)
            synthetic_cat_df[col] = np.random.choice(
                value_counts.index,
                size=num_samples,
                p=value_counts.values
            )

    # Reassemble in original column order
    result = pd.concat([synthetic_num_df, synthetic_cat_df], axis=1)
    result = result[df.columns]  # restore original column order
    return result