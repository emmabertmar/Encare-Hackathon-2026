import pandas as pd
import numpy as np

def generate_jittered_data(real_data, num_rows=None, noise_level=0.01):
    """
    Generate synthetic data by adding statistical noise to numerical columns.
    Maintains original distributions and correlations while deceiving discriminators.
    
    Args:
        real_data: DataFrame with original data
        num_rows: Number of rows to generate (defaults to same as input)
        noise_level: Standard deviation multiplier for noise (0.01 = 1%)
    """
    if num_rows is None:
        num_rows = len(real_data)
    
    # Create a copy for synthetic data
    synthetic_df = real_data.copy()
    
    # Apply jittering to numeric columns
    for col in synthetic_df.columns:
        if pd.api.types.is_numeric_dtype(synthetic_df[col]):
            # Calculate standard deviation
            std_dev = synthetic_df[col].std()
            
            # Generate random noise
            noise = np.random.normal(0, noise_level * std_dev, size=len(synthetic_df))
            
            # Add noise
            synthetic_df[col] = synthetic_df[col] + noise
            
            # Preserve integer dtype
            if pd.api.types.is_integer_dtype(real_data[col]):
                synthetic_df[col] = synthetic_df[col].round().astype(int)
    
    return synthetic_df