"""
LLM Archetype Expansion — Synthetic Patient Generator

Strategy:
1. Summarize the real data schema and key statistics
2. Ask Claude to generate N clinically coherent patient archetypes as JSON
3. Expand each archetype into many rows using constrained statistical perturbation
4. Recombine with real categorical distributions for columns Claude doesn't cover

Requires: pip install anthropic
Set ANTHROPIC_API_KEY in your .env file.
"""

import json
import os
import re
import pandas as pd
import numpy as np

# Columns Claude will fill; the rest are filled statistically from the real data
LLM_TARGET_COLS = [
    "Age::40",
    "Gender::5",
    "BMI::24",
    "Preoperative body weight (kg)::20",
    "Height (cm)::23",
    "Diabetes Mellitus::11",
    "Smoker::9",
    "Alcohol overconsumption::10",
    "Severe heart disease::12",
    "Severe Pulmonary Disease::13",
    "ASA physical status class::78",
    "Preoperative WHO performance score::14",
    "Surgical approach::62",
    "Main procedure name::56",
    "Length of stay (nights in hospital after primary operation)::179",
    "Complications at all during primary stay::183",
    "Grading of most severe complication::186",
    "Survival status::229",
    "Final diagnosis::221",
]

NUM_ARCHETYPES = 40  # Claude will generate this many diverse patients


def run_llm_archetype(df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """
    Generates synthetic patients by:
    1. Asking Claude to create clinically realistic patient archetypes.
    2. Expanding each archetype into (num_samples / NUM_ARCHETYPES) variants.
    3. Filling remaining columns from the real data distribution.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    print(f"  [LLM] Requesting {NUM_ARCHETYPES} patient archetypes from Claude...")
    archetypes = _generate_archetypes(client, df)
    print(f"  [LLM] Received {len(archetypes)} archetypes. Expanding to {num_samples} rows...")

    expanded = _expand_archetypes(archetypes, df, num_samples)
    print(f"  [LLM] Done. Synthetic dataset shape: {expanded.shape}")
    return expanded


# ---------------------------------------------------------------------------
# Step 1: Claude generates archetypes
# ---------------------------------------------------------------------------

def _build_schema_summary(df: pd.DataFrame) -> str:
    """Build a compact schema description for the prompt."""
    lines = []
    for col in LLM_TARGET_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            lines.append(
                f'  "{col}": numeric, range [{series.min():.1f}, {series.max():.1f}], '
                f'mean {series.mean():.1f}'
            )
        else:
            top = series.value_counts().head(6).index.tolist()
            lines.append(f'  "{col}": categorical, values: {top}')
    return "\n".join(lines)


def _generate_archetypes(client, df: pd.DataFrame) -> list[dict]:
    schema = _build_schema_summary(df)

    prompt = f"""You are a clinical data expert. Generate exactly {NUM_ARCHETYPES} realistic and \
diverse synthetic surgical patient records for an ERAS (Enhanced Recovery After Surgery) dataset.

Each patient must be clinically coherent — comorbidities, age, procedure, complications, and \
survival should all be consistent with real surgical patients. Include diversity in age, sex, \
procedure type, and risk profile.

For EACH patient, output a JSON object with ONLY these fields:
{schema}

Rules:
- BMI must equal weight_kg / (height_cm/100)^2, rounded to 2 decimals
- Age must be between 18 and 100
- ASA class correlates with comorbidities (sicker patients = higher ASA)
- Complications and length of stay should be clinically plausible
- Survival status must be one of: "Yes, alive", "Yes, dead", "Unknown"
- Use exactly the category strings shown above (copy them precisely)

Respond with a valid JSON array of {NUM_ARCHETYPES} patient objects. Nothing else."""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text

    # Extract JSON array from the response (robust to markdown fences)
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"Claude did not return a JSON array. Response:\n{raw[:500]}")

    archetypes = json.loads(json_match.group(0))
    return archetypes


# ---------------------------------------------------------------------------
# Step 2: Expand archetypes statistically
# ---------------------------------------------------------------------------

def _expand_archetypes(archetypes: list[dict], df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """
    For each archetype, generate (num_samples / len(archetypes)) perturbed variants.
    Non-LLM columns are sampled from the real data distribution.
    """
    rows_per_archetype = max(1, num_samples // len(archetypes))
    remainder = num_samples - rows_per_archetype * len(archetypes)

    all_rows = []
    for i, archetype in enumerate(archetypes):
        n = rows_per_archetype + (1 if i < remainder else 0)
        variants = _perturb_archetype(archetype, n)
        all_rows.extend(variants)

    llm_df = pd.DataFrame(all_rows)

    # Fill the remaining columns from real data distribution
    full_df = _fill_remaining_columns(llm_df, df, num_samples)

    # Enforce BMI consistency
    full_df = _fix_bmi(full_df)

    # Restore original column order
    return full_df[[c for c in df.columns if c in full_df.columns]]


def _perturb_archetype(archetype: dict, n: int) -> list[dict]:
    """Add small realistic noise to numeric fields of an archetype."""
    rows = []
    for _ in range(n):
        row = {}
        for key, val in archetype.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                # Add ~3% Gaussian noise, but keep it clinically realistic
                noise_scale = abs(val) * 0.03 + 0.5
                row[key] = round(val + np.random.normal(0, noise_scale), 2)
            else:
                row[key] = val
        rows.append(row)
    return rows


def _fill_remaining_columns(llm_df: pd.DataFrame, real_df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """Sample all non-LLM columns from the real data distribution."""
    result = llm_df.copy()
    remaining_cols = [c for c in real_df.columns if c not in llm_df.columns]

    for col in remaining_cols:
        series = real_df[col].dropna()
        if len(series) == 0:
            result[col] = np.nan
            continue

        if pd.api.types.is_numeric_dtype(series):
            # Sample from empirical distribution via quantile interpolation
            quantiles = np.random.uniform(0, 1, num_samples)
            result[col] = np.quantile(series.values, quantiles)
        else:
            counts = series.value_counts(normalize=True)
            result[col] = np.random.choice(counts.index, size=num_samples, p=counts.values)

        # Re-introduce NaN at the same rate as real data
        nan_rate = real_df[col].isna().mean()
        if nan_rate > 0:
            mask = np.random.random(num_samples) < nan_rate
            result.loc[mask, col] = np.nan

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
