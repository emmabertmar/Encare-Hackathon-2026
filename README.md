# Encare Synthetic Data Hackathon

This goal of this hackathon is to create new synthetic data of medical records

## Project Structure
- `/data`: Place your raw `synthetic-data-hackaton-sample.csv` here.
- `/results`: Synthetic outputs will be saved here with timestamps.
- `/examples`: Baseline generators (e.g., Random Sampler).
- `data_processor.py`: Cleaning and imputation logic.
- `validator.py`: Statistical (KS-test) and clinical validation.

## Setup Instructions

### 1. Install Visual Studio Code (VS Code)
Download the installer from code.visualstudio.com.

Run the installer and follow the instructions.

Once open, go to the Extensions view (square icon on the left) and search for "Python" (by Microsoft) and click Install.

### 2. Install Python
### Windows
Download the installer from python.org.

IMPORTANT: Check the box "Add Python to PATH" at the start of the installation. If you miss this, the python command won't work in your terminal.

Verify in PowerShell: python --version

### macOS
macOS comes with an older version of Python. Install the latest version using Homebrew: brew install python or download the .pkg from python.org.

Verify in Terminal: python3 --version (Note: You usually must use python3, not python).

### Linux (Ubuntu/Debian)
Update your package manager: sudo apt update

Install Python: sudo apt install python3 python3-venv python3-pip

Verify: python3 --version

### 3. Create a Virtual Environment
Open your terminal in the project folder and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

Ensure your csv file is in the /data folder and run main.py

The repo also contains some validation for sense checks. The generated data passing these tests shall not be seen as an indicator for a high score.

---

## Approach: KNN Perturbation

Synthetic patients are generated using a SMOTE-inspired K-Nearest Neighbor interpolation approach. Instead of sampling from statistical distributions, it creates new patients by blending pairs of real patients together, ensuring all output stays within clinically plausible ranges.

For each synthetic patient:
1. A real patient is randomly selected as the **anchor**
2. Its **k nearest neighbors** are found in normalized numeric space
3. A random neighbor is picked and the anchor's numeric values are **linearly interpolated** toward it using a random weight
4. **Categorical fields** are sampled proportionally from observed frequencies in the original data
5. **BMI is recomputed** from the interpolated weight and height to guarantee clinical consistency

### Key Design Decisions

**Why KNN over random sampling?**
Random sampling generates values independently per column, ignoring relationships between variables (e.g. weight, height, BMI). KNN interpolation preserves these correlations because synthetic values are always blends of real patients.

**Why impute NaNs before fitting?**
KNN distance calculations require complete numeric data. Missing values are temporarily filled with column medians for fitting, then NaN rates are restored in the output to match the original missingness pattern.

**Why recompute BMI?**
After interpolating weight and height independently, the stored BMI value would be inconsistent. `_fix_bmi` recalculates it from the interpolated values to pass clinical validation.

### Limitations

- With very small datasets (< k patients), neighbors are reused heavily and output diversity is limited
- Categorical columns are sampled independently — correlations between categoricals and numerics are not preserved
- Interpolation between two real patients can still produce edge cases if the two patients are very dissimilar

### Dependencies

- `scikit-learn` — KNN fitting and StandardScaler
- `pandas`, `numpy` — data handling
