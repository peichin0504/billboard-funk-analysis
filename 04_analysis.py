"""
04_analysis.py
Difference-in-Differences and event study analysis.
Tests whether funk-like songs experienced a relative decline in danceability
after the 2015 Blurred Lines copyright verdict.
Requires: classified_data.csv
Output: event_study.png, regression_table.tex
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.iolib.summary2 import summary_col

# Load data
df = pd.read_csv("classified_data.csv")

# Create post-2015 indicator (Blurred Lines verdict: March 2015)
df["post"] = (df["year"] >= 2015).astype(int)

# ── Difference-in-Differences Regression ──────────────────────────────────────
# Outcome: danceability
# Treatment: is_funk (funk-like songs identified via clustering)
# Key coefficient: is_funk:post — the DiD estimate
model = smf.ols(
    "danceability ~ is_funk * post + C(year)",
    data=df
).fit(cov_type="HC3")  # heteroskedasticity-robust standard errors

print(model.summary())

# ── Export Regression Table to LaTeX ──────────────────────────────────────────
table = summary_col(
    [model],
    stars=True,
    float_format="%.3f",
    model_names=["Danceability"],
    info_dict={
        "N": lambda x: f"{int(x.nobs)}",
        "R²": lambda x: f"{x.rsquared:.3f}"
    }
)

latex_str = table.as_latex()
with open("regression_table.tex", "w") as f:
    f.write(latex_str)
print("Saved: regression_table.tex")

# ── Event Study ────────────────────────────────────────────────────────────────
# Fully interacted model: is_funk × year dummies
# Baseline year: 2010 (omitted)
event_model = smf.ols(
    "danceability ~ is_funk * C(year)",
    data=df
).fit(cov_type="HC3")

# Extract interaction coefficients for each year
coeffs = []
for year in range(2010, 2021):
    if year == 2010:
        # Baseline year: coefficient normalized to 0
        coeffs.append({"year": year, "coef": 0, "ci_low": 0, "ci_high": 0})
        continue
    key = f"is_funk:C(year)[T.{year}]"
    if key in event_model.params:
        coef = event_model.params[key]
        ci = event_model.conf_int().loc[key]
        coeffs.append({"year": year, "coef": coef, "ci_low": ci[0], "ci_high": ci[1]})
    else:
        coeffs.append({"year": year, "coef": np.nan, "ci_low": np.nan, "ci_high": np.nan})

coeff_df = pd.DataFrame(coeffs)

# Plot event study
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(coeff_df["year"], coeff_df["coef"], marker="o", color="steelblue", label="Funk vs Control")
ax.fill_between(coeff_df["year"], coeff_df["ci_low"], coeff_df["ci_high"], alpha=0.2, color="steelblue")
ax.axvline(x=2015, color="red", linestyle="--", label="Blurred Lines Verdict (2015)")
ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
ax.set_xlabel("Year")
ax.set_ylabel("Differential Danceability (Funk vs Control)")
ax.set_title("Event Study: Effect of Blurred Lines Verdict on Funk Song Danceability")
ax.legend()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.tight_layout()
plt.savefig("event_study.png", dpi=150)
plt.show()
print("Saved: event_study.png")
