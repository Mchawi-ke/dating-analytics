# Setup
# !pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap
# INTRODUCTION & PROBLEM FRAMING
# ==========================================

intro_text = """
=================================================================================
💘 SPEED DATING ANALYSIS & MATCH PREDICTION FRAMEWORK 💘
=================================================================================

📌 PROJECT OVERVIEW:
   Speed dating provides a unique dataset to analyze interpersonal dynamics, preference 
   alignments, and decision-making drivers. This project aims to perform an end-to-end 
   exploratory data analysis (EDA), feature engineering, time-series analysis, and 
   machine learning modeling to predict match outcomes.

🎯 OBJECTIVES:
   1. 🔍 Understand demographic distributions & preference attributes.
   2. 📊 Explore key factors influencing romantic matches (`match`).
   3. 🛠️ Engineer features reflecting rating gaps, preference mismatches, and overall scores.
   4. 📈 Forecast speed dating event activity and match counts over 5 years.
   5. 🤖 Train ML classification models to predict successful matches.
   6. 💡 Interpret model predictions using SHAP and LIME framework tools.

📁 DATASET DESCRIPTION:
   - File: `speed_dating_master.csv`
   - Target Variable: `match` (Binary: 1 = Both agreed to match, 0 = Otherwise)
   - Features include:
     • Demographics: `male_age`, `female_age`, `age_gap`, `same_race`, `same_field`
     • Ratings Given: Attractiveness, Sincerity, Intelligence, Fun, Ambition
     • Stated Preferences: Attribute weightings (Attractiveness, Intelligence)
     • Self Ratings: Self-perceived attractiveness
     • Activity Habits: Frequency of going out (`male_goes_out`, `female_goes_out`)
     • Decisions: `male_decision`, `female_decision`
=================================================================================
"""

# print(intro_text)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              roc_curve, precision_recall_curve, confusion_matrix,
                              ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve

import xgboost as xgb
import shap

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Professional plotting defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (9, 5.5),
    'figure.dpi': 120,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
sns.set_palette(PALETTE)

print("Environment ready.")




# 1. Data Understanding

df = pd.read_csv('archive(2)/speed_dating_master.csv')
data_dict = pd.read_csv('archive(2)/data_dictionary.csv')

print(f"Shape: {df.shape[0]:,} dates  x  {df.shape[1]} columns")
print(df.head())




summary = pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'n_missing': df.isna().sum(),
    'pct_missing': (df.isna().mean()*100).round(2),
    'n_unique': df.nunique()
})
print(summary)



print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Unique events: {df['event_id'].nunique()}")
print(f"Unique men: {df['male_id'].nunique()}   Unique women: {df['female_id'].nunique()}")
print(f"Dates per event -> min: {df.groupby('event_id').size().min()}, "
      f"median: {int(df.groupby('event_id').size().median())}, "
      f"max: {df.groupby('event_id').size().max()}")

# Does any man/woman appear across more than one event?
spans_multiple_male = (df.groupby('male_id')['event_id'].nunique() > 1).sum()
spans_multiple_female = (df.groupby('female_id')['event_id'].nunique() > 1).sum()
print(f"Men appearing in >1 event: {spans_multiple_male}  |  Women appearing in >1 event: {spans_multiple_female}")




# Data quality observations

#     No missing values, no duplicate rows. This is unusually clean for speed-dating data — the well-known real-world speed-dating studies this kind of dataset is modeled after are notoriously messy (missing surveys, inconsistent scales, dropped waves). That's worth remembering when we get to the Limitations section: results here will look cleaner than a messier real-world version would.
#     Every person appears in exactly one event. This matters enormously for how we split data for modeling (Section 4) — if we split rows randomly, the same person could appear in both the training and validation sets under a different partner, letting the model partially "memorize" that person rather than learn generalizable patterns. Because people never cross events, splitting by event_id is equivalent to splitting by person — a clean, leak-free grouping variable.
#     120 events, ~120 dates each on average, 1,274 unique men and 1,274 unique women. A balanced, fully-crossed design (every attendee of a given gender rotates through every attendee of the other gender at their event).
#     Two columns are pre-registered as leakage by the data dictionary itself: male_decision and female_decision. We'll prove exactly why in Section 3, and treat that as a first-class modeling decision rather than a footnote.


# 2. Exploratory Data Analysis

match_rate = df['match'].mean()

fig, ax = plt.subplots(figsize=(6, 5))
counts = df['match'].value_counts().sort_index()
bars = ax.bar(['No match', 'Match'], counts.values, color=[PALETTE[3], PALETTE[2]], width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 100, f"{v:,}\n({v/len(df):.1%})",
            ha='center', va='bottom', fontsize=11)
ax.set_title(f"Match Rate is Only {match_rate:.1%} — a Meaningfully Imbalanced Target")
ax.set_ylabel("Number of dates")
ax.set_ylim(0, counts.max()*1.2)
plt.tight_layout()
plt.show()