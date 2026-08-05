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



# Takeaway: Only 19.6% of speed dates end in a mutual match. This isn't extreme imbalance, but it's enough that plain accuracy would be a misleading metric — a model that always predicts "no match" is already 80% "accurate" while being completely useless. We'll standardize on ROC-AUC, average precision (PR-AUC), and Brier score (calibration) throughout this notebook instead.


traits = ['attr', 'sinc', 'intel', 'fun', 'amb']
fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)
for ax, t in zip(axes, traits):
    sns.kdeplot(df[f'{t}_of_female'], ax=ax, label='Rated (women)', fill=True, alpha=0.35, color=PALETTE[0])
    sns.kdeplot(df[f'{t}_of_male'], ax=ax, label='Rated (men)', fill=True, alpha=0.35, color=PALETTE[1])
    ax.set_title(t.capitalize())
    ax.set_xlabel('Rating (0-10)')
    ax.set_xlim(0, 10)
axes[0].set_ylabel('Density')
axes[-1].legend(loc='upper left', fontsize=9)
fig.suptitle("How Men Rate Women vs. How Women Rate Men, Across All Five Traits", y=1.05, fontweight='bold')
plt.tight_layout()
plt.show()



# Takeaway: The rating distributions given by men and given by women are nearly identical across all five traits (means cluster tightly around 6.0–6.3 out of 10, roughly bell-shaped, mild left skew). There's no strong asymmetry where, say, men rate women systematically higher/lower than women rate men — the rating scales are being used the same way by both genders, which is reassuring for comparing _of_male and _of_female columns on equal footing later.





numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['event_id'])
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.4, cbar_kws={'shrink': 0.7, 'label': 'Pearson r'}, ax=ax)
ax.set_title("Correlation Matrix — Every Numeric Column", pad=14)
plt.tight_layout()
plt.show()




# Takeaway (read this one carefully): match correlates at r ≈ 0.56–0.57 with both male_decision and female_decision — far higher than with anything else on the board. That's the first visual hint of the leakage trap we formalize in Section 3. Beyond that: the five rating traits (attr/sinc/intel/fun/amb) are all moderately-to-strongly correlated with each other within the same gender (a "halo effect" — people who are rated attractive also tend to get rated as more sincere, fun, etc.), and attr_of_* has the strongest correlation with match among the non-decision columns. same_race, same_field, and age_gap barely register.



fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Match rate by shared_interests decile
df['shared_interests_bin'] = pd.qcut(df['shared_interests'], 8, duplicates='drop')
si_rate = df.groupby('shared_interests_bin', observed=True)['match'].mean()
si_mid = [interval.mid for interval in si_rate.index]
axes[0].plot(si_mid, si_rate.values, marker='o', color=PALETTE[0], linewidth=2)
axes[0].set_title("Match Rate Rises With Shared Interests")
axes[0].set_xlabel("Shared-interests score (binned)")
axes[0].set_ylabel("Match rate")
axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# Match rate by age gap
ag_rate = df.groupby('age_gap')['match'].mean()
ag_n = df.groupby('age_gap').size()
valid = ag_n[ag_n >= 30].index
axes[1].bar(ag_rate.loc[valid].index, ag_rate.loc[valid].values, color=PALETTE[1])
axes[1].set_title("Match Rate by Age Gap (bins with n≥30)")
axes[1].set_xlabel("Age gap (years)")
axes[1].set_ylabel("Match rate")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.tight_layout()
plt.show()




# Takeaway: Shared interests show a clean, fairly monotonic relationship with match rate — moving from the bottom to the top octile roughly doubles the match rate. Age gap shows a mild downward drift (bigger gap, somewhat lower match rate) but it's noisy and much weaker than shared interests or attractiveness. Of the "context" variables (age, race, field), shared interests is the one that actually earns its keep.


# Per-person calibration gap: how you rate yourself vs. how partners rated you, averaged across your dates
male_perceived = df.groupby('male_id')['attr_of_male'].mean()
male_self = df.groupby('male_id')['male_self_attr'].first()
male_gap = (male_self - male_perceived).rename('gap')

female_perceived = df.groupby('female_id')['attr_of_female'].mean()
female_self = df.groupby('female_id')['female_self_attr'].first()
female_gap = (female_self - female_perceived).rename('gap')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(male_perceived, male_self, s=14, alpha=0.35, color=PALETTE[0], label='Men')
axes[0].scatter(female_perceived, female_self, s=14, alpha=0.35, color=PALETTE[1], label='Women')
lims = [2, 10]
axes[0].plot(lims, lims, ls='--', color='gray', linewidth=1.5, label='Perfect calibration')
axes[0].set_xlabel("Attractiveness received (avg. from dates)")
axes[0].set_ylabel("Self-rated attractiveness")
axes[0].set_title("Self-Perception vs. How Others Actually Rate You")
axes[0].legend(fontsize=9)

both_gap = pd.concat([male_gap, female_gap])
axes[1].hist(both_gap, bins=40, color=PALETTE[2], edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5)
axes[1].set_title("Distribution of Self-Perception Gap\n(self-rating minus avg. rating received)")
axes[1].set_xlabel("Gap (positive = overestimates own attractiveness)")
axes[1].set_ylabel("Number of people")

plt.tight_layout()
plt.show()

overest = (both_gap > 0.5).mean()
underest = (both_gap < -0.5).mean()
calibrated = 1 - overest - underest
print(f"Overestimate their attractiveness (gap > 0.5):  {overest:.1%}")
print(f"Underestimate their attractiveness (gap < -0.5): {underest:.1%}")
print(f"Reasonably well-calibrated (within +-0.5):       {calibrated:.1%}")




# Takeaway: At the population level, self-rated attractiveness and received attractiveness are almost perfectly matched on average (~6.1 both ways) — so if you only looked at the means, you'd conclude "people are well calibrated." But at the individual level that's misleading: only about a quarter of people fall within half a point of perfect calibration, while roughly 37% meaningfully overestimate and 37% meaningfully underestimate their own attractiveness. The averages cancel out; the individual miscalibration doesn't. This is a good reminder that population-level statistics can hide substantial individual-level disagreement.


# Stated preference vs. revealed behavior: does saying "attractiveness matters a lot to me"
# actually predict a *stronger* attr -> decision relationship for that person?
tmp = df.copy()
tmp['male_pref_attr_tier'] = pd.qcut(tmp['male_pref_attr'], 3, labels=['Says it matters\nLEAST', 'Middle', 'Says it matters\nMOST'])
tmp['female_pref_attr_tier'] = pd.qcut(tmp['female_pref_attr'], 3, labels=['Says it matters\nLEAST', 'Middle', 'Says it matters\nMOST'])

male_corrs = tmp.groupby('male_pref_attr_tier', observed=True).apply(
    lambda g: g['attr_of_female'].corr(g['male_decision']), include_groups=False)
female_corrs = tmp.groupby('female_pref_attr_tier', observed=True).apply(
    lambda g: g['attr_of_male'].corr(g['female_decision']), include_groups=False)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, male_corrs.values, width, label='Men', color=PALETTE[0])
ax.bar(x + width/2, female_corrs.values, width, label='Women', color=PALETTE[1])
ax.set_xticks(x)
ax.set_xticklabels(male_corrs.index)
ax.set_ylabel("Correlation: partner's attractiveness rating <-> saying yes")
ax.set_title("Stated Importance of Looks vs. How Much Looks Actually Sway the Decision")
ax.legend()
plt.tight_layout()
plt.show()

print("Men   -> correlation by stated-preference tier:", dict(male_corrs.round(3)))
print("Women -> correlation by stated-preference tier:", dict(female_corrs.round(3)))


# Takeaway: There is a real, directionally-consistent relationship — people who claim attractiveness matters more to them do show a stronger attr-to-decision link (roughly 0.12 to 0.25 for men going from the bottom to top tier of stated importance, and similarly 0.14 to 0.28 for women). So stated preferences aren't meaningless. But the effect is far smaller than the roughly 2x spread in stated importance itself would suggest. People's self-reported priorities are a weak, noisy proxy for what actually drives their in-the-moment decisions — a pattern broadly consistent with the stated-vs-revealed-preference literature in mate-choice research, and a caution against taking any single self-report survey column at face value as a "feature that explains behavior."


# 3. The Leakage Trap (Read This Before You Model Anything)



identity_check = (df['match'] == (df['male_decision'] & df['female_decision'])).mean()
print(f"match == (male_decision AND female_decision) for {identity_check:.4%} of rows")

# Drive the point home: a single line of boolean logic "predicts" match perfectly.
naive_prediction = (df['male_decision'] & df['female_decision']).astype(int)
from sklearn.metrics import accuracy_score
print(f"'Model-free' accuracy using just this identity: {accuracy_score(df['match'], naive_prediction):.4%}")




# This is not a modeling result — it's a tautology. match is defined as the logical AND of the two decision columns, so including either of them as a predictor turns "predict whether two people matched" into "look up two columns and multiply them." Any notebook reporting ~100% accuracy on this kind of dataset almost certainly did exactly this.

# Our modeling decision going forward: male_decision and female_decision are excluded from every feature set used to predict match. This is the only way to build a model that's actually learning something about attraction, rather than something about arithmetic. We revisit this pair of columns constructively in Section 9, where we use them as separate, gender-specific targets instead of features — which turns out to be both leak-free and genuinely useful.


# 4. Data Cleaning & Preprocessing

# Decision 	Rationale
# No imputation needed 	Zero missing values anywhere
# No duplicate removal needed 	Zero duplicate rows
# Drop male_decision, female_decision 	Perfect leakage (Section 3)
# Drop male_id, female_id as features 	High-cardinality identifiers with no direct predictive meaning; kept aside for grouped cross-validation only
# Keep event_id only as a CV group, not a feature 	It's an arbitrary index with no causal effect on attraction, but it's the exact key that defines "which rows contaminate each other"
# Ordinal-encode male_goes_out / female_goes_out 	rarely < sometimes < often < very_often is a genuine order — treating it as unordered one-hot would throw away information a tree/linear model could use directly
# No scaling for tree models; not needed for our regularized logistic baseline either 	All numeric features are already on comparable, bounded scales (0–10 ratings, 0–~100 preference points, small integer ages/gaps)


FEATURE_COLS = [
    'male_age', 'female_age', 'age_gap', 'same_race', 'same_field', 'shared_interests',
    'attr_of_female', 'sinc_of_female', 'intel_of_female', 'fun_of_female', 'amb_of_female',
    'attr_of_male', 'sinc_of_male', 'intel_of_male', 'fun_of_male', 'amb_of_male',
    'male_pref_attr', 'male_pref_intel', 'female_pref_attr', 'female_pref_intel',
    'male_self_attr', 'female_self_attr', 'male_goes_out', 'female_goes_out'
]
TARGET = 'match'
GROUP_COL = 'event_id'
LEAKAGE_COLS = ['male_decision', 'female_decision']  # never used as features for `match`

CAT_COLS = ['male_goes_out', 'female_goes_out']
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]
GOES_OUT_ORDER = ['rarely', 'sometimes', 'often', 'very_often']

X = df[FEATURE_COLS].copy()
y = df[TARGET].values
groups = df[GROUP_COL].values

print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns")
print(f"Target balance: {y.mean():.1%} positive")
print(f"CV groups (events): {len(np.unique(groups))}")



# 5. Feature Engineering

# Rather than throwing dozens of speculative interaction terms at the model, we engineer a small number of features with a clear behavioral hypothesis behind each one, and then empirically check whether they actually help (Section 6) — consistent with the principle that a feature earns its place with evidence, not decoration.

#     mutual_attr = average of attr_of_female and attr_of_male. Hypothesis: a date where both people find each other attractive should predict a match better than either person's rating alone, since match requires mutual agreement.
#     pref_weighted_attr_male = male_pref_attr / 100 * attr_of_female. Hypothesis: attractiveness should matter more for men who say it matters more to them (Section 2 showed this effect is real, just weaker than self-reports would suggest — worth letting the model use it directly rather than assuming it's negligible).
#     pref_weighted_attr_female = female_pref_attr / 100 * attr_of_male. Symmetric version for women.
#     rating_spread_of_female / rating_spread_of_male = standard deviation across the five trait ratings a person received. Hypothesis: a partner who is "consistently liked across the board" (low spread, uniformly decent scores) may read differently than one who is "polarizing" (e.g. very attractive but rated low on sincerity) even if the mean is the same.


def engineer_features(X_in: pd.DataFrame) -> pd.DataFrame:
    '''Add behaviorally-motivated engineered features. Pure function - safe to reuse train/val/test.'''
    X_out = X_in.copy()
    X_out['mutual_attr'] = (X_out['attr_of_female'] + X_out['attr_of_male']) / 2
    X_out['pref_weighted_attr_male'] = (X_out['male_pref_attr'] / 100) * X_out['attr_of_female']
    X_out['pref_weighted_attr_female'] = (X_out['female_pref_attr'] / 100) * X_out['attr_of_male']
    X_out['rating_spread_of_female'] = X_out[['attr_of_female', 'sinc_of_female', 'intel_of_female',
                                               'fun_of_female', 'amb_of_female']].std(axis=1)
    X_out['rating_spread_of_male'] = X_out[['attr_of_male', 'sinc_of_male', 'intel_of_male',
                                             'fun_of_male', 'amb_of_male']].std(axis=1)
    return X_out

X_eng = engineer_features(X)
ENGINEERED_COLS = ['mutual_attr', 'pref_weighted_attr_male', 'pref_weighted_attr_female',
                    'rating_spread_of_female', 'rating_spread_of_male']

print("New feature correlations with match:")
print(X_eng[ENGINEERED_COLS].assign(match=y).corr()['match'].sort_values(ascending=False))


 

# Takeaway: mutual_attr (r ≈ 0.15) is a noticeably stronger single predictor than either one-sided attractiveness rating alone (each was r ≈ 0.11 in Section 2's correlation matrix) — confirming the "mutuality" hypothesis empirically, not just intuitively. The preference-weighted features land close to their un-weighted counterparts (consistent with Section 2's finding that stated preferences only weakly modulate behavior), and the rating-spread features show only a weak negative correlation ("polarizing" partners are matched slightly less often). We keep all five for the model and let Section 6 tell us, via ablation, whether they earn their complexity.





# 6. Modeling Strategy

# Task: binary classification, predict match (excluding male_decision/female_decision).

# Validation strategy — the single most important methodological choice in this notebook: We use GroupKFold grouped by event_id (5 folds) rather than a plain random split or StratifiedKFold. Since every person belongs to exactly one event (Section 1), grouping by event is equivalent to grouping by person — it guarantees that no individual's other dates leak into the fold used to evaluate a date involving them. A random row-level split would let the model implicitly learn "this particular person ID tends to get matched a lot" from other rows of the same person sitting in the training fold, inflating validation performance in a way that wouldn't generalize to genuinely new people.

# Models compared, in increasing complexity:

#     Dummy classifier (predicts the prior) — the metric floor.
#     Logistic Regression (class_weight='balanced') — an interpretable linear baseline.
#     Random Forest — a strong non-linear, non-parametric baseline requiring minimal tuning.
#     XGBoost — gradient boosting, typically the strongest tabular-data performer, with scale_pos_weight set to the class-imbalance ratio.

# Metrics (chosen for an imbalanced binary target, per Section 2):

#     ROC-AUC — ranking quality, threshold-independent.
#     Average Precision (PR-AUC) — more informative than ROC-AUC under class imbalance, since it focuses on how well the model ranks the rare positive class.
#     Brier score — calibration quality (lower is better); important if predicted probabilities will be used downstream (e.g., ranking candidate matches by likelihood, as a real dating app would).




preprocessor = ColumnTransformer([
    ('num', 'passthrough', NUM_COLS + ENGINEERED_COLS),
    ('cat', OrdinalEncoder(categories=[GOES_OUT_ORDER, GOES_OUT_ORDER]), CAT_COLS)
])

gkf = GroupKFold(n_splits=5)

def cv_evaluate(model, X_data, y_data, groups_data, name):
    '''Group-aware CV evaluation. Returns a results dict and prints a one-line summary.'''
    aucs, aps, briers = [], [], []
    for train_idx, val_idx in gkf.split(X_data, y_data, groups_data):
        model.fit(X_data.iloc[train_idx], y_data[train_idx])
        proba = model.predict_proba(X_data.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_data[val_idx], proba))
        aps.append(average_precision_score(y_data[val_idx], proba))
        briers.append(brier_score_loss(y_data[val_idx], proba))
    result = {'model': name,
              'roc_auc_mean': np.mean(aucs), 'roc_auc_std': np.std(aucs),
              'pr_auc_mean': np.mean(aps), 'pr_auc_std': np.std(aps),
              'brier_mean': np.mean(briers), 'brier_std': np.std(briers)}
    print(f"{name:22s}  ROC-AUC={result['roc_auc_mean']:.3f}+-{result['roc_auc_std']:.3f}  "
          f"PR-AUC={result['pr_auc_mean']:.3f}+-{result['pr_auc_std']:.3f}  "
          f"Brier={result['brier_mean']:.3f}+-{result['brier_std']:.3f}")
    return result

results = []

dummy_pipe = Pipeline([('pre', preprocessor), ('clf', DummyClassifier(strategy='prior'))])
results.append(cv_evaluate(dummy_pipe, X_eng, y, groups, 'Dummy (prior)'))

logreg_pipe = Pipeline([('pre', preprocessor),
                         ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))])
results.append(cv_evaluate(logreg_pipe, X_eng, y, groups, 'Logistic Regression'))

rf_pipe = Pipeline([('pre', preprocessor),
                     ('clf', RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                                     class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1))])
results.append(cv_evaluate(rf_pipe, X_eng, y, groups, 'Random Forest'))

pos_weight = (y == 0).sum() / (y == 1).sum()
xgb_pipe = Pipeline([('pre', preprocessor),
                      ('clf', xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                                 eval_metric='logloss', scale_pos_weight=pos_weight,
                                                 random_state=RANDOM_STATE, n_jobs=-1))])
results.append(cv_evaluate(xgb_pipe, X_eng, y, groups, 'XGBoost'))



results_df = pd.DataFrame(results).set_index('model')
results_df.round(3)




# Takeaway: Every real model clears the dummy floor by a wide margin (ROC-AUC 0.50 → ~0.70-0.71), confirming there's genuine, learnable signal in attraction ratings and shared interests. But notice what doesn't happen: Random Forest and XGBoost don't meaningfully beat plain Logistic Regression. That's an honest and useful finding, not a disappointing one — it tells us the relationship between these features and match probability is mostly additive and monotonic (more attraction, more shared interest → higher match probability, without strong interaction effects or thresholds that only a tree model could capture). For a real deployment, that's actually good news: a simple, fast, fully interpretable logistic regression is competitive with — and arguably preferable to — a more complex model here.




#  7. Evaluation Deep-Dive

# The table above tells us which model is best on average; this section looks at how the best model (XGBoost) behaves in more detail, using a single held-out group-split for clean, non-overlapping curves.


# One clean, held-out, group-based split for diagnostic plots
unique_events = np.unique(groups)
train_events, val_events = train_test_split(unique_events, test_size=0.25, random_state=RANDOM_STATE)
train_mask = np.isin(groups, train_events)
val_mask = np.isin(groups, val_events)

X_train, X_val = X_eng[train_mask], X_eng[val_mask]
y_train, y_val = y[train_mask], y[val_mask]

final_model = Pipeline([('pre', preprocessor),
                         ('clf', xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                                    eval_metric='logloss', scale_pos_weight=pos_weight,
                                                    random_state=RANDOM_STATE, n_jobs=-1))])
final_model.fit(X_train, y_train)
val_proba = final_model.predict_proba(X_val)[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ROC
fpr, tpr, _ = roc_curve(y_val, val_proba)
axes[0].plot(fpr, tpr, color=PALETTE[0], linewidth=2, label=f"XGBoost (AUC={roc_auc_score(y_val, val_proba):.3f})")
axes[0].plot([0, 1], [0, 1], ls='--', color='gray')
axes[0].set_title("ROC Curve (held-out events)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(fontsize=9)

# Precision-Recall
prec, rec, _ = precision_recall_curve(y_val, val_proba)
axes[1].plot(rec, prec, color=PALETTE[1], linewidth=2,
             label=f"XGBoost (AP={average_precision_score(y_val, val_proba):.3f})")
axes[1].axhline(y_val.mean(), ls='--', color='gray', label=f'No-skill baseline ({y_val.mean():.2f})')
axes[1].set_title("Precision-Recall Curve (held-out events)")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(fontsize=9)

# Calibration
frac_pos, mean_pred = calibration_curve(y_val, val_proba, n_bins=10, strategy='quantile')
axes[2].plot(mean_pred, frac_pos, marker='o', color=PALETTE[2], linewidth=2, label='XGBoost')
axes[2].plot([0, 1], [0, 1], ls='--', color='gray', label='Perfectly calibrated')
axes[2].set_title("Calibration Curve")
axes[2].set_xlabel("Mean predicted probability")
axes[2].set_ylabel("Observed match rate")
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.show()





# Takeaway: The ROC curve confirms solid (not spectacular) ranking ability — an AUC around 0.70 means the model correctly ranks a random matched pair above a random non-matched pair about 70% of the time. The PR curve is the more honest picture given the 80/20 imbalance: precision starts near 60-70% at very low recall and decays as we try to capture more true matches, which is exactly the kind of precision/recall trade-off a real product would need to navigate (e.g., "only surface your top-3 most likely matches" vs. "surface everyone with >10% predicted match probability"). The calibration curve tracks the diagonal reasonably well across most of the probability range, meaning predicted probabilities are broadly trustworthy as probabilities, not just as a ranking score — useful if this model fed a downstream decision (e.g., a threshold-based recommendation).



# Confusion matrix at two candidate thresholds, to make the precision/recall trade-off concrete
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, thresh in zip(axes, [0.5, 0.3]):
    preds = (val_proba >= thresh).astype(int)
    cm = confusion_matrix(y_val, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No match', 'Match'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"Threshold = {thresh}")
plt.tight_layout()
plt.show()



# Takeaway: Lowering the decision threshold from 0.5 to 0.3 trades some precision for a lot more recall — exactly the lever a product team would tune depending on whether the cost of a missed match (false negative) or a wasted introduction (false positive) matters more for their use case




# 8. Hyperparameter Optimization

# We tune XGBoost with RandomizedSearchCV, using the same grouped CV splitter as before (tuning against a leaky, ungrouped CV would silently pick hyperparameters that overfit to person-specific patterns — the exact mistake Section 6 was designed to avoid).


param_distributions = {
    'clf__n_estimators': [100, 200, 300, 400, 500],
    'clf__max_depth': [2, 3, 4, 5, 6],
    'clf__learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
    'clf__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'clf__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'clf__reg_lambda': [0.5, 1, 2, 5, 10],
}

tuning_pipe = Pipeline([('pre', preprocessor),
                         ('clf', xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_weight,
                                                    random_state=RANDOM_STATE, n_jobs=-1))])

search = RandomizedSearchCV(tuning_pipe, param_distributions, n_iter=40, scoring='roc_auc',
                             cv=list(gkf.split(X_eng, y, groups)), random_state=RANDOM_STATE,
                             n_jobs=-1, refit=True, verbose=0)
search.fit(X_eng, y)

print(f"Untuned XGBoost (5-fold grouped CV):  ROC-AUC = {results_df.loc['XGBoost', 'roc_auc_mean']:.4f}")
print(f"Tuned XGBoost   (5-fold grouped CV):  ROC-AUC = {search.best_score_:.4f}")
print(f"Improvement: {search.best_score_ - results_df.loc['XGBoost', 'roc_auc_mean']:+.4f}")
print("\nBest hyperparameters found:")
for k, v in search.best_params_.items():
    print(f"  {k.replace('clf__', ''):18s} = {v}")



# Takeaway (honest, not hyped): Tuning moves ROC-AUC by roughly one hundredth of a point (~0.01) — comparable in size to the noise band we already saw across CV folds (±0.011–0.015). This is consistent with Section 6's finding that the signal here is mostly linear and additive: there's no complex non-linear structure for a more carefully-tuned tree ensemble to unlock. Per the "don't overcomplicate if gains are minimal" principle, we don't recommend shipping the tuned model over the simpler default one — the extra tuning complexity and search cost isn't earning its keep here.




# 9. Interpretability: What Is the Model Actually Learning?

# We use two complementary views: XGBoost's built-in gain-based feature importance (how much each feature improved the loss when it was used to split) and SHAP values (how much each feature pushes an individual prediction away from the average, in probability terms, honoring feature interactions).



fitted_clf = final_model.named_steps['clf']
feature_names = NUM_COLS + ENGINEERED_COLS + CAT_COLS

importances = pd.Series(fitted_clf.feature_importances_, index=feature_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 8))
importances.tail(15).plot(kind='barh', ax=ax, color=PALETTE[0])
ax.set_title("Top 15 Features by XGBoost Gain Importance")
ax.set_xlabel("Relative importance")
plt.tight_layout()
plt.show()





X_val_transformed = preprocessor.transform(X_val)
explainer = shap.TreeExplainer(fitted_clf)
shap_values = explainer.shap_values(X_val_transformed)

shap.summary_plot(shap_values, X_val_transformed, feature_names=feature_names, show=False, max_display=15)
plt.title("SHAP Summary — Direction and Magnitude of Each Feature's Effect", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()




# Takeaway: Both views agree on the headline story: mutual_attr and the raw attr_of_* ratings dominate, followed by shared_interests — confirming, with model-based evidence, exactly what the EDA suggested. The SHAP plot adds direction: high mutual_attr and high shared_interests (red points, high feature value) push predictions toward "match" (positive SHAP value), while high age_gap mildly pushes toward "no match." Demographic and "stated preference" features (male_pref_attr, same_race, same_field) sit near the bottom — the model has independently rediscovered that these barely move the needle, echoing Section 2's correlation analysis. No surprising new driver emerges that wasn't already visible in the EDA — which is itself a meaningful finding: for this dataset, careful EDA would have gotten an analyst 90% of the way to the model's conclusions without touching a single tree.




# 10. Beyond "Match": Two-Stage Decision Modeling

# Here's the reframing promised back in Section 3. Predicting match directly, from features observed during the date, is a slightly artificial task: in real life, neither person can see whether their date said yes before making their own decision. A more faithful — and more useful — framing is:

#     Model each person's decision independently, using only information that belongs to their side of the interaction. Then derive the probability of a mutual match as the product of the two.

# This isn't leakage, because male_decision and female_decision are now targets of separate models, never features of each other's model, and never features of the match model. It's also directly useful for a real product: a dating app could estimate "how likely is this person to say yes to a match like this" from your profile and your candidate's profile, without ever having observed a live decision.

# Feature separation, by construction:

#     Male-decision model uses only: the male's own preferences/self-view, shared context (age gap, same race/field, shared interests), and how he rated her (attr_of_female, etc.) — never her rating of him.
#     Female-decision model is the mirror image.


MALE_DECISION_FEATS = ['age_gap', 'same_race', 'same_field', 'shared_interests',
                        'attr_of_female', 'sinc_of_female', 'intel_of_female', 'fun_of_female', 'amb_of_female',
                        'male_pref_attr', 'male_pref_intel', 'male_self_attr', 'male_goes_out']
FEMALE_DECISION_FEATS = ['age_gap', 'same_race', 'same_field', 'shared_interests',
                          'attr_of_male', 'sinc_of_male', 'intel_of_male', 'fun_of_male', 'amb_of_male',
                          'female_pref_attr', 'female_pref_intel', 'female_self_attr', 'female_goes_out']

def oof_decision_proba(feats, target_col, cat_col):
    '''Out-of-fold predicted probability for one side's decision, using grouped CV (no leakage across events).'''
    y_side = df[target_col].values
    X_side = df[feats]
    pre_side = ColumnTransformer([
        ('num', 'passthrough', [c for c in feats if c != cat_col]),
        ('cat', OrdinalEncoder(categories=[GOES_OUT_ORDER]), [cat_col])
    ])
    pipe_side = Pipeline([('pre', pre_side),
                           ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))])
    oof = np.zeros(len(y_side))
    for train_idx, val_idx in gkf.split(X_side, y_side, groups):
        pipe_side.fit(X_side.iloc[train_idx], y_side[train_idx])
        oof[val_idx] = pipe_side.predict_proba(X_side.iloc[val_idx])[:, 1]
    return oof

p_male_yes = oof_decision_proba(MALE_DECISION_FEATS, 'male_decision', 'male_goes_out')
p_female_yes = oof_decision_proba(FEMALE_DECISION_FEATS, 'female_decision', 'female_goes_out')

print(f"Male-decision model   (out-of-fold): ROC-AUC = {roc_auc_score(df['male_decision'], p_male_yes):.3f}")
print(f"Female-decision model (out-of-fold): ROC-AUC = {roc_auc_score(df['female_decision'], p_female_yes):.3f}")

p_match_two_stage = p_male_yes * p_female_yes

print(f"Two-stage match probability (product of the two): ROC-AUC = {roc_auc_score(df['match'], p_match_two_stage):.3f}")


# Fair, apples-to-apples comparison: the SAME out-of-fold protocol applied to the
# direct single-stage match model from Section 6 (logistic regression, for a like-for-like comparison).
direct_pipe = Pipeline([('pre', preprocessor),
                         ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))])
p_match_direct = np.zeros(len(y))
for train_idx, val_idx in gkf.split(X_eng, y, groups):
    direct_pipe.fit(X_eng.iloc[train_idx], y[train_idx])
    p_match_direct[val_idx] = direct_pipe.predict_proba(X_eng.iloc[val_idx])[:, 1]

comparison = pd.DataFrame({
    'approach': ['Direct single-stage model (Sec. 6 features -> match)',
                 'Two-stage model (decision x decision -> match)'],
    'roc_auc': [roc_auc_score(y, p_match_direct), roc_auc_score(y, p_match_two_stage)],
    'pr_auc': [average_precision_score(y, p_match_direct), average_precision_score(y, p_match_two_stage)],
    'brier': [brier_score_loss(y, p_match_direct), brier_score_loss(y, p_match_two_stage)],
}).set_index('approach')
comparison.round(4)

