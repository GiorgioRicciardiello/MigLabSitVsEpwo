"""
The dataset contains subjects that answered the questionnaire at multiple time points. This script will take
those duplicates and run statistical analysis into those

"""
import pandas as pd
from typing import Union, Optional, Tuple, List
import numpy as np
from config.config import config, mapper_diagnosis
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
import seaborn as sns
import json
from questionnaires.questionnaires import SituationalSleepinessScale, EpworthScale
from matplotlib.gridspec import GridSpec
from sklearn.metrics import cohen_kappa_score
from utils.functions import pca_interpretation, cronbach_alpha
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
sns.set_context('poster')
#  'paper', 'notebook', 'talk', and 'poster'

def calculate_test_retest_reliability(df: pd.DataFrame, cols: List[str]) -> float:
    df['date'] = pd.to_datetime(df['date'])
    multi_response_ids = df['study_id'].value_counts()[df['study_id'].value_counts() > 1].index
    df_multi = df[df['study_id'].isin(multi_response_ids)]
    df_multi = df_multi.sort_values(by=['study_id', 'date'])
    correlations = []
    for study_id in multi_response_ids:
        df_study = df_multi[df_multi['study_id'] == study_id]
        if len(df_study) > 1:
            scores1 = df_study[cols].iloc[0].values
            scores2 = df_study[cols].iloc[1].values
            correlation = np.corrcoef(scores1, scores2)[0, 1]
            correlations.append(correlation)
    return np.mean(correlations)


def calculate_within_subject_variability(df: pd.DataFrame, cols: List[str]) -> float:
    df['date'] = pd.to_datetime(df['date'])
    multi_response_ids = df['study_id'].value_counts()[df['study_id'].value_counts() > 1].index
    df_multi = df[df['study_id'].isin(multi_response_ids)]
    df_multi = df_multi.sort_values(by=['study_id', 'date'])
    variabilities = []
    for study_id in multi_response_ids:
        df_study = df_multi[df_multi['study_id'] == study_id]
        if len(df_study) > 1:
            scores = df_study[cols].values
            variability = np.std(scores, axis=0, ddof=1).mean()
            variabilities.append(variability)
    return np.mean(variabilities)


def perform_mixed_effects_model(df: pd.DataFrame, cols: List[str], scale_name: str) -> None:
    df_long = df.melt(id_vars=['study_id', 'date'], value_vars=cols, var_name='item', value_name='score')
    df_long['time'] = df_long.groupby('study_id').cumcount()

    # Fit mixed-effects model
    md = mixedlm("score ~ time", df_long, groups=df_long["study_id"], re_formula="~time")
    mdf = md.fit()

    print(f"Mixed-Effects Model for {scale_name}:\n", mdf.summary())


if __name__ == '__main__':
    df_data = pd.read_csv(config.get('data_pp_path'))
    col_sss = [col for col in df_data.columns if 'sss' in col and ' ' not in col and len(col) < 10]

    col_ess = [col for col in df_data.columns if 'ess' in col and len(col) < 10]
    lbls = {
        'ess': 'Epworth Sleepiness Scale',
        'sss': 'Situational Sleepiness Scale'
    }
    # get unique IDs, keeping the last record of duplicates based on date
    df_duplicates = df_data[df_data.duplicated('study_id', keep=False)]
    df_duplicates.sort_values(by=['study_id', 'date'], ascending=[True, False], inplace=True)
    df_data = df_data.drop_duplicates(subset='study_id', keep='first')
    df_data.reset_index(drop=True, inplace=True)
    questions_score = col_ess + col_sss
    cmp_heatmap = sns.color_palette("Spectral_r", as_cmap=True)
    # call the classes
    ess_quest = EpworthScale()
    sit_quest = SituationalSleepinessScale()

    # Get the labels
    ess_labels = ess_quest.get_labels()
    sss_labels = sit_quest.get_labels()
    # %%  reliability test
    # compare the duplicate responses are different time point - repeated measure anova

    # test forte
    test_retest_sss = calculate_test_retest_reliability(df_duplicates, list(sss_labels.keys()))
    test_retest_ess = calculate_test_retest_reliability(df_duplicates, list(ess_labels.keys()))

    within_subject_variability_sss = calculate_within_subject_variability(df_duplicates, list(sss_labels.keys()))
    within_subject_variability_ess = calculate_within_subject_variability(df_duplicates, list(ess_labels.keys()))

    print(f"Test-Retest Reliability for SSS: {test_retest_sss}")
    print(f"Test-Retest Reliability for ESS: {test_retest_ess}")
    print(f"Within-Subject Variability for SSS: {within_subject_variability_sss}")
    print(f"Within-Subject Variability for ESS: {within_subject_variability_ess}")

    perform_mixed_effects_model(df_duplicates, list(sss_labels.keys()), 'Situational Sleepiness Scale')
    perform_mixed_effects_model(df_duplicates, list(ess_labels.keys()), 'Epworth Sleepiness Scale')

    test_retest_sss = calculate_test_retest_reliability(df=df_duplicates, cols=[*ess_labels.keys()])

    test_retest_ess = calculate_test_retest_reliability(df=df_duplicates, cols=list(ess_labels.keys()))

    # Extract the data for SSS and ESS
    sss_data = df_data[sss_labels].dropna()
    ess_data = df_data[ess_labels].dropna()

    # Calculate Cronbach's alpha for SSS and ESS
    alpha_sss = cronbach_alpha(sss_data.T)
    alpha_ess = cronbach_alpha(ess_data.T)

    print(f"Cronbach's alpha for SSS: {alpha_sss}")
    print(f"Cronbach's alpha for ESS: {alpha_ess}")
