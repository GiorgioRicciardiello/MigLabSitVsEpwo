"""
Following a similar approach of the study
https://pubmed.ncbi.nlm.nih.gov/16171277/

Factor Analysis and Reliability Assessment of Sleepiness Measures Overview This code performs a factor analysis on
self-reported sleepiness measures, identifies the underlying factors, and assesses the internal consistency (
reliability) of these factors using Cronbach's alpha. The analysis aims to determine whether different facets of
subjective daytime sleepiness exist and how these facets are related to various demographic and sleep-related variables.

Factor Analysis:
1. Standardization: Standardize the sleepiness measures to ensure they have a mean of 0 and a standard deviation of 1.
2. Factor Analysis: Perform principal-axis factor analysis with varimax rotation to identify underlying factors in the
    sleepiness measures. This helps to reduce the dimensionality of the data and interpret the latent structure.
3. actor Loadings: Create a DataFrame to store the loadings of each item on the identified factors. Use a threshold to
    determine significant loadings.

Reliability Analysis (Cronbach's Alpha)::

1. Cronbach's Alpha Calculation: Define a function to calculate Cronbach's alpha, a measure of internal consistency.
    The formula takes into account the variance of each item and the total variance of the sum of items.
2. Identify Loading Items: Determine which items load significantly on each factor based on the factor loadings.
3. Calculate Cronbach's Alpha for Each Factor: Apply the Cronbach's alpha function to the items loading on each factor
    to assess the reliability of the identified factors.
"""

import pathlib
from config.config import config, mapper
import pandas as pd
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import json
import matplotlib.pyplot as plt

sns.set_context('talk')

def cronbach_alpha(df):
    items = df.shape[1]
    total_var = df.sum(axis=1).var(ddof=1)
    sum_var = df.var(axis=0, ddof=1).sum()
    alpha = (items / (items - 1)) * (1 - sum_var / total_var)
    return alpha

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
    # %% Similar analysis to https://pubmed.ncbi.nlm.nih.gov/16171277/
    print(df_data.describe())

    # Factor Analysis
    sleepiness_measures = df_data.filter(regex='^ess|sss')
    col_div = [div for div in sleepiness_measures.columns if '_div_' in div]
    sleepiness_measures.drop(columns=col_div,
                             inplace=True)

    scaler = StandardScaler()
    scaled_sleepiness_measures = scaler.fit_transform(sleepiness_measures)

    fa = FactorAnalysis(n_components=3, rotation='varimax')
    factors = fa.fit_transform(scaled_sleepiness_measures)

    # Create a DataFrame for factors
    factor_df = pd.DataFrame(factors, columns=['Factor1', 'Factor2', 'Factor3'])

    # Identify which items load on each factor
    loadings = pd.DataFrame(fa.components_.T, columns=['Factor1', 'Factor2', 'Factor3'],
                            index=sleepiness_measures.columns)


    loadings.to_csv(config.get('results_path').joinpath('FA_Loadings.csv'), index=True)
    # Threshold to consider an item as loading on a factor
    threshold = 0.3

    # Items loading on Factor 1
    factor1_items = loadings[abs(loadings['Factor1']) > threshold].index.tolist()
    # Items loading on Factor 2
    factor2_items = loadings[abs(loadings['Factor2']) > threshold].index.tolist()
    # Items loading on Factor 3
    factor3_items = loadings[abs(loadings['Factor3']) > threshold].index.tolist()

    # Reliability analysis (Cronbach's alpha)
    # Identify the items that load on each factor and calculate Cronbach's alpha for these sets of items.
    factor1_alpha = cronbach_alpha(sleepiness_measures[factor1_items])
    factor2_alpha = cronbach_alpha(sleepiness_measures[factor2_items])
    factor3_alpha = cronbach_alpha(sleepiness_measures[factor3_items])

    print(f"Cronbach's alpha for Factor 1: {factor1_alpha:.2f}")
    print(f"Cronbach's alpha for Factor 2: {factor2_alpha:.2f}")
    print(f"Cronbach's alpha for Factor 3: {factor3_alpha:.2f}")

    cronbach_loadings = {
        'factor1_alpha': factor1_alpha,
        'factor2_alpha': factor2_alpha,
        'factor3_alpha': factor3_alpha,
    }

    path = config.get('results_path').joinpath('FA_cronbach_projections.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cronbach_loadings, f, ensure_ascii=False, indent=4)

    # Add factor scores to the original DataFrame
    df = pd.concat([df_data, factor_df], axis=1)

    # Regression analysis
    # Model Factor 1
    model1 = smf.ols(
        'Factor1 ~ age + gender + bmi + race + ethnicity',
        data=df).fit()
    print(model1.summary())

    # Model Factor 2
    model2 = smf.ols(
        'Factor2 ~ age + gender + bmi + race + ethnicity',
        data=df).fit()
    print(model2.summary())

    # Model Factor 3
    model3 = smf.ols(
        'Factor3 ~ age + gender + bmi + race + ethnicity',
        data=df).fit()
    print(model3.summary())

    with open("FA_fa1_ols_summary.txt", "w") as f:
        f.write(model1.summary().as_text())

    with open("FA_fa2_ols_summary.txt", "w") as f:
        f.write(model2.summary().as_text())

    with open("FA_fa3_ols_summary.txt", "w") as f:
        f.write(model3.summary().as_text())
    # %% Plots
    plt.figure(figsize=(12, 10))  # You can adjust the size as needed
    cbar_kws = {'label': 'Correlation coefficient', 'ticks': 10}
    cmp_heatmap = sns.color_palette("Spectral_r", as_cmap=True)
    sns.heatmap(loadings,
                annot=True,
                fmt=".2f",
                cmap=cmp_heatmap,
                cbar_kws={'label': 'Correlation coefficient'})
    plt.title('Factorial Analysis Loadings - Varimax')
    plt.xlabel('Factors')
    plt.ylabel('Variables')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.get('results_path').joinpath(f'FA_Heatmap_Loadings.png'), dpi=300)
    plt.show()
