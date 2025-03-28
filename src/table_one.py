import pathlib
import pandas as pd
import statsmodels.api as sm
from typing import Union, Optional, Tuple, List, Dict
import numpy as np
from config.config import config, mapper_diagnosis
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
import seaborn as sns
import json
from questionnaires.questionnaires import SituationalSleepinessScale, EpworthScale
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score, roc_curve
import math
from scipy.stats import mannwhitneyu, mode
from scipy.stats import wilcoxon


class QuestionnaireComparator:
    def __init__(self, df: pd.DataFrame,
                 questionnaire_map: Dict[str, str],
                 alias_dict: Dict[str, str],
                 output_path: pathlib.Path = None):
        self.df = df.copy()
        self.map = questionnaire_map
        self.alias = alias_dict
        self.output_path = output_path

    def _get_mode(self, series:pd.Series) -> float:
        try:
            return series.mode()[0]
        except:
            return np.nan

    def compare(self):
        rows = []

        for sss_col, ess_col in self.map.items():
            alias_name = self.alias.get(sss_col, sss_col)
            is_score = 'score' in sss_col.lower()

            if sss_col not in self.df.columns or ess_col not in self.df.columns:
                continue

            sss_data = self.df[sss_col].dropna().astype(int)
            ess_data = self.df[ess_col].dropna().astype(int)

            # Only use paired data (drop rows with any missing)
            paired_data = self.df[[sss_col, ess_col]].dropna().astype(int)
            sss_vals = paired_data[sss_col]
            ess_vals = paired_data[ess_col]

            if len(paired_data) < 5:
                p_value = np.nan  # Not enough data
            else:
                try:
                    if is_score:
                        # Continuous score, non-parametric
                        p_value = mannwhitneyu(sss_vals, ess_vals, alternative='two-sided').pvalue
                    else:
                        # Ordinal, Wilcoxon is best for paired, else fallback to Mann-Whitney
                        try:
                            p_value = wilcoxon(sss_vals, ess_vals).pvalue
                        except:
                            p_value = mannwhitneyu(sss_vals, ess_vals, alternative='two-sided').pvalue
                except:
                    p_value = np.nan

            if is_score:
                sss_stat = f"{sss_data.mean():.2f} ± {sss_data.std():.2f}"
                ess_stat = f"{ess_data.mean():.2f} ± {ess_data.std():.2f}"
            else:
                sss_stat = f"Median: {sss_data.median()}, Mode: {self._get_mode(sss_data)}"
                ess_stat = f"Median: {ess_data.median()}, Mode: {self._get_mode(ess_data)}"

            rows.append({
                # "Item": alias_name,
                "SSS": sss_col,
                "ESS": ess_col,
                "SSS_stats": sss_stat,
                "ESS_stats": ess_stat,
                "p-value_formatted": f"{p_value:.4f}" if not pd.isna(p_value) else "N/A",
                "p-value": p_value

            })

        return pd.DataFrame(rows)


class CrossQuestionPermutation:
    def __init__(self, df, sss_cols, ess_cols):
        self.df = df.copy()
        self.sss_cols = sss_cols
        self.ess_cols = ess_cols

    def compare_all(self):
        results = []

        for sss_col in self.sss_cols:
            for ess_col in self.ess_cols:
                if sss_col not in self.df.columns or ess_col not in self.df.columns:
                    continue

                data = self.df[[sss_col, ess_col]].dropna()
                sss_vals = data[sss_col]
                ess_vals = data[ess_col]

                if len(data) < 5:
                    p_value = np.nan
                else:
                    try:
                        # Use Wilcoxon if data is paired and not constant, otherwise Mann-Whitney
                        if sss_vals.equals(ess_vals):
                            p_value = 1.0
                        else:
                            try:
                                p_value = wilcoxon(sss_vals, ess_vals).pvalue
                            except:
                                p_value = mannwhitneyu(sss_vals, ess_vals, alternative='two-sided').pvalue
                    except:
                        p_value = np.nan

                results.append({
                    "SSS Question": sss_col,
                    "ESS Question": ess_col,
                    # "p-value": f"{p_value:.4f}" if not pd.isna(p_value) else "N/A",
                    "p-value": p_value,
                })

        return pd.DataFrame(results)


if __name__ == '__main__':
    df_data = pd.read_csv(config.get('data_pp_path').get('pp_data'))

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
    output_path = config.get('results_path').joinpath('first_order_stats')
    if not output_path.exists():
        output_path.mkdir(exist_ok=True, parents=True)

    # %% define the questionnaire classes
    ess_quest = EpworthScale()
    sit_quest = SituationalSleepinessScale()

    # Get the labels
    ess_labels = ess_quest.get_labels()
    sss_labels = sit_quest.get_labels()

    alias = ess_labels.copy()
    alias.update(sss_labels)

    alias_no_enter = {key:val.replace('\n', ' ') for key, val in alias.items()}
    alias_no_enter_both = {key:f'{val}|{key}' for key, val in alias_no_enter.items()}

    questionnaire_map = {
        'sss1': 'ess1',  # General unintentional sleep likelihood ~ Sitting and reading
        'sss2': 'ess2',  # On computer/tablet ~ Watching TV
        'sss3': 'ess5',  # Talking on phone ~ Talking to someone
        'sss4': 'ess6',  # Meeting ~ Sitting quietly after lunch
        'sss5': 'ess3',  # Listening to speaker ~ Sitting in a meeting, lecture, or theater
        'sss6': 'ess8',  # Board game ~ In a car, while stopped in traffic
        'sss7': 'ess7',  # Driving ~ Driving in traffic
        'sss8': 'ess4',  # Playing video game ~ Lying down to rest in the afternoon
        'sss9': 'ess4',  # Lying down ~ Lying down to rest
        'sss10': 'ess1',  # Traveling as passenger ~ Sitting and reading
        'sss11': 'ess2',  # Watching a film ~ Watching TV
        # 'sss_score': 'ess_score'  # Total scores comparison
    }
    df_mapper = pd.DataFrame()
    df_mapper['sss'] = questionnaire_map.keys()
    df_mapper['ess'] = questionnaire_map.values()
    df_mapper['ess'] = df_mapper['ess'].map(alias_no_enter_both)
    df_mapper['sss'] = df_mapper['sss'].map(alias_no_enter_both)

    # %% Table One
    comparator = QuestionnaireComparator(df=df_data,
                                         questionnaire_map=questionnaire_map,
                                         alias_dict=alias_no_enter)
    df_comparison_table = comparator.compare()
    df_comparison_table['ESS'] = df_comparison_table['ESS'].map(alias_no_enter_both)
    df_comparison_table['SSS'] = df_comparison_table['SSS'].map(alias_no_enter_both)

    # %%
    sss_cols = [col for col in sss_labels.keys() if 'score' not in col.lower()]
    ess_cols = [col for col in ess_labels.keys() if 'score' not in col.lower()]

    permutation = CrossQuestionPermutation(df=df_data,
                                           sss_cols=sss_cols,
                                           ess_cols=ess_cols)
    df_permutation_results = permutation.compare_all()

    df_permutation_results.sort_values(by=['SSS Question', 'p-value'], ascending=[True, True], inplace=True)
    df_permutation_results['SSS Question'] = df_permutation_results['SSS Question'].map(alias_no_enter_both)
    df_permutation_results['ESS Question'] = df_permutation_results['ESS Question'].map(alias_no_enter_both)


    def plot_sss_vs_ess_heatmaps(df_permutation_results):
        # Convert p-values to float
        df = df_permutation_results.copy()
        df['p-value'] = pd.to_numeric(df['p-value'], errors='coerce')

        # Convert to -log10(p-value) for better visualization
        df['-log10(p)'] = -np.log10(df['p-value'])

        for sss_question in df['SSS Question'].unique():
            subset = df[df['SSS Question'] == sss_question]
            if subset.empty:
                continue

            # Prepare heatmap data: one row (SSS question), many ESS columns
            heatmap_data = subset.pivot_table(index='SSS Question',
                                              columns='ESS Question',
                                              values='-log10(p)',
                                              aggfunc='first')

            plt.figure(figsize=(12, 1.8))
            ax = sns.heatmap(heatmap_data, annot=True, cmap='Blues',
                             cbar_kws={'label': '-log10(p-value)'}, fmt=".2f",
                             linewidths=0.5, linecolor='white')

            plt.title(f"Statistical Comparison: {sss_question}", fontsize=12)
            plt.yticks(rotation=0)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()


    plot_sss_vs_ess_heatmaps(df_permutation_results)
