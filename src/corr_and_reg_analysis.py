"""
Perform the basic statistical analysis and visualization to compare the questionnaires.

We explore correlations, clustering, counts, t-test, linear regression

"""
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

sns.set_context('poster')


#  'paper', 'notebook', 'talk', and 'poster'

def get_min_max(frame: pd.DataFrame, col: str) -> list[int, int]:
    """return the min and max as a lsit to make the limtis of the plot"""
    return [frame[col].min(), frame[col].max()]


def plot_ess_sss_question_correlation(df: pd.DataFrame,
                                      ess_labels: Dict[str, str],
                                      sss_labels: Dict[str, str],
                                      lbls: Dict[str, str],
                                      output_path:Optional[pathlib.Path] = None,
                                      figsize: Tuple[int, int] = (16, 16),
                                      method: str = 'spearman',
                                      plot: bool = True,
                                      save: bool = True) -> pd.DataFrame:
    """
    Plot and return the pairwise correlation between SSS and ESS questionnaire responses.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing questionnaire responses.

    ess_labels : Dict[str, str]
        Mapping of ESS column names to display labels.

    sss_labels : Dict[str, str]
        Mapping of SSS column names to display labels.

    lbls : Dict[str, str]
        Dictionary with overall labels for axes (keys: 'ess', 'sss').

    output_path : pathlib.Path
        Directory where to save the plot and CSV.

    method : str, default='spearman'
        Correlation method to use ('spearman', 'pearson', or 'kendall').

    plot : bool, default=True
        Whether to display the heatmap.

    save : bool, default=True
        Whether to save the heatmap and correlation table.

    Returns
    -------
    pd.DataFrame
        A tidy DataFrame of pairwise correlations between ESS and SSS questions.
    """
    # questions_score = col_ess + col_sss
    # questions_no_score = [q for q in questions_score if 'score' not in q]
    #
    # corr_questions = df[questions_no_score].corr(method=method)
    corr_questions = df.corr(method=method)
    corr_df_masked = corr_questions.loc[
        corr_questions.index.str.startswith("sss"),
        corr_questions.columns.str.startswith("ess")
    ]

    lbl_sit_corr = {k: v for k, v in sss_labels.items() if 'score' not in k}
    lbl_ess_corr = {k: v for k, v in ess_labels.items() if 'score' not in k}

    if plot:
        plt.figure(figsize=figsize)
        cmap = sns.color_palette("Spectral_r", as_cmap=True)
        ax = sns.heatmap(corr_df_masked,
                         annot=True,
                         fmt=".2f",
                         cmap=cmap,
                         cbar_kws={'label': 'Correlation coefficient'})

        plt.title(f'{method.capitalize()} Correlation of \nQuestionnaire Responses - Observations: {df.shape[0]}')
        plt.xlabel(lbls.get('ess'))
        plt.ylabel(lbls.get('sss'))

        ax.set_xticks(ticks=np.arange(len(corr_df_masked.columns)) + 0.5)
        ax.set_xticklabels([lbl_ess_corr.get(col, col) for col in corr_df_masked.columns],
                           rotation=45, ha='right')

        ax.set_yticks(ticks=np.arange(len(corr_df_masked.index)) + 0.5)
        ax.set_yticklabels([lbl_sit_corr.get(row, row) for row in corr_df_masked.index],
                           rotation=0)

        plt.tight_layout()
        if save and output_path:
            plt.savefig(output_path.joinpath('correlation_ess_sss.png'), dpi=300)
        plt.show()

    # Create long-form correlation table
    corr_long = corr_df_masked.stack().reset_index()
    corr_long.columns = ['variable_one', 'variable_two', 'correlation']
    corr_long = corr_long.sort_values(by='correlation', ascending=False).reset_index(drop=True)

    if save and output_path:
        corr_long.to_csv(output_path.joinpath('correlation_table_ess_sss.csv'), index=False)

    return corr_long


def compare_ess_sss_scores(df: pd.DataFrame,
                           lbls: Dict[str, str],
                           get_min_max_func,
                           col_sss_score:str = 'sss_score',
                           col_ess_score:str = 'ess_score',
                           file_name:str = 'ess_vs_sss_score',
                           output_path: Optional[pathlib.Path] = None,
                           figsize:Tuple[int, int] = (8,8)) -> Dict[str, any]:
    """
    Perform OLS regression, scatterplot comparison, and t-test between ESS and SSS scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'ess_score' and 'sss_score' columns.

    output_path : pathlib.Path
        Path to save plots and result files.

    lbls : Dict[str, str]
        Dictionary with axis labels (keys: 'ess', 'sss').

    get_min_max_func : callable
        Function to get min-max for axis scaling, should accept (frame, col).

    Returns
    -------
    Dict[str, any]
        Dictionary with regression summary and t-test results.
    """
    df = df.copy()
    df['intercept'] = 1  # For OLS intercept
    rho = df[[col_sss_score, col_ess_score]].corr().iloc[0, 1].round(3)
    # --- OLS Regression ---
    model = sm.OLS(endog=df[col_sss_score], exog=df[['intercept', col_ess_score]])
    result = model.fit(cov_type='HC1')

    sample_size = result.nobs
    alpha = result.params['intercept']
    beta = result.params[col_ess_score]

    result_summary = {
        'params': result.params.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'rsquared': result.rsquared,
        'rsquared_adj': result.rsquared_adj,
        'fvalue': result.fvalue,
        'f_pvalue': result.f_pvalue,
        'conf_int': result.conf_int().to_dict(),
        'nobs': result.nobs,
        'df_model': result.df_model,
        'df_resid': result.df_resid
    }

    df_ols = pd.DataFrame({
        'coef': result.params,
        'std_err': result.bse,
        'p_value': result.pvalues,
        'CI_lower': result.conf_int()[0],
        'CI_upper': result.conf_int()[1]
    })
    if output_path:
        df_ols.to_csv(output_path.joinpath('OLS_results.csv'), index=False)
        with open(output_path.joinpath('Regression_ess_sss_scores_summary.json'), 'w') as f:
            json.dump(result_summary, f, indent=4)

    # --- Scatterplot + Regression Line ---
    xminmax = get_min_max_func(df, col=col_ess_score)
    yminmax = get_min_max_func(df, col=col_sss_score)

    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=col_ess_score, y=col_sss_score, label='ESS vs SSS')

    # Plot y = x
    plt.plot(yminmax, yminmax, color='gray', linestyle='--', label='y = x')

    # Plot regression line
    x_vals = np.array(xminmax)
    y_vals = alpha + beta * x_vals
    plt.plot(x_vals, y_vals, 'r-', label=f'y = {alpha:.2f} + {beta:.2f} x ESS')

    plt.title(f'ESS vs SSS Scores - ρ {rho}\nSample Size: {sample_size}')
    plt.xlabel(lbls.get('ess'))
    plt.ylabel(lbls.get('sss'))
    plt.xlim([xminmax[0] - 0.5, xminmax[1] + 0.5])
    plt.ylim([yminmax[0] - 0.5, yminmax[1] + 0.5])
    plt.legend()
    plt.grid(alpha=0.7)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path.joinpath(f'{file_name}_regression_ess_sss.png'), dpi=300)
    plt.show()

    # --- T-Test ---
    t_stat, p_val = ttest_ind(df[col_ess_score], df[col_sss_score])
    ttest_results = {
        "t_statistic": t_stat,
        "p_value": p_val,
        'sample_size_ess': df[col_ess_score].shape[0],
        'sample_size_sss': df[col_sss_score].shape[0],
    }

    df_ttest = pd.DataFrame(ttest_results, index=[0])
    df_ttest.to_csv(output_path.joinpath(f'{file_name}_stat_ttest_scores_ess_sss.csv'), index=False)
    if output_path:
        with open(output_path.joinpath(f'{file_name}_stat_ttest_scores_ess_sss.json'), 'w', encoding='utf-8') as f:
            json.dump(ttest_results, f, indent=4)

    return {
        "ols_summary": result_summary,
        "ttest": ttest_results
    }


def plot_ess_sss_distribution_by_diagnosis(df_data: pd.DataFrame,
                                           mapper_diagnosis: Dict[str, Dict[int, str]],
                                           output_path,
                                           lbls_narc: Dict[int, str],
                                           palette: Dict[str, str] = None,
                                           figsize=(30, 22)) -> None:
    """
    Generate boxplots comparing ESS and SSS scores across diagnosis categories.

    Parameters
    ----------
    df_data : pd.DataFrame
        The full DataFrame containing 'ess_score', 'sss_score', and diagnostic columns.

    mapper_diagnosis : Dict[str, Dict[int, str]]
        Mapping of diagnostic levels to string labels for each diagnosis variable.

    output_path : pathlib.Path
        Path to save the output plot.

    lbls_narc : Dict[int, str]
        Custom labels for narcolepsy levels (e.g. 0: 'No Narcolepsy').

    palette : Dict[str, str], optional
        Dictionary mapping score types to colors. Default uses 'skyblue' and 'salmon'.

    figsize : tuple, optional
        Size of the figure (width, height).
    """
    if palette is None:
        palette = {"sss_score": "skyblue", "ess_score": "salmon"}

    df_box_scatter = df_data[['sss_score', 'ess_score', 'insomnia', 'narc_level',
                              'osa_level_ahi_3per', 'osa_level_ahi_4per',
                              'osa_level_odi_3per', 'osa_level_odi_4per']].copy()

    # Exclude undefined narcolepsy levels
    df_box_scatter.loc[df_box_scatter['narc_level'] >= 4, 'narc_level'] = np.nan

    df_melted = pd.melt(df_box_scatter,
                        id_vars=['insomnia', 'narc_level', 'osa_level_ahi_3per', 'osa_level_ahi_4per'],
                        value_vars=['sss_score', 'ess_score'],
                        var_name='score_type',
                        value_name='score_value')

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=2, ncols=2, wspace=0.1, hspace=0.4)
    ax_insomnia = fig.add_subplot(gs[0, 0])
    ax_narc = fig.add_subplot(gs[0, 1])
    ax_osa_ahi_3per = fig.add_subplot(gs[1, 0])
    ax_osa_ahi_4per = fig.add_subplot(gs[1, 1])

    # --- Plot by insomnia ---
    sample_sizes_insomnia = df_box_scatter['insomnia'].value_counts()
    insomnia_levels_inv = {v: k for k, v in mapper_diagnosis.get('insomnia').items()}

    sns.boxplot(data=df_melted, x='insomnia', y='score_value', hue='score_type',
                palette=palette, ax=ax_insomnia)
    ax_insomnia.set_title(f'SSS & ESS Score by Insomnia\nSamples {sample_sizes_insomnia.sum()}')
    ax_insomnia.set_xlabel('Insomnia Diagnosis')
    ax_insomnia.set_ylabel('Sleep Questionnaire\nScore')
    ax_insomnia.set_xticklabels(
        [mapper_diagnosis['insomnia'][int(float(tick.get_text()))] for tick in ax_insomnia.get_xticklabels()])
    ax_insomnia.grid(axis='y', alpha=0.7)
    for tick, label in zip(ax_insomnia.get_xticks(), ax_insomnia.get_xticklabels()):
        level = insomnia_levels_inv[label.get_text()]
        size = sample_sizes_insomnia[level]
        ax_insomnia.annotate(f'n={size}', xy=(tick, 1), xytext=(0, 10),
                             textcoords='offset points', ha='center', va='bottom')

    # --- Plot by narcolepsy level ---
    sample_sizes_narc = df_box_scatter['narc_level'].value_counts()
    sns.boxplot(data=df_melted, x='narc_level', y='score_value', hue='score_type',
                palette=palette, ax=ax_narc)
    ax_narc.set_title(f'SSS & ESS Score by Hypersomnolence Disorders\nSamples {sample_sizes_narc.sum()}')
    ax_narc.set_xlabel('')
    ax_narc.set_ylabel('')
    ax_narc.set_xticklabels([lbls_narc[int(float(tick.get_text()))] for tick in ax_narc.get_xticklabels()])
    ax_narc.grid(axis='y', alpha=0.7)
    narc_levels_inv = {v: k for k, v in lbls_narc.items()}
    for tick, label in zip(ax_narc.get_xticks(), ax_narc.get_xticklabels()):
        level = narc_levels_inv[label.get_text()]
        size = sample_sizes_narc[level]
        ax_narc.annotate(f'n={size}', xy=(tick, 1), xytext=(0, 10),
                         textcoords='offset points', ha='center', va='bottom')

    # --- Plot by OSA AHI 3% ---
    sample_sizes_osa3 = df_box_scatter['osa_level_ahi_3per'].value_counts()
    osa_levels_inv = {v: k for k, v in mapper_diagnosis['osa_levels'].items()}
    sns.boxplot(data=df_melted, x='osa_level_ahi_3per', y='score_value', hue='score_type',
                palette=palette, ax=ax_osa_ahi_3per)
    ax_osa_ahi_3per.set_title(f'SSS & ESS Score by OSA AHI 3%\nSamples {sample_sizes_osa3.sum()}')
    ax_osa_ahi_3per.set_xlabel('OSA Levels')
    ax_osa_ahi_3per.set_ylabel('Sleep Questionnaire\nScore')
    ax_osa_ahi_3per.set_xticklabels(
        [mapper_diagnosis['osa_levels'][int(float(tick.get_text()))] for tick in ax_osa_ahi_3per.get_xticklabels()])
    ax_osa_ahi_3per.grid(axis='y', alpha=0.7)
    for tick, label in zip(ax_osa_ahi_3per.get_xticks(), ax_osa_ahi_3per.get_xticklabels()):
        level = osa_levels_inv[label.get_text()]
        size = sample_sizes_osa3[level]
        ax_osa_ahi_3per.annotate(f'n={size}', xy=(tick, 1), xytext=(0, 10),
                                 textcoords='offset points', ha='center', va='bottom')

    # --- Plot by OSA AHI 4% ---
    sample_sizes_osa4 = df_box_scatter['osa_level_ahi_4per'].value_counts()
    sns.boxplot(data=df_melted, x='osa_level_ahi_4per', y='score_value', hue='score_type',
                palette=palette, ax=ax_osa_ahi_4per)
    ax_osa_ahi_4per.set_title(f'SSS & ESS Score by OSA AHI 4%\nSamples {sample_sizes_osa4.sum()}')
    ax_osa_ahi_4per.set_xlabel('OSA Levels')
    ax_osa_ahi_4per.set_ylabel('')
    ax_osa_ahi_4per.set_xticklabels(
        [mapper_diagnosis['osa_levels'][int(float(tick.get_text()))] for tick in ax_osa_ahi_4per.get_xticklabels()])
    ax_osa_ahi_4per.grid(axis='y', alpha=0.7)
    for tick, label in zip(ax_osa_ahi_4per.get_xticks(), ax_osa_ahi_4per.get_xticklabels()):
        level = osa_levels_inv[label.get_text()]
        size = sample_sizes_osa4[level]
        ax_osa_ahi_4per.annotate(f'n={size}', xy=(tick, 1), xytext=(0, 10),
                                 textcoords='offset points', ha='center', va='bottom')

    # Global legend from first plot
    handles, labels = ax_insomnia.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, title='Score Type')

    # Remove repeated legends
    for ax in [ax_insomnia, ax_narc, ax_osa_ahi_3per, ax_osa_ahi_4per]:
        ax.legend_.remove()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path.joinpath('Distribution_boxplot_scores_diagnosis.png'), dpi=300)
    plt.show()

def compute_auc_multiclass_vs_zero(df,
                                    disorder_col,
                                    n_rows=1,
                                    figsize=(16, 5),
                                    alias: Dict[int, str] = None,
                                    output_path: pathlib.Path = None,
                                    plot=True):
    """
    Compute and visualize ROC AUC scores for ESS and SSS by comparing class 0
    to each positive class individually in a multiclass sleep disorder column.
    Also computes Youden index, sensitivity, specificity, and optimal cutoff.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the disorder column, 'ess_score', and 'sss_score'.

    disorder_col : str
        Name of the multiclass diagnosis column. Class 0 is the negative reference.

    n_rows : int, optional
        Number of subplot rows.

    figsize : tuple, optional
        Size of the figure.

    alias : Dict[int, str], optional
        Class label names for plotting.

    output_path : pathlib.Path, optional
        Where to save plots.

    plot : bool, default=True
        Whether to show the ROC subplot.

    Returns
    -------
    pd.DataFrame
        AUC scores, best cutoffs, sensitivity, specificity, and Youden index per class.
    """
    df_disorder = df.loc[~df[disorder_col].isna(), [disorder_col, 'ess_score', 'sss_score']].copy()
    df_disorder[disorder_col] = df_disorder[disorder_col].astype(int)

    unique_classes = sorted(df_disorder[disorder_col].unique())
    positive_classes = [cls for cls in unique_classes if cls != 0]
    n_plots = len(positive_classes)

    auc_results = []

    if plot:
        n_cols = math.ceil(n_plots / n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
        axes = axes.flatten() if n_plots > 1 else [axes]

    for i, pos_class in enumerate(positive_classes):
        binary_labels = df_disorder[disorder_col].apply(
            lambda x: 1 if x == pos_class else (0 if x == 0 else np.nan))
        df_binary = df_disorder.loc[~binary_labels.isna()].copy()
        binary_labels = binary_labels.loc[df_binary.index]

        y_true = binary_labels.values
        ess_scores = df_binary['ess_score'].values
        sss_scores = df_binary['sss_score'].values

        # ESS
        fpr_ess, tpr_ess, thresholds_ess = roc_curve(y_true, ess_scores)
        youden_ess = tpr_ess - fpr_ess
        idx_ess = np.argmax(youden_ess)
        cutoff_ess = thresholds_ess[idx_ess]
        sensitivity_ess = tpr_ess[idx_ess]
        specificity_ess = 1 - fpr_ess[idx_ess]
        auc_ess = roc_auc_score(y_true, ess_scores)

        # SSS
        fpr_sss, tpr_sss, thresholds_sss = roc_curve(y_true, sss_scores)
        youden_sss = tpr_sss - fpr_sss
        idx_sss = np.argmax(youden_sss)
        cutoff_sss = thresholds_sss[idx_sss]
        sensitivity_sss = tpr_sss[idx_sss]
        specificity_sss = 1 - fpr_sss[idx_sss]
        auc_sss = roc_auc_score(y_true, sss_scores)

        auc_results.append({
            'comparison': f'0 vs {pos_class}',
            'n_class_0': (y_true == 0).sum(),
            'n_class_pos': (y_true == 1).sum(),
            'AUC_ESS': auc_ess,
            'ESS_cutoff': cutoff_ess,
            'ESS_sensitivity': sensitivity_ess,
            'ESS_specificity': specificity_ess,
            'ESS_youden_index': youden_ess[idx_ess],
            'AUC_SSS': auc_sss,
            'SSS_cutoff': cutoff_sss,
            'SSS_sensitivity': sensitivity_sss,
            'SSS_specificity': specificity_sss,
            'SSS_youden_index': youden_sss[idx_sss],

        })

        if plot:
            ax = axes[i]
            ax.plot(fpr_ess, tpr_ess, label=f'ESS (AUC = {auc_ess:.2f})', color='salmon')
            ax.plot(fpr_sss, tpr_sss, label=f'SSS (AUC = {auc_sss:.2f})', color='skyblue')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            title = f'{alias.get(0)} vs \n{alias.get(pos_class)}' if alias else f'0 vs {pos_class}'
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend()

    if plot:
        # Hide unused plots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.supxlabel('False Positive Rate', fontsize=16)
        fig.supylabel('True Positive Rate', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        if output_path:
            plt.savefig(output_path.joinpath(f'auc_ess_vs_sss_{disorder_col}.png'), dpi=300)
        plt.show()

    return pd.DataFrame(auc_results)



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

    # %% cut offs
    # df_subjects_ess_score_eqge_10 = df_data.loc[df_data.ess_score >= 10, ['ess_score', 'sss_score']]
    # df_subjects_ess_score_eqge_10.loc[df_subjects_ess_score_eqge_10.sss_score >= 10, 'sss_score']
    #
    #
    # df_subjects_sss_score_eqge_10 = df_data.loc[df_data.sss_score >= 10, ['ess_score', 'sss_score']]
    # df_subjects_sss_score_eqge_10.loc[df_subjects_sss_score_eqge_10.ess_score >= 10, 'ess_score']
    # %% define the questionnaire classes
    ess_quest = EpworthScale()
    sit_quest = SituationalSleepinessScale()

    # Get the labels
    ess_labels = ess_quest.get_labels()
    sss_labels = sit_quest.get_labels()

    alias = ess_labels.copy()
    alias.update(sss_labels)

    # %% 0. Compare ordinal responses
    def compare_ordinal_questions(df:pd.DataFrame,
                                  questionnaire_map:Dict[str, str],
                                  alias_dict:Dict[str,str],
                                  output_path:pathlib.Path=None):
        """
        Generate side-by-side bar plots comparing ordinal responses from two questionnaires.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the responses
        - questionnaire_map (dict): Mapping of column names from questionnaire 1 to questionnaire 2
        - alias_dict (dict): Mapping of column names to formal titles for plots
        """
        sns.set(style='whitegrid')

        for col1, col2 in questionnaire_map.items():
            # col1 = [*questionnaire_map.keys()][0]
            # col2 = questionnaire_map.get(col1)
            if col1 not in df.columns or col2 not in df.columns:
                print(f"Skipping: {col1} or {col2} not in DataFrame.")
                continue

            # Count the responses
            q1_counts = df[col1].value_counts(normalize=True).sort_index()
            q2_counts = df[col2].value_counts(normalize=True).sort_index()

            n_q1 = int(df.shape[0] - df[col1].isna().sum())
            n_q2 = int(df.shape[0] - df[col2].isna().sum())

            # Create a combined DataFrame
            combined = pd.DataFrame({
                col1: q1_counts,
                col2: q2_counts
            })  # .fillna(0)

            # Reorder index if it's ordinal
            if combined.index.dtype.name == 'category' or combined.index.dtype == object:
                combined = combined.sort_index()

            # Plot
            ax = combined.plot(kind='bar', width=0.8)
            plt.title(f'{col1}: {alias_dict.get(col1)} ({n_q1}) \n vs \n {col2}: {alias_dict.get(col2)} ({n_q2})')
            plt.xlabel("Response")
            plt.ylabel("Proportion")
            plt.xticks(rotation=0)
            plt.legend()
            plt.tight_layout()
            if output_path:
                plt.savefig(output_path.joinpath(f'{col1}_{col2}.png'), dpi=300)
            plt.show()


    questionnaire_map = {
        'sss1': 'ess1',  # unintentional sleep likelihood ~ Sitting and reading
        'sss2': 'ess2',  # On computer/tablet ~ Watching TV
        'sss3': 'ess5',  # Talking on phone ~ Talking to someone
        'sss4': 'ess6',  # Meeting ~ Talking to someone
        'sss5': 'ess6',  # Listening to speaker ~ Talking to someone
        'sss6': 'ess8',  # Board game ~ In a car, while stopped in traffic
        'sss7': 'ess7',  # Driving ~ Driving in traffic
        'sss8': 'ess4',  # Playing video game ~ Lying down to rest in the afternoon
        'sss9': 'ess4',  # Lying down ~ Lying down to rest
        'sss10': 'ess1',  # Traveling as passenger ~ Sitting and reading
        'sss11': 'ess2',  # Watching a film ~ Watching TV
        # 'sss_score': 'ess_score'  # Total scores comparison
    }

    compare_ordinal_questions(df=df_data,
                              questionnaire_map=questionnaire_map,
                              alias_dict={key:val.replace('\n', ' ') for key, val in alias.items()},
                              output_path=output_path)

    # %% 0.1 Table One


    # %% 1. Correlation among features
    # generating correlation_table_ess_sss.csv and correlation_table_ess_sss.png
    questions_score = col_ess + col_sss
    questions_no_score = [ques for ques in questions_score if 'score' not in ques]
    corr_questions = df_data[questions_no_score].corr(method='spearman')

    corr_df = plot_ess_sss_question_correlation(df=df_data[questions_no_score],
                                                ess_labels=ess_labels,
                                                sss_labels=sss_labels,
                                                lbls=lbls,
                                                figsize=(8,8),
                                                output_path=output_path)

    # %% 2. comparison of the scores
    # generates: OLS_results.csv, Regression_ess_sss.png, stat_ttest_scores_ess_sss.csv
    compare_results = compare_ess_sss_scores(df=df_data,
                                             output_path=output_path,
                                             col_sss_score='sss_score',
                                             col_ess_score='ess_score',
                                             file_name='ess_vs_sss',
                                             lbls=lbls,
                                             figsize=(8,6),
                                             get_min_max_func=get_min_max)

    compare_results_normalied = compare_ess_sss_scores(df=df_data,
                                             output_path=output_path,
                                             col_sss_score='sss_score_div_num_quest',
                                             col_ess_score='ess_score_div_num_quest',
                                            file_name='ess_vs_sss_normalized',
                                            lbls=lbls,
                                             figsize=(8,6),
                                             get_min_max_func=get_min_max)

    def generate_adamant_plot(df: pd.DataFrame, output_path: pathlib.Path = None, figsize=(10, 6)):
        import statsmodels.api as sm

        # Ensure required columns exist
        if 'ess_score' not in df.columns or 'sss_score' not in df.columns:
            print("Columns 'ess_score' and/or 'sss_score' not found in DataFrame.")
            return

        df = df[['ess_score', 'sss_score']].dropna()
        df['intercept'] = 1

        model = sm.OLS(endog=df['sss_score'], exog=df[['intercept', 'ess_score']])
        result = model.fit(cov_type='HC1')

        infl = result.get_influence()
        leverage = infl.hat_matrix_diag
        standardized_residuals = infl.resid_studentized_internal

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(leverage, standardized_residuals ** 2, alpha=0.7)
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals Squared')
        ax.set_title('Adamant Plot: Leverage vs. Standardized Residuals Squared')
        ax.axhline(y=4, color='r', linestyle='--', label='Residual Threshold')
        ax.axvline(x=0.2, color='g', linestyle='--', label='Leverage Threshold')
        ax.legend()

        if output_path:
            file_path = output_path.joinpath("adamant_plot.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()


    generate_adamant_plot(df=df_data, output_path=output_path, figsize=(10, 6))


    # %% 2.2 comparison of the scores with diagnosis as hues
    # generates Distribution_boxplot_scores_diagnosis.png
    lbls_narc = {
        0: 'No Narcolepsy',
        1: 'Narcolepsy\nwith\ncataplexy',
        2: 'Narcolepsy\nwithout\ncataplexy',
        3: 'Idiopathic\nhypersomnia'
    }
    plot_ess_sss_distribution_by_diagnosis(
        df_data=df_data,
        mapper_diagnosis=mapper_diagnosis,
        output_path=output_path,
        lbls_narc=lbls_narc,
        figsize=(16, 14)
    )

    # %% Different thresholds ESS and SSS for positive diagnosis
    # Question: How does the ESS thresholds translates to the SSS

    df_auc_narc = compute_auc_multiclass_vs_zero(
        df=df_data,
        disorder_col='narc_level',
        n_rows=2,
        figsize=(8, 6),
        alias={key:val.replace('\n', '') for key, val in lbls_narc.items()},
        plot=True,
        output_path=output_path
    )

    df_auc_insomnia = compute_auc_multiclass_vs_zero(
        df=df_data,
        disorder_col='insomnia',
        n_rows=1,
        figsize=(8, 6),
        plot=True,
        alias={0:'No Insomnia', 1:'Insomnia'},
        output_path=output_path
    )

    # %% Statistical test to evalaute the score between the SSS and the ESS
    # TODO: stats test

    # %% 3. Comparing the responses
    # get the y ticks for the ess and sss
    ess_levels = {v: k for k, v in ess_quest.levels.items()}
    ess_levels[4] = ''
    sss_levels = {v: k for k, v in sit_quest.levels.items() if v < 5}
    palette = ['darkorchid' if 'ess' in col else 'c' for col in questions_no_score]

    fig, ax1 = plt.subplots(figsize=(16, 10))
    sns.boxplot(data=df_data[questions_no_score],
                palette=palette,
                ax=ax1)
    ax1.set_title('Box Plot of Survey Questions', fontsize=16)
    ax1.set_xlabel('Questions', fontsize=14)
    ax1.set_ylabel('Responses', fontsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='center')

    # Set y-ticks to be only integers, important to place our ordinal scale labels
    max_value = int(df_data[questions_no_score].max().max())
    ax1.set_yticks(np.arange(0, max_value + 1, step=1))
    # left side y ticks (ess)
    ax1.set_yticks(ax1.get_yticks())
    ax1_ticks_labels = [lbl.replace(" ", "\n", 1) for lbl in list(ess_levels.values())]
    ax1.set_yticklabels(labels=ax1_ticks_labels,
                        rotation=0,
                        verticalalignment='center',
                        fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    # right side y ticks (sss)
    ax2 = ax1.twinx()
    ax2.set_yticks(ax1.get_yticks())
    ax2_ticks_labels = [lbl.replace(' ', '\n') for lbl in list(sss_levels.values())]
    ax2.set_yticklabels(labels=ax2_ticks_labels,
                        rotation=0,
                        verticalalignment='center',
                        fontsize=12)
    plt.savefig(output_path.joinpath('Distribution_boxplot_ess_sss.png'), dpi=300)
    plt.show()


    # %% measure the internal consistency or reliability
    # Cronbach's alpha
    def cronbach_alpha(df: pd.DataFrame) -> float:
        # Number of items
        items = df.shape[1]
        # Variance of total scores
        total_var = df.sum(axis=1).var(ddof=1)
        # Sum of item variances
        sum_var = df.var(axis=0, ddof=1).sum()
        # Calculate Cronbach's alpha
        return (items / (items - 1)) * (1 - sum_var / total_var)


    # Separate the data for ESS and SSS
    df_ess = df_data.filter(like='ess')
    df_sss = df_data.filter(like='sss')

    # Calculate Cronbach's alpha for ESS and SSS
    alpha_ess = cronbach_alpha(df_ess)
    alpha_sss = cronbach_alpha(df_sss)

    print(f"Cronbach's alpha for ESS questions: {alpha_ess:.2f}")
    print(f"Cronbach's alpha for SSS questions: {alpha_sss:.2f}")

    cronbach_alpha = {
        'alpha_ess': alpha_ess,
        'alpha_sss': alpha_sss,
        'columns_ess': df_ess.columns.tolist(),
        'columns_sss': df_sss.columns.tolist(),
    }
    path = config.get('results_path').joinpath('cronbach_alpha.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cronbach_alpha, f, ensure_ascii=False, indent=4)
