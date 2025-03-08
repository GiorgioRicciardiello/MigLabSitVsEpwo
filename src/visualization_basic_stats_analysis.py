"""
Perform the basic statistical analysis and visualization to compare the questionnaires.

We explore correlations, clustering, counts, t-test, linear regression

"""
import pathlib
import pandas as pd
import statsmodels.api as sm
from typing import Union, Optional, Tuple, List
import numpy as np
from config.config import config, mapper_diagnosis
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
import seaborn as sns
import json
from questionnaires.questionnaires import SituationalSleepinessScale, EpworthScale
from matplotlib.gridspec import GridSpec

sns.set_context('poster')


#  'paper', 'notebook', 'talk', and 'poster'

def get_min_max(frame: pd.DataFrame, col: str) -> list[int, int]:
    """return the min and max as a lsit to make the limtis of the plot"""
    return [frame[col].min(), frame[col].max()]


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

    # %% 1. Correlation among features
    questions_score = col_ess + col_sss
    questions_no_score = [ques for ques in questions_score if 'score' not in ques]
    corr_questions = df_data[questions_no_score].corr(method='spearman')
    # get only the correlation heatmap when the questionnaires are being compared
    corr_df_masked = corr_questions.loc[
        corr_questions.index.str.startswith("sss"), corr_questions.columns.str.startswith("ess")]
    lbl_sit_corr = {key: val for key, val in sss_labels.items() if 'score' not in key}
    lbl_ess_corr = {key: val for key, val in ess_labels.items() if 'score' not in key}
    plt.figure(figsize=(20, 18))  # You can adjust the size as needed
    # cbar_kws = {'label': 'Correlation coefficient', 'ticks': 10}
    cmp_heatmap = sns.color_palette("Spectral_r", as_cmap=True)
    ax = sns.heatmap(corr_df_masked,
                     annot=True,
                     fmt=".2f",
                     cmap=cmp_heatmap,
                     cbar_kws={'label': 'Correlation coefficient'})
    plt.title(f'Spearman Correlation of \n Questionnaire Responses - Observations: {df_data.shape[0]}')
    plt.xlabel(lbls.get('ess'))
    plt.ylabel(lbls.get('sss'))

    # Center the ticks with the boxes
    ax.set_xticks(ticks=np.arange(len(corr_df_masked.columns)) + 0.5,
                  labels=[lbl_ess_corr.get(col, col) for col in corr_df_masked.columns],
                  rotation=45, ha='right'
                  )
    ax.set_yticks(ticks=np.arange(len(corr_df_masked.index)) + 0.5,
                  labels=[lbl_sit_corr.get(row, row) for row in corr_df_masked.index],
                  rotation=0
                  )
    # plt.xticks(ticks=np.arange(len(lbl_ess_corr)),
    #            labels=[lbl_ess_corr.get(col, col) for col in corr_df_masked.columns],
    #            rotation=45, ha='right')
    # plt.yticks(ticks=np.arange(len(lbl_sit_corr)),
    #            labels=[lbl_sit_corr.get(row, row) for row in corr_df_masked.index],
    #            rotation=0)
    plt.tight_layout()
    plt.savefig(config.get('results_path').joinpath('correlation_ess_sss.png'), dpi=300)
    plt.show()

    # get a three columns table with the variables and their correlation
    corr_df_masked = corr_df_masked.stack().reset_index()
    corr_df_masked.columns = ['variable_one', 'variable_two', 'correlation']
    # corr_df_masked = corr_df_masked[(corr_df_masked['correlation'] > 0.4) | (corr_df_masked['correlation'] < -0.4)]
    corr_df_masked.sort_values(by='correlation',
                               ascending=False,
                               inplace=True)
    corr_df_masked.reset_index(inplace=True,
                               drop=True)
    corr_df_masked.to_csv(config.get('results_path').joinpath('correlation_table_ess_sss.csv'), index=False)

    # %% 2. comparison of the scores
    df_data['intercept'] = 1  # Add an intercept (beta_0)
    model_ess_sss_scores = sm.OLS(endog=df_data['sss_score'],
                                  exog=df_data[['intercept', 'ess_score']])
    result = model_ess_sss_scores.fit()
    print(result.summary())
    alpha = result.params['intercept']
    beta = result.params['ess_score']
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

    # Save as JSON
    with open(config.get('results_path').joinpath('Regression_ess_sss_scores_summary.json'), 'w') as f:
        json.dump(result_summary, f, indent=4)

    # comparing two continous outcomes, we can use a t-test
    xminmax = get_min_max(frame=df_data, col='ess_score')
    yminmax = get_min_max(frame=df_data, col='sss_score')
    plt.figure(figsize=(20, 14))
    sns.scatterplot(data=df_data,
                    x='ess_score',
                    y='sss_score',
                    label='ESS VS SSS'
                    )

    # travers the plane with Y = X
    plt.plot(yminmax,
             yminmax,
             color='gray',
             linestyle='--',
             label='y = x')

    # Plot regression line
    x_vals = np.array(xminmax)
    y_vals = alpha + beta * x_vals
    plt.plot(x_vals,
             y_vals,
             'r-',
             label=f' y = {alpha:.2f} + {beta:.2f} x ESS')
    plt.title(f'Comparison of ESS and SSS Scores\n Sample Size {df_data.shape[0]}')
    plt.xlabel(lbls.get('ess'))
    plt.ylabel(lbls.get('sss'))
    plt.xlim(xminmax)
    plt.ylim(yminmax)
    plt.legend()
    plt.grid(alpha=.7)
    plt.tight_layout()
    plt.savefig(config.get('results_path').joinpath('Regression_ess_sss.png'), dpi=300)
    plt.show()

    # T-test
    t_stat, p_value = ttest_ind(df_data['ess_score'], df_data['sss_score'])
    ttest_results = {
        "t_statistic": t_stat,
        "p_value": p_value,
        'sample_size_ess': df_data['ess_score'].shape[0],
        'sample_size_sss': df_data['sss_score'].shape[0],
    }
    print("T-statistic:", ttest_results.get('t_statistic'))
    print("P-value:", ttest_results.get('p_value'))
    path = config.get('results_path').joinpath('stat_ttest_scores_ess_sss.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ttest_results, f, ensure_ascii=False, indent=4)

    # %% 2.2 comparison of the scores with diagnosis as hues
    # to make a pried box plot, we need to have the dataset in long form
    df_box_scatter = df_data[['sss_score', 'ess_score', 'insomnia', 'narc_level',
                              'osa_level_ahi_3per', 'osa_level_ahi_4per',
                              'osa_level_odi_3per', 'osa_level_odi_4per']].copy()

    # exclude narcolepsy diagnosis
    df_box_scatter.loc[df_box_scatter['narc_level'] >= 4, 'narc_level'] = np.nan
    lbls_narc = {
        0: 'No Narcolepsy',
        1: 'Narcolepsy\nwith\ncataplexy',
        2: 'Narcolepsy\nwithout\ncataplexy',
        3: 'Idiopathic\nhypersomnia'
    }
    diagnosis_categories = ['insomnia', 'narc_level', 'osa_level_ahi_3per', 'osa_level_ahi_4per',
                            'osa_level_odi_3per', 'osa_level_odi_4per']

    df_melted = pd.melt(df_box_scatter,
                        id_vars=['insomnia', 'narc_level', 'osa_level_ahi_3per', 'osa_level_ahi_4per'],
                        value_vars=['sss_score', 'ess_score'],
                        var_name='score_type',
                        value_name='score_value')

    palette = {"sss_score": "skyblue", "ess_score": "salmon"}

    fig = plt.figure(figsize=(30, 22))
    gs = GridSpec(nrows=2, ncols=2, wspace=0.1, hspace=0.4)
    ax_insomnia = fig.add_subplot(gs[0, 0])
    ax_narc = fig.add_subplot(gs[0, 1])
    ax_osa_ahi_3per = fig.add_subplot(gs[1, 0])
    ax_osa_ahi_4per = fig.add_subplot(gs[1, 1])

    # insomnia
    insomnia_levels_inv = {val: key for key, val in mapper_diagnosis.get('insomnia').items()}

    # fig = plt.figure(figsize=(16, 8))
    # ax_insomnia = fig.add_subplot(111)
    sns.boxplot(data=df_melted,
                y='score_value',
                x='insomnia',
                hue='score_type',
                palette=palette,
                ax=ax_insomnia)
    ax_insomnia.set_title(f'SSS & ESS Score by Insomnia\nSamples {df_box_scatter.insomnia.value_counts().sum()}')
    ax_insomnia.set_xlabel(xlabel='Insomnia Diagnosis')
    ax_insomnia.set_ylabel(ylabel='Sleep Questionnaire\nScore', rotation=90)
    ax_insomnia.set_xticklabels(
        [mapper_diagnosis.get('insomnia')[int(float(tick.get_text()))] for tick in ax_insomnia.get_xticklabels()])
    ax_insomnia.grid(axis='y', alpha=0.7)

    # Annotate sample sizes on top of each x-axis tick
    sample_sizes_insomnia = df_box_scatter['insomnia'].value_counts()
    for tick, label in zip(ax_insomnia.get_xticks(), ax_insomnia.get_xticklabels()):
        level = insomnia_levels_inv[label.get_text()]
        size = sample_sizes_insomnia[level]
        ax_insomnia.annotate(f'n={size}', xy=(tick, 1), xytext=(0, 10),
                         textcoords='offset points', ha='center', va='bottom')

    # narcolepsy
    sample_sizes_hypersomnia = df_box_scatter['narc_level'].value_counts()
    sns.boxplot(data=df_melted,
                y='score_value',
                x='narc_level',
                hue='score_type',
                palette=palette,
                ax=ax_narc)
    ax_narc.set_title(f'SSS & ESS Score by hypersomnolence Disorders\n'
                      f'Samples {df_box_scatter.narc_level.value_counts().sum()}')
    ax_narc.set_xlabel(xlabel=' ')
    ax_narc.set_ylabel(ylabel='')
    # narc_levels_inv = {val: key for key, val in mapper_diagnosis.get('narc_level').items()}
    # ax_narc.set_xticklabels(
    #     [narc_levels_inv[int(float(tick.get_text()))] for tick in ax_narc.get_xticklabels()])
    ax_narc.set_xticklabels(
        [lbls_narc[int(float(tick.get_text()))] for tick in ax_narc.get_xticklabels()])
    ax_narc.grid(axis='y', alpha=0.7)

    # Annotate sample sizes on top of each x-axis tick
    narc_levels_inv = {val: key for key, val in lbls_narc.items()}
    for tick, label in zip(ax_narc.get_xticks(), ax_narc.get_xticklabels()):
        level = narc_levels_inv[label.get_text()]
        size = sample_sizes_hypersomnia[level]
        ax_narc.annotate(f'n={size}', xy=(tick, 1), xytext=(0, 10),
                         textcoords='offset points', ha='center', va='bottom')

    # OSA 3%
    osa_levels_inv = {val: key for key, val in mapper_diagnosis.get('osa_levels').items()}
    sns.boxplot(data=df_melted,
                y='score_value',
                x='osa_level_ahi_3per',
                hue='score_type',
                palette=palette,
                ax=ax_osa_ahi_3per)
    ax_osa_ahi_3per.set_title(f'SSS & ESS Score by OSA AHI 3%\n'
                              f'Samples {df_box_scatter.osa_level_ahi_3per.value_counts().sum()}')
    ax_osa_ahi_3per.set_xlabel(xlabel='OSA Levels')
    ax_osa_ahi_3per.set_ylabel(ylabel='Sleep Questionnaire\nScore')
    ax_osa_ahi_3per.set_xticklabels(
        [mapper_diagnosis.get('osa_levels')[int(float(tick.get_text()))] for tick in ax_osa_ahi_3per.get_xticklabels()])
    ax_osa_ahi_3per.grid(axis='y', alpha=0.7)

    # Annotate sample sizes on top of each x-axis tick
    sample_sizes_osa = df_box_scatter['osa_level_ahi_3per'].value_counts()
    for tick, label in zip(ax_osa_ahi_3per.get_xticks(), ax_osa_ahi_3per.get_xticklabels()):
        level = osa_levels_inv[label.get_text()]
        size = sample_sizes_osa[level]
        ax_osa_ahi_3per.annotate(f'n={size}', xy=(tick, 1), xytext=(0, 10),
                         textcoords='offset points', ha='center', va='bottom')

    # OSA 4%
    # fig = plt.figure(figsize=(16, 8))
    # ax_osa_ahi_4per = fig.add_subplot(111)
    sns.boxplot(data=df_melted,
                y='score_value',
                x='osa_level_ahi_4per',
                hue='score_type',
                palette=palette,
                ax=ax_osa_ahi_4per)
    ax_osa_ahi_4per.set_title(f'SSS & ESS Score by OSA AHI 4%\n'
                              f'Samples {df_box_scatter.osa_level_ahi_4per.value_counts().sum()}')
    ax_osa_ahi_4per.set_xlabel(xlabel='OSA Levels')
    ax_osa_ahi_4per.set_ylabel(ylabel='')
    ax_osa_ahi_4per.set_xticklabels(
        [mapper_diagnosis.get('osa_levels')[int(float(tick.get_text()))] for tick in ax_osa_ahi_4per.get_xticklabels()])
    ax_osa_ahi_4per.grid(axis='y', alpha=0.7)

    # Annotate sample sizes on top of each x-axis tick
    sample_sizes_osa = df_box_scatter['osa_level_ahi_4per'].value_counts()
    for tick, label in zip(ax_osa_ahi_4per.get_xticks(), ax_osa_ahi_4per.get_xticklabels()):
        level = osa_levels_inv[label.get_text()]
        size = sample_sizes_osa[level]
        ax_osa_ahi_4per.annotate(f'n={size}', xy=(tick, 1), xytext=(0, 10),
                         textcoords='offset points', ha='center', va='bottom')
    # plt.show()

    # Add a legend from one of the plots to the entire figure
    handles, labels = ax_insomnia.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, title='Score Type')

    # Remove individual legends
    ax_insomnia.legend_.remove()
    ax_narc.legend_.remove()
    ax_osa_ahi_3per.legend_.remove()
    ax_osa_ahi_4per.legend_.remove()

    plt.tight_layout()
    plt.savefig(config.get('results_path').joinpath('Distribution_boxplot_scores_diagnosis.png'), dpi=300)
    plt.show()

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
    plt.savefig(config.get('results_path').joinpath('Distribution_boxplot_ess_sss.png'), dpi=300)
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
