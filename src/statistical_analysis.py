import pathlib
import pandas as pd
import statsmodels.api as sm
from typing import Union, Optional, Tuple
import numpy as np
from config.config import config, mapper_diagnosis
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
import seaborn as sns
import json
from questionnaires.questionnaires import SituationalSleepinessScale, EpworthScale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.gridspec import GridSpec

sns.set_context('poster')
#  'paper', 'notebook', 'talk', and 'poster'

def get_min_max(frame: pd.DataFrame, col: str) -> list[int, int]:
    """return the min and max as a lsit to make the limtis of the plot"""
    return [frame[col].min(), frame[col].max()]


def pca_interpretation(frame: pd.DataFrame,
                       columns: list[str],
                       n_components: Optional[int] = None,
                       figsize: Tuple = (8, 8),
                       output_path: Optional[pathlib.Path] = None,
                       plot: Optional[bool] = True,
                       file_name:Optional[str]='_') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute PCA and the associated descriptive measure for evaluating the PCA
    :param frame:dataset to apply pca
    :param columns: which columns of the dataset to use
    :param n_components: number of components to do the PCA
    :return:
        PCA Dataframe
        PCA Loadings
    """
    if not file_name.startswith('_'):
        file_name = "_"+file_name
    lbls_size = 15
    title_size = 18
    frame = frame[columns].copy()  # Ensure no NaN values for analysis

    # Standardize the data
    scaler = StandardScaler()
    df_frame_scaled = pd.DataFrame(scaler.fit_transform(frame),
                                   columns=frame.columns,
                                   index=range(0, questions_scaled.shape[0]))
    if n_components is None:
        n_components = frame.shape[1] - 1

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_frame_scaled)
    df_pca = pd.DataFrame(data=principal_components, columns=[f'PC_{i}' for i in range(0, pca.n_components)])

    explained_variance = pca.explained_variance_
    explained_variance_ratio_percent = np.round(pca.explained_variance_ratio_ * 100, 2)
    total_variance = pca.n_features_in_
    eigenvalues = pca.explained_variance_
    prop_var = eigenvalues / np.sum(eigenvalues)

    print(f'Total Variance: {total_variance}')
    print(f'Explained Variance Percentage: \n\t{explained_variance_ratio_percent}')

    # Project the factors
    factor_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    df_loadings = pd.DataFrame(factor_loadings,
                                    columns=[f'PC_{i}' for i in range(0, pca.n_components)],
                                    index=frame.columns)
    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        ax1, ax2, ax3, ax4 = axs.ravel()  # Unpack the subplots
        # Plotting the eigenvalues
        ax1.plot(np.arange(1, len(eigenvalues) + 1),
                 eigenvalues,
                 marker='o',
                 label='Eigenvalues')
        ax1.set_xlabel('Principal Component', size=lbls_size)
        ax1.set_ylabel('Eigenvalues', size=12)
        ax1.set_title('Scree Plot for Eigenvalues', size=title_size)
        ax1.axhline(y=1, color='r', linestyle='--', label='Threshold')
        ax1.grid(True)
        ax1.legend()

        # Scree plot of the proportional variance
        ax2.plot(np.arange(1, len(prop_var) + 1),
                 prop_var,
                 marker='o')
        ax2.set_xlabel('Principal Component', size=lbls_size)
        ax2.set_ylabel('Proportion of Variance Explained', size=lbls_size)
        ax2.set_title('Scree Plot for Proportion of \nVariance Explained', size=title_size)
        ax2.grid(True)

        # Scree plot of the explained variance
        ax3.plot(np.arange(1, len(prop_var) + 1),
                 explained_variance_ratio_percent,
                 marker='o')
        ax3.set_xlabel('Principal Component', size=lbls_size)
        ax3.set_ylabel('Explained Variance Percentage', size=lbls_size)
        ax3.set_title('Explained Variance Percentage \nPer Component', size=title_size)
        ax3.grid(True)

        # Scree plot of the explained variance
        ax4.plot(np.arange(1, len(prop_var) + 1),
                 explained_variance,
                 marker='o')
        ax4.set_xlabel('Principal Component', size=lbls_size)
        ax4.set_ylabel('Explained Variance Percentage', size=lbls_size)
        ax4.set_title('Explained Variance \nPer Component', size=title_size)
        ax4.grid(True)
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path.joinpath(f'PCA_figures{file_name}.png'), dpi=300)
        plt.show()

    if output_path is not None:
        df_pca.to_csv(output_path.joinpath(f'PCA_components{file_name}.csv'), index=False)
        df_loadings.to_csv(output_path.joinpath(f'PCA_loadings{file_name}.csv'), index=False)

    return df_pca, df_loadings


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
    corr_df_masked = corr_questions.loc[corr_questions.index.str.startswith("sss"), corr_questions.columns.str.startswith("ess")]
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

    diagnosis_categories = ['insomnia', 'narc_level','osa_level_ahi_3per', 'osa_level_ahi_4per',
                              'osa_level_odi_3per', 'osa_level_odi_4per']

    df_melted = pd.melt(df_box_scatter,
                        id_vars=['insomnia', 'narc_level', 'osa_level_ahi_3per', 'osa_level_ahi_4per'],
                        value_vars=['sss_score', 'ess_score'],
                        var_name='score_type',
                        value_name='score_value')

    palette = {"sss_score": "skyblue", "ess_score": "salmon"}

    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(nrows=2, ncols=2, wspace=0.2, hspace=0.4)
    ax_insomnia = fig.add_subplot(gs[0,0])
    ax_narc = fig.add_subplot(gs[0, 1])
    ax_osa_ahi_3per = fig.add_subplot(gs[1, 0])
    ax_osa_ahi_4per = fig.add_subplot(gs[1, 1])

    # insomnia
    sns.boxplot(data=df_melted,
                y='score_value',
                x='insomnia',
                hue='score_type',
                palette=palette,
                ax=ax_insomnia)
    ax_insomnia.set_title(f'SSS & ESS Score by Insomnia\nSamples {pd.isna(df_melted.insomnia).sum()}')
    ax_insomnia.set_xlabel(xlabel='Insomnia Diagnosis')
    ax_insomnia.set_ylabel(ylabel='Sleep Questionnaire\nScore', rotation=90)
    ax_insomnia.set_xticklabels(
        [mapper_diagnosis.get('insomnia')[int(float(tick.get_text()))] for tick in ax_insomnia.get_xticklabels()])
    ax_insomnia.grid(axis='y', alpha=0.7)

    sns.boxplot(data=df_melted,
                y='score_value',
                x='narc_level',
                hue='score_type',
                palette=palette,
                ax=ax_narc)
    ax_narc.set_title(f'SSS & ESS Score by Narcolepsy\nSamples {pd.isna(df_melted.narc_level).sum()}')
    ax_narc.set_xlabel(xlabel='Narcolepsy Diagnosis')
    ax_narc.set_ylabel(ylabel='')
    narc_levels_inv = {val:key for key, val in mapper_diagnosis.get('narc_level').items()}
    ax_narc.set_xticklabels(
        [narc_levels_inv[int(float(tick.get_text()))] for tick in ax_narc.get_xticklabels()])
    ax_narc.grid(axis='y', alpha=0.7)

    sns.boxplot(data=df_melted,
                y='score_value',
                x='osa_level_ahi_3per',
                hue='score_type',
                palette=palette,
                ax=ax_osa_ahi_3per)
    ax_osa_ahi_3per.set_title(f'SSS & ESS Score by OSA AHI 3%\nSamples {pd.isna(df_melted.osa_level_ahi_3per).sum()}')
    ax_osa_ahi_3per.set_xlabel(xlabel='OSA Levels')
    ax_osa_ahi_3per.set_ylabel(ylabel='Sleep Questionnaire\nScore')
    ax_osa_ahi_3per.set_xticklabels(
        [mapper_diagnosis.get('osa_levels')[int(float(tick.get_text()))] for tick in ax_osa_ahi_3per.get_xticklabels()])
    ax_osa_ahi_3per.grid(axis='y', alpha=0.7)

    sns.boxplot(data=df_melted,
                y='score_value',
                x='osa_level_ahi_4per',
                hue='score_type',
                palette=palette,
                ax=ax_osa_ahi_4per)
    ax_osa_ahi_4per.set_title(f'SSS & ESS Score by OSA AHI 4%\nSamples {pd.isna(df_melted.osa_level_ahi_4per).sum()}')
    ax_osa_ahi_4per.set_xlabel(xlabel='OSA Levels')
    ax_osa_ahi_4per.set_ylabel(ylabel='')
    ax_osa_ahi_4per.set_xticklabels(
        [mapper_diagnosis.get('osa_levels')[int(float(tick.get_text()))] for tick in ax_osa_ahi_4per.get_xticklabels()])
    ax_osa_ahi_4per.grid(axis='y', alpha=0.7)

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
    ax1_ticks_labels = [lbl.replace(" ", "\n", 1)  for lbl in list(ess_levels.values())]
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

    #%% measure the internal consistency or reliability
    # Cronbach's alpha
    def cronbach_alpha(df:pd.DataFrame) -> float:
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

    # %% Clustering self-supervised learning
    df_questions = df_data[questions_score].copy()  # Ensure no NaN values for analysis
    # Standardize the data
    scaler = StandardScaler()
    questions_scaled = scaler.fit_transform(df_questions)
    df_questions_scaled = pd.DataFrame(questions_scaled,
                                       columns=df_questions.columns,
                                       index=range(0, questions_scaled.shape[0]))
    # Apply PCA
    # Explore all the factors to determine number of components
    pca_interpretation(frame=df_data,
                       columns=questions_score,
                       figsize=(12, 12),
                       output_path=config.get('results_path'),
                       file_name='all_questions')
    # 3 factors seems to be enough
    df_pca, df_loadings = pca_interpretation(frame=df_data,
                                columns=questions_score,
                                figsize=(12, 12),
                                n_components=2,
                                file_name='two_components',
                                plot=False)
    # dataframe from poster where we place the desired labels
    df_loadings.reset_index(inplace=True, drop=False, names=['question'])

    questionnaire_lbls = sss_labels.copy()
    questionnaire_lbls.update(ess_labels)
    df_loadings.index = df_loadings['question'].map(questionnaire_lbls)

    # Biplot
    plt.figure(figsize=(16, 18))
    # sns.scatterplot(x='PC_0', y='PC_1', data=df_loadings)
    for idx, txt in zip(df_loadings.index, df_loadings.question):
        if 'score' in txt:
            color_vector = 'red'
        elif 'sss' in txt:
            color_vector = 'orange'
        else:
            color_vector = 'green'
        plt.arrow(x=0, y=0,
                  dx=df_loadings.at[idx, 'PC_0'], dy=df_loadings.at[idx, 'PC_1'],
                  color=color_vector,
                  alpha=0.5,
                  head_width=0.02, head_length=0.02, length_includes_head=False
                  )
        plt.text(x=df_loadings.at[idx, 'PC_0'] * 1.1,
                 y=df_loadings.at[idx, 'PC_1'] * 1.1,
                 s=idx,
                 color=color_vector,
                 ha='center',
                 va='center',
                 fontsize=13
                 )
    # for i, txt in enumerate(df_loadings.index):
    #     if 'score' in txt:
    #         color_vector = 'red'
    #     elif 'sss' in txt:
    #         color_vector = 'orange'
    #     else:
    #         color_vector = 'green'
    #     plt.arrow(0, 0, df_loadings.iloc[i, 0], df_loadings.iloc[i, 1], color=color_vector, alpha=0.5)
    #     plt.text(x=df_loadings.iloc[i, 0] * 1.2,
    #              y=df_loadings.iloc[i, 1] * 1.2,
    #              s=txt,
    #              color=color_vector,
    #              ha='center',
    #              va='center',
    #              # fontsize=11
    #              )
    circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.title('Factor Loadings ESS & SSS')
    plt.xlabel('Principal Component 0')
    plt.ylabel('Principal Component 1')
    plt.xlim([-0.25, 1.30])
    plt.ylim([-1, 1])
    plt.grid(alpha=0.4, axis='both')
    plt.tight_layout()
    plt.savefig(config.get('results_path').joinpath('PCA_Biplot_Lables.png'), dpi=300)
    plt.show()


    # %% PCA on each questionnaire to see how they are clustered
    df_pca_ess, df_loadings_ess = pca_interpretation(frame=df_data,
                                columns=col_ess,
                                figsize=(12, 12),
                                n_components=2,
                                file_name='two_components',
                                plot=False)

    df_pca_sss, df_loadings_sss = pca_interpretation(frame=df_data,
                                columns=col_sss,
                                figsize=(12, 12),
                                n_components=2,
                                file_name='two_components',
                                plot=False)
    # heat map to compare each factor loading
    fig = plt.figure(figsize=(16,16))
    gs = GridSpec(nrows=2, ncols=3, width_ratios=[0.8, 0.8, 1.5], wspace=0.8)  # 2 rows, 3 columns
    ax1 = fig.add_subplot(gs[0, 0:2])  # First row, first two columns
    ax3 = fig.add_subplot(gs[0:2, 2])  # Both rows, third column
    ax4 = fig.add_subplot(gs[1, 0:2])  # Second row, first two columns

    sns.heatmap(df_loadings_ess,
                annot=True,
                fmt=".2f",
                cmap=cmp_heatmap,
                cbar=False,
                ax=ax1)
    ax1.set_title('ESS Loadings')

    sns.heatmap(df_loadings,
                annot=True,
                fmt=".2f",
                cmap=cmp_heatmap,
                ax=ax3)
    ax3.set_title('Both Questionnaires Loadings')

    sns.heatmap(df_loadings_sss,
                annot=True,
                fmt=".2f",
                cbar=False,
                cmap=cmp_heatmap,
                ax=ax4)
    ax4.set_title('SSS Loadings')
    plt.subplots_adjust(wspace=0.4)
    fig.suptitle(f'Factor Loadings When PCA Applied Separately and Both Questionnaires')
    plt.savefig(config.get('results_path').joinpath('PCA_Loading_HeatMap_ThreeSubPlots.png'), dpi=300)
    plt.show()

    # scatter plot of the pca


    # %%  K-means Clustering on the PCA components
    # PCA using all the questions
    kmeans_all = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
    clusters = kmeans_all.fit_predict(df_pca)
    df_pca['Cluster'] = clusters

    # PCA using all the ESS questions
    kmeans_ess = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
    clusters = kmeans_ess.fit_predict(df_pca_ess)
    df_pca_ess['Cluster'] = clusters

    # PCA using all the sss questions
    kmeans_sss = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
    clusters = kmeans_sss.fit_predict(df_pca_sss)
    df_pca_sss['Cluster'] = clusters

    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(nrows=3, ncols=1, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Plot for df_pca
    sns.scatterplot(x='PC_0',
                    y='PC_1',
                    hue='Cluster',
                    palette='Set2',
                    data=df_pca,
                    alpha=0.8,
                    ax=ax1)
    ax1.set_title('PCA of ESS and SSS Questions with K-means Clusters')
    ax1.set_xlabel(' ')
    ax1.set_ylabel('Principal Component 2')
    ax1.legend(title='Cluster')
    ax1.grid(True)

    # Plot for df_pca_ess
    sns.scatterplot(x='PC_0',
                    y='PC_1',
                    hue='Cluster',
                    palette='Set2',
                    data=df_pca_ess,
                    alpha=0.8,
                    ax=ax2)
    ax2.set_title('PCA of ESS with K-means Clusters')
    ax2.set_xlabel(' ')
    ax2.set_ylabel('Principal Component 2')
    ax2.legend(title='Cluster')
    ax2.grid(True)

    # Plot for df_pca_sss
    sns.scatterplot(x='PC_0',
                    y='PC_1',
                    hue='Cluster',
                    palette='Set2',
                    data=df_pca_sss,
                    alpha=0.8,
                    ax=ax3)
    ax3.set_title('PCA of SSS Questions with K-means Clusters')
    ax3.set_xlabel('Principal Component 1')
    ax3.set_ylabel('Principal Component 2')
    ax3.legend(title='Cluster')
    ax3.grid(True)
    fig.suptitle(t='PCA Projections', fontsize=20, y=0.92)
    plt.savefig(config.get('results_path').joinpath('PCA_Projection_Kmeans_3Clusters.png'), dpi=300)
    plt.show()


    # TODO: new regression plot where we mark the high narc, osa and insomnia cases, so it can show the the
    #  questionnaire

    # TODO: color code diagnosis
    #     - red: narc
    #     - blue: apnea, osa level
    #     - pinc: insomnia
    #     - controld: black
    # TODO: change the size of the dots, increase the alpha, border thicker