"""
Perform PCA and kmeans analysis on the dataset to evalaute
1. factor loadings
2. clustering

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.gridspec import GridSpec
from utils.functions import pca_interpretation, cronbach_alpha

sns.set_context('poster')
#  'paper', 'notebook', 'talk', and 'poster'


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

    # %% define the questionnaire classes
    ess_quest = EpworthScale()
    sit_quest = SituationalSleepinessScale()

    # Get the labels
    ess_labels = ess_quest.get_labels()
    sss_labels = sit_quest.get_labels()

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
    fig = plt.figure(figsize=(16, 16))
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

