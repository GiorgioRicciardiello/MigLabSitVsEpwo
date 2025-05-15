from matplotlib.patches import FancyArrowPatch
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
from utils.functions import pca_interpretation, cronbach_alpha
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cross_decomposition import CCA
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import gower
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from itertools import combinations
from scipy.spatial.distance import pdist
sns.set_context('poster')
#  'paper', 'notebook', 'talk', and 'poster'

def perform_efa_analysis(data:pd.DataFrame,
                         columns:List[str],
                         questionnaire_name:str="Questionnaire"):
    """
    Performs exploratory factor analysis (EFA) on a set of questionnaire items,
    prints sampling adequacy measures, and plots the scree plot of eigenvalues.

    Parameters:
    - data: pandas DataFrame containing the questionnaire responses.
    - columns: list of column names corresponding to the questionnaire items.
    - questionnaire_name: string name to display in titles and print statements.

    Returns:
    - fa: the fitted FactorAnalyzer object.
    - eigenvalues: array of eigenvalues from the analysis.
    - df_loadings: loadings with index names and column names.
    """

    print(f"\n--- {questionnaire_name} Questionnaire ---")

    # Check sampling adequacy using Bartlett's test and KMO measure
    chi_square_value, p_value = calculate_bartlett_sphericity(data[columns])
    kmo_all, kmo_model = calculate_kmo(data[columns])
    print("Bartlett's test chi-square:", chi_square_value, "p-value:", p_value)
    print("KMO Measure:", kmo_model)

    # Perform EFA using varimax rotation
    n_factors = 3
    fa = FactorAnalyzer(rotation='varimax', n_factors=n_factors)
    fa.fit(data[columns])

    # Get eigenvalues to help decide the number of factors
    eigenvalues, _ = fa.get_eigenvalues()

    # Plot the scree plot
    plt.figure(figsize=(8, 6))
    x_values = range(1, len(eigenvalues) + 1)
    plt.scatter(x_values, eigenvalues, label="Eigenvalues")
    plt.plot(x_values, eigenvalues, 'r-', label="Line")
    plt.title(f'Scree Plot for {questionnaire_name}')
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, color='black', linestyle='--', label="Eigenvalue = 1")
    plt.legend()
    plt.tight_layout()
    plt.show()

    df_loadings = pd.DataFrame(data=fa.loadings_,
                               index=list(ess_labels.keys()),
                               columns=[f'CCA_{idx}' for idx in range(0, n_factors)])
    return fa, eigenvalues, df_loadings

def plot_factor_visualizations(df_loadings:pd.DataFrame,
                               n_factors:int=2,
                               questionnaire_name="Questionnaire",
                               output_path:pathlib.Path=None):
    """
    Visualizes EFA results through various plots:
    - Factor Loadings Heatmap
    - Factor Scores Scatter Plot (if n_factors >= 2)
    - Hierarchical Clustering Dendrogram based on factor loadings

    Parameters:
    - columns: list of column names corresponding to the questionnaire items.
    - fa: the fitted FactorAnalyzer object.
    - n_factors: number of factors extracted; default is 2.
    - questionnaire_name: name of the questionnaire for titles and labels.
    """

    # 1. Factor Loadings Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df_loadings,
        annot=True,            # Show numeric values in cells
        fmt=".2f",             # Format annotations to 2 decimal places
        cmap="Blues",          # Blue color palette
        # linewidths=0.5,        # Draw lines between cells
        # linecolor="white",
        yticklabels=df_loadings.index,
        annot_kws={"fontsize": 10},
        # cbar_kws={"shrink": 0.8, "aspect": 20},
        cbar=False
    )
    plt.title(f"Factor Loadings Heatmap - {questionnaire_name}", fontsize=16, pad=12)
    plt.xlabel("Factors", fontsize=14)
    plt.ylabel("Items", fontsize=14)
    plt.xticks(ha="center", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path.joinpath(f'{questionnaire_name}_loadings.png'), dpi=300)
    plt.show()

    # 2. Factor Scores Scatter Plot (if at least 2 factors are available)
    if n_factors >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(df_loadings.iloc[:, 0], df_loadings.iloc[:, 1], alpha=0.7)
        plt.xlabel("Factor 1 Scores")
        plt.ylabel("Factor 2 Scores")
        plt.title(f"Scatter Plot of Factor Scores (Factors 1 vs 2) - {questionnaire_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 3. Hierarchical Clustering Dendrogram of Items Based on Factor Loadings
    # Using Ward's method for clustering
    Z = linkage(df_loadings, method='ward')
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(12, 8))
    dendrogram(
        Z,
        labels=df_loadings.index,
        orientation='top',  # Place the root at the top
        leaf_rotation=45,  # Rotate x-axis (leaf) labels
        leaf_font_size=12,  # Increase font size for leaf labels
        above_threshold_color='grey',
        # color_threshold=0.7 * max(Z[:, 2])  # Optionally set a threshold for color-coded clusters
    )
    plt.title(f"Hierarchical Clustering Dendrogram of Items - {questionnaire_name}", fontsize=16)
    plt.xlabel("Items", fontsize=14)
    plt.ylabel("Distance", fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12, rotation=45)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path.joinpath(f'{questionnaire_name}_dendrogram.png'), dpi=300)
    plt.show()


def compare_questionnaires(data,
                           col_ess,
                           col_sss,
                           aliases:Dict[str, str],
                           n_components=2,):
    """
    Compare two questionnaires (ESS and SSS) by assessing:
    - Convergent validity using total scores and Spearman correlation.
    - Construct mapping using Canonical Correlation Analysis (CCA).
    - Visualization of canonical variates and item correlation heatmap.

    Parameters:
    - data: pandas DataFrame containing questionnaire responses.
    - col_ess: list of column names for ESS items.
    - col_sss: list of column names for SSS items.
    - n_components: number of canonical components to extract (default=2).

    Returns:
    - cca: the fitted CCA model.
    - ess_cca: canonical variates for ESS items.
    - sss_cca: canonical variates for SSS items.
    """

    # --- 2. Canonical Correlation Analysis ---
    cca = CCA(n_components=n_components)
    cca.fit(data[col_ess], data[col_sss])
    ess_cca, sss_cca = cca.transform(data[col_ess], data[col_sss])

    df_loadings_ess = pd.DataFrame(data=cca.x_loadings_,
                                   index=col_ess,
                                   columns=[f'CCA_{idx}' for idx in range(0, n_components)])

    df_loadings_sss = pd.DataFrame(data=cca.y_loadings_,
                                   index=col_sss,
                                   columns=[f'CCA_{idx}' for idx in range(0, n_components)])

    df_loadings = pd.concat([df_loadings_ess, df_loadings_sss], axis=0)
    df_loadings.rename(index=aliases, inplace=True)
    df_loadings.sort_values(by='CCA_0', inplace=True)

    plt.figure(figsize=(8, 10))
    sns.heatmap(
        df_loadings,
        annot=True,            # Show numeric values in cells
        fmt=".2f",             # Format annotations to 2 decimal places
        cmap="Blues",          # Blue color palette
        # linewidths=0.5,        # Draw lines between cells
        # linecolor="white",
        yticklabels=df_loadings.index,
        annot_kws={"fontsize": 10},
        # cbar_kws={"shrink": 0.8, "aspect": 20},
        cbar=False
    )
    plt.title(f"Factor Loadings Heatmap", fontsize=16, pad=12)
    plt.xlabel("Factors", fontsize=14)
    plt.ylabel("Items", fontsize=14)
    plt.xticks(ha="center", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if output_path_cca:
        plt.savefig(output_path_cca.joinpath(f'cca_both_loadings.png'), dpi=300)
    plt.show()

    print("\nCanonical Coefficients for ESS (X) variables:")
    print(cca.x_loadings_)
    print("\nCanonical Coefficients for SSS (Y) variables:")
    print(cca.y_loadings_)

    # --- Plot: Scatter Plot of First Canonical Variates ---
    plt.figure(figsize=(8, 6))
    plt.scatter(ess_cca[:, 0], sss_cca[:, 0], alpha=0.7)
    plt.xlabel("ESS Canonical Variate 1")
    plt.ylabel("SSS Canonical Variate 1")
    plt.title("Scatter Plot of First Canonical Variates")
    plt.grid(True)
    plt.tight_layout()
    if output_path_cca:
        plt.savefig(output_path_cca.joinpath(f'cca_both_scatter_first_components.png'), dpi=300)
    plt.show()

    # --- 3. Correlation Heatmap ordinal responses ---
    combined_data = pd.concat([data[col_ess], data[col_sss]], axis=1)
    corr_matrix = combined_data.corr(method='spearman')
    # Exclude rows that are in col_sss or contain "score"
    indexes = [idx for idx in corr_matrix.index if (idx not in col_sss) and ("score" not in idx.lower())]

    # Exclude columns that are in col_ess or contain "score"
    columns = [col for col in corr_matrix.columns if (col not in col_ess) and ("score" not in col.lower())]

    filtered_corr_matrix = corr_matrix.loc[indexes, columns]
    filtered_corr_matrix.rename(index=aliases, inplace=True)
    filtered_corr_matrix.rename(columns=aliases, inplace=True)
    nrows, ncols = corr_matrix.shape
    fig_width = 10 # max(10, ncols * 1.2)  # Adjust multiplier as needed
    fig_height = 8 # max(8, nrows * 1.2)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        filtered_corr_matrix,
        annot=True,
        fmt=".2f",  # Annotation format to 2 decimal places
        cmap="coolwarm",
        center=0,
        annot_kws={"size": 12},
        cbar_kws={"shrink": 0.8},
        cbar=False
    )
    plt.title("Spearman Correlation Matrix: ESS & SSS Items", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=12, rotation_mode='anchor')
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if output_path_cca:
        plt.savefig(output_path_cca.joinpath(f'cca_both_corr_matrix.png'), dpi=300)
    plt.show()

    return cca, ess_cca, sss_cca


def cca_biplot(ess_loadings:np.ndarray,
               sss_loadings:np.ndarray,
               ess_labels:List[str],
               sss_labels:List[str],
               aliases:Dict[str,str],
               figsize:Tuple[int, int] = (16, 18),
               output_path:pathlib.Path = None):
    """
    Creates a biplot for CCA results, showing the first two canonical variates for ESS and SSS scores,
    along with arrows representing the loadings of the original variables.

    Parameters:
    - ess_scores: numpy array of ESS canonical variates (first two dimensions).
    - sss_scores: numpy array of SSS canonical variates (first two dimensions).
    - ess_loadings: numpy array of canonical loadings for ESS variables (from cca.x_loadings_).
    - sss_loadings: numpy array of canonical loadings for SSS variables (from cca.y_loadings_).
    - ess_labels: list of ESS variable names.
    - sss_labels: list of SSS variable names.
    """
    # Create DataFrames for loadings and add a column with the variable names.
    df_ess = pd.DataFrame(ess_loadings[:, :2],
                          index=ess_labels,
                          columns=['CC_0', 'CC_1'])
    df_ess['question'] = df_ess.index
    df_sss = pd.DataFrame(sss_loadings[:, :2],
                          index=sss_labels,
                          columns=['CC_0', 'CC_1'])
    df_sss['question'] = df_sss.index

    # Combine the loadings for ESS and SSS into one DataFrame.
    df_loadings = pd.concat([df_ess, df_sss])
    df_loadings['question'] = df_loadings['question'].map(aliases)
    # df_loadings.rename(index=aliases, inplace=True)
    df_loadings['number'] = range(1, df_loadings.shape[0]+1)


    plt.figure(figsize=figsize)
    legend_handles = []
    legend_labels = []
    sns.set_theme('notebook')
    # Iterate over each variable to plot arrows and labels.
    for idx, row in df_loadings.iterrows():
        txt = idx
        if 'score' in txt:
            color_vector = 'red'
        elif 'sss' in txt:
            color_vector = 'orange'
        else:
            color_vector = 'green'

        plt.arrow(x=0, y=0,
                  dx=row['CC_0'], dy=row['CC_1'],
                  color=color_vector,
                  alpha=0.5,
                  head_width=0.02, head_length=0.02, length_includes_head=False)
        plt.text(x=row['CC_0'] * 1.1,
                 y=row['CC_1'] * 1.1,
                 s=f" {row['number']} ",
                 # s=idx,
                 color=color_vector,
                 ha='center',
                 va='center',
                 fontsize=13)
        # Create a legend entry for this variable.
        handle = Line2D([0], [0], color=color_vector, lw=2)
        legend_handles.append(handle)
        legend_labels.append(f"{row['number']}: {row['question']}")


    # Draw a unit circle.
    circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

    # Adjust plot settings.
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.title('CCA Biplot for ESS & SSS Loadings', fontsize=20)
    plt.xlabel('Canonical Variate 1', fontsize=16)
    plt.ylabel('Canonical Variate 2', fontsize=16)
    plt.xlim([0, 1.2])
    plt.ylim([-1, 1])
    plt.grid(alpha=0.4, axis='both')
    plt.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, frameon=True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path.joinpath(f'cca_both_biplot.png'), dpi=300)
    plt.show()


# def pca_biplot(df_loadings:pd.DataFrame,
#                figsize:Tuple[int, int] = (16, 18),
#                output_path:pathlib.Path = None):
#     """
#     Creates a biplot for PCA results, showing the first two canonical variates for ESS and SSS scores,
#     along with arrows representing the loadings of the original variables.
#
#     Parameters:
#     - ess_scores: numpy array of ESS canonical variates (first two dimensions).
#     - sss_scores: numpy array of SSS canonical variates (first two dimensions).
#     - ess_loadings: numpy array of canonical loadings for ESS variables (from cca.x_loadings_).
#     - sss_loadings: numpy array of canonical loadings for SSS variables (from cca.y_loadings_).
#     - ess_labels: list of ESS variable names.
#     - sss_labels: list of SSS variable names.
#     """
#
#     plt.figure(figsize=figsize)
#     legend_handles = []
#     legend_labels = []
#     sns.set_theme('notebook')
#     # Iterate over each variable to plot arrows and labels.
#     for idx, row in df_loadings.iterrows():
#         txt = idx
#         if 'score' in txt:
#             color_vector = 'crimson'
#         elif 'sss' in txt:
#             color_vector = 'darkorange'
#         else:
#             color_vector = 'seagreen'
#
#         # Define line style based on latent state
#         linestyle = '--' if row.get('hidden_factors') == 'passive' else '-'
#
#
#         arrow = FancyArrowPatch(
#             (0, 0), (row['PC_0'], row['PC_1']),
#             arrowstyle='->',
#             color=color_vector,
#             linestyle=linestyle,
#             mutation_scale=15,
#             lw=2,
#             alpha=0.5
#         )
#         plt.gca().add_patch(arrow)
#
#         plt.text(x=row['PC_0'] * 1.1,
#                  y=row['PC_1'] * 1.1,
#                  s=f" {row['number']} ",
#                  # s=idx,
#                  color=color_vector,
#                  ha='center',
#                  va='center',
#                  fontsize=13)
#         # Create a legend entry for this variable.
#         handle = Line2D([0], [0], color=color_vector, lw=2)
#         legend_handles.append(handle)
#         legend_labels.append(f"{row['number']}: {row['question']}")
#
#
#     # Draw a unit circle.
#     circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
#     plt.gca().add_artist(circle)
#
#     # Adjust plot settings.
#     plt.gca().set_aspect('equal', adjustable='datalim')
#     plt.title('PCA Biplot for ESS & SSS Loadings', fontsize=20)
#     plt.xlabel('Principal Component 1', fontsize=16)
#     plt.ylabel('Principal Component 2', fontsize=16)
#     plt.xlim([0, 1.2])
#     plt.ylim([-1, 1])
#     plt.grid(alpha=0.4, axis='both')
#     plt.legend(legend_handles, legend_labels,
#                loc='upper left',
#                bbox_to_anchor=(1.05, 1),
#                fontsize=12,
#                frameon=True)
#     plt.tight_layout()
#     if output_path:
#         plt.savefig(output_path.joinpath(f'pca_both_biplot.png'), dpi=300)
#     plt.show()


def pca_biplot(df_loadings: pd.DataFrame,
               figsize: Tuple[int, int] = (16, 18),
               output_path: pathlib.Path = None):
    """
    Creates multiple biplots for all 2D combinations of PCA loadings.

    Parameters:
    - df_loadings: DataFrame containing PCA loadings with columns like 'PC_0', 'PC_1', etc.
    - figsize: Size of each biplot figure.
    - output_path: Optional path to save the figures.
    """
    sns.set_theme(style='whitegrid')
    pc_columns = [col for col in df_loadings.columns if col.startswith('PC_')]
    pc_indices = [int(col.split('_')[1]) for col in pc_columns]
    pc_indices = sorted(set(pc_indices))

    for pc_x, pc_y in combinations(pc_indices, 2):
        pc_x_col = f'PC_{pc_x}'
        pc_y_col = f'PC_{pc_y}'

        # Compute dynamic limits with margin
        margin = 0.2  # 20% margin
        x_vals = df_loadings[pc_x_col]
        y_vals = df_loadings[pc_y_col]

        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()

        x_margin = (x_max - x_min) * margin
        y_margin = (y_max - y_min) * margin

        plt.figure(figsize=figsize)
        ax = plt.gca()
        legend_handles = []
        legend_labels = []

        for idx, row in df_loadings.iterrows():
            txt = idx
            if 'score' in txt:
                color_vector = 'crimson'
            elif 'sss' in txt:
                color_vector = 'darkorange'
            else:
                color_vector = 'seagreen'

            linestyle = '--' if row.get('hidden_factors') == 'passive' else '-'

            arrow = FancyArrowPatch(
                posA=(0, 0),
                posB=(row[pc_x_col], row[pc_y_col]),
                arrowstyle='->',
                color=color_vector,
                linestyle=linestyle,
                mutation_scale=15,
                lw=2,
                alpha=0.5
            )
            ax.add_patch(arrow)
            ax.text(x=row[pc_x_col] * 1.1,
                    y=row[pc_y_col] * 1.1,
                    s=f" {row['number']} ",
                    color=color_vector,
                    ha='center',
                    va='center',
                    fontsize=13)

            handle = Line2D([0], [0], color=color_vector, lw=2)
            legend_handles.append(handle)
            legend_labels.append(f"{row['number']}: {row['question']}")

        # Add unit circle
        circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--', alpha=0.4)
        ax.add_artist(circle)

        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f'PCA Biplot: PC{pc_x + 1} vs PC{pc_y + 1}', fontsize=20)
        ax.set_xlabel(f'Principal Component {pc_x + 1}', fontsize=16)
        ax.set_ylabel(f'Principal Component {pc_y + 1}', fontsize=16)
        ax.grid(alpha=0.4)
        ax.set_xlim([x_min - x_margin, x_max + x_margin])
        ax.set_ylim([y_min - y_margin, y_max + y_margin])
        plt.legend(legend_handles, legend_labels,
                   loc='upper left', bbox_to_anchor=(1.05, 1),
                   fontsize=12, frameon=True)
        plt.tight_layout()
        if output_path:
            filename = f'pca_biplot_PC{pc_x + 1}_vs_PC{pc_y + 1}.png'
            plt.savefig(output_path / filename, dpi=300)
        plt.show()


def plot_pca_heatmap(df_loadings: pd.DataFrame,
                    figsize=(14, 12),
                    output_path: pathlib.Path = None,
                    cmap='vlag'):
    """
    Plots a heatmap of PCA loadings with row color coding based on:
    - Source system (ESS or SSS)
    - Latent state (active, passive)
    - Annotates each cell with the actual loading value
    - Colors the y-axis tick labels using system_palette
    """

    # Select PCA columns
    pca_columns = [col for col in df_loadings.columns if col.startswith('PC_')]
    loadings = df_loadings[pca_columns]

    # --------------------------------------
    # ✅ Method 1: Gower Distance + Average Linkage
    # --------------------------------------
    # Use this if:
    # - Your original data includes ordinal or mixed types
    # - You want to respect the non-metric nature of ordinal values
    # - You're clustering PCA loadings derived from such data
    # Gower computes normalized dissimilarities that respect ordinal scales.
    # Average linkage balances compactness and chaining.

    # distance_matrix = gower.gower_matrix(loadings)
    # condensed = squareform(distance_matrix, checks=False)
    # linkage_matrix = linkage(condensed, method='average')

    # --------------------------------------
    # ✅ Method 2: Euclidean Distance + Ward Linkage
    # --------------------------------------
    # Use this if:
    # - Your data is fully numeric and standardized (like PCA loadings)
    # - You want compact, spherical clusters
    # - Ward linkage minimizes within-cluster variance — ideal for numeric spaces

    # Alternative approach:
    condensed = pdist(loadings.values, metric='euclidean')
    linkage_matrix = linkage(condensed, method='ward')

    # --------------------------------------
    # Recommendation:
    # - If you're respecting ordinal properties → use Method 1
    # - If you're treating PCA loadings purely as numeric features → Method 2


    # Define palettes
    system_palette = {'ESS': '#66c2a5', 'SSS': '#fc8d62', 'Other': '#8da0cb'}
    state_palette = {'passive': '#a6d854', 'active': '#ffd92f', 'neutral': '#e5c3ff'}

    # Create row color annotations
    row_colors = pd.DataFrame({
        'System': df_loadings['system'].map(system_palette),
        'State': df_loadings['state'].map(state_palette)
    }, index=df_loadings.index)

    # Custom yticklabels with questions and system colors
    row_labels = [f"{row['question']} | {idx}" for idx, row in df_loadings.iterrows()]
    label_colors = df_loadings['system'].map(system_palette).values

    # Plot heatmap with annotations
    sns.set_theme(style='white')
    g = sns.clustermap(loadings,
                       # metric='correlation',  # correlation → 1 - Pearson
                       # method='average',  # linkage method
                       row_linkage=linkage_matrix,
                       row_colors=row_colors,
                       figsize=figsize,
                       cmap=cmap,
                       annot=True,
                       fmt=".2f",
                       annot_kws={"size": 8},
                       linewidths=0.5,
                       xticklabels=True,
                       yticklabels=row_labels,
                       dendrogram_ratio=(.1, .2),
                       cbar_pos=None,
                       # cbar_pos=(0.02, 0.8, 0.05, 0.18),
                       )

    # Top legend (system and state)
    system_legend = [Patch(facecolor=color, label=f'System: {label}')
                     for label, color in system_palette.items()]
    state_legend = [Patch(facecolor=color, label=f'State: {label}')
                    for label, color in state_palette.items()]
    g.ax_col_dendrogram.legend(handles=system_legend + state_legend,
                               loc='center',
                               ncol=2,
                               bbox_to_anchor=(0.5, 1.2),
                               fontsize=9,
                               frameon=False)

    # Re-color ytick labels
    ax = g.ax_heatmap
    for tick_label, color in zip(ax.get_yticklabels(), label_colors[g.dendrogram_row.reordered_ind]):
        tick_label.set_color(color)
        tick_label.set_fontweight('bold')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path.joinpath('clustermap.png'), dpi=300)
    plt.show()


def plot_pca_dendrogram(df_loadings: pd.DataFrame,
                        figsize:Tuple[int, int]=(10, 12),
                        output_path:pathlib.Path=None):
    """
    Plots only the dendrogram of the PCA loading clustering, with color-coded labels
    based on the source system and latent state.

    The x-axis represents the distance (or dissimilarity) between clusters at each merge step during hierarchical
    clustering.

    If we used metric='euclidean' and method='ward' then,
    the x-axis shows the within-cluster variance at each merge step.


    Parameters:
    - df_loadings: DataFrame including PCA loadings and metadata columns 'system' and 'state'
    - figsize: size of the figure
    - method: linkage method (e.g., 'ward', 'average')
    - metric: distance metric (e.g., 'euclidean', 'correlation')
    """
    # Select PCA columns and remove the scores
    pca_columns = [col for col in df_loadings.columns if col.startswith('PC_')]
    ordinal_vars = [idx for idx in df_loadings.index if not 'score' in idx]
    loadings = df_loadings.loc[df_loadings.index.isin(ordinal_vars), pca_columns]

    metric_method_combos = [
        ('euclidean', 'ward'),  # ✅ Best for numeric/PCA (compact clusters)
        ('euclidean', 'average'),  # More flexible, still geometric
        ('euclidean', 'complete'),  # Strong separation
        ('euclidean', 'single'),  # Very sensitive to outliers (not ideal for PCA)
        ('cityblock', 'average'),  # Manhattan distance + average
        ('cosine', 'average'),  # Shape-based (angle between vectors)
        ('correlation', 'average'),  # 1 - Pearson correlation (for profile similarity)
        ('chebyshev', 'average'),  # Max coordinate difference
    ]
    # Define color palettes
    system_palette = {'ESS': '#66c2a5', 'SSS': '#fc8d62', 'Other': '#8da0cb'}
    state_palette = {'passive': '#a6d854', 'active': '#ffd92f', 'neutral': '#e5c3ff'}

    for metric, method in metric_method_combos:
        # Compute linkage matrix
        condensed = pdist(loadings.values, metric=metric)
        linkage_matrix = linkage(condensed, method=method)

        # Get reordered labels based on clustering
        dendro = dendrogram(linkage_matrix, labels=loadings.index.tolist(), no_plot=True)
        ordered_labels = dendro['ivl']

        # Build color-coded labels
        system_colors = df_loadings.loc[ordered_labels, 'system'].map(system_palette)
        state_colors = df_loadings.loc[ordered_labels, 'state'].map(state_palette)
        label_texts = [f"{df_loadings.loc[idx, 'question']} | {idx}" for idx in ordered_labels]

        # Plot dendrogram
        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(linkage_matrix,
                   labels=label_texts,
                   orientation='right',
                   leaf_font_size=9,
                   leaf_rotation=0,
                   ax=ax)

        # Recolor tick labels by system
        tick_labels = ax.get_yticklabels()
        for tick, sys_color in zip(tick_labels, system_colors):
            tick.set_color(sys_color)
            tick.set_fontweight('bold')

        # Add legends for system and state
        system_legend = [Patch(color=color, label=f'System: {key}') for key, color in system_palette.items()]
        state_legend = [Patch(color=color, label=f'State: {key}') for key, color in state_palette.items()]
        ax.legend(handles=system_legend + state_legend, loc='lower right', fontsize=8, frameon=False)

        ax.set_title(f"Dendrogram (Metric: {metric}, Method: {method})", fontsize=13)
        plt.tight_layout()
        plt.grid()
        if output_path:
            plt.savefig(output_path.joinpath(f'dendrogram_{metric}_{method}.png'), dpi=300)
        plt.show()


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
    questions_score = col_ess + col_sss
    cmp_heatmap = sns.color_palette("Spectral_r", as_cmap=True)

    df_questions = df_data[questions_score].copy()  # Ensure no NaN values for analysis
    output_path_cca = config.get('results_path').joinpath('canonical_corr')
    if not output_path_cca.exists():
        output_path_cca.mkdir(parents=True)

    output_path_pca = config.get('results_path').joinpath('principal_comp')
    if not output_path_pca.exists():
        output_path_pca.mkdir(parents=True)


    # %% Get the labels
    ess_quest = EpworthScale()
    sit_quest = SituationalSleepinessScale()

    ess_labels = ess_quest.get_labels()
    sss_labels = sit_quest.get_labels()

    sss_factors = sit_quest.get_latent_factor()
    ess_factors = ess_quest.get_latent_factor()

    aliases = ess_labels.copy()
    aliases.update(sss_labels)

    factors_lbls = sss_factors.copy()
    factors_lbls.update(ess_factors)

    # %% Exploratory Factor Analysis (EFA)
    # --- ESS Analysis ---
    fa_ess, eigenvalues_ess, df_loadings_ess = perform_efa_analysis(data=df_questions,
                                                   columns=list(ess_labels.keys()),
                                                   questionnaire_name="ESS")
    plot_factor_visualizations(df_loadings=df_loadings_ess.rename(index=ess_labels),
                               n_factors=2,
                               questionnaire_name="ESS",
                               output_path=output_path_cca)

    # --- SSS Analysis ---
    fa_sss, eigenvalues_sss, df_loadings_sss = perform_efa_analysis(df_questions,
                                                   columns=list(sss_labels.keys()),
                                                   questionnaire_name="SSS")

    plot_factor_visualizations(df_loadings=df_loadings_sss.rename(index=sss_labels),
                               n_factors=2,
                               questionnaire_name="SSS")

    # %% CCA
    cca_model, ess_canonical, sss_canonical = compare_questionnaires(data=df_questions,
                                                                     col_ess=col_ess,
                                                                     col_sss=col_sss,
                                                                     aliases=aliases,
                                                                     n_components=4)

    cca_biplot(ess_loadings=cca_model.x_loadings_,
               sss_loadings=cca_model.y_loadings_,
               ess_labels=col_ess,
               sss_labels=col_sss,
               aliases={key:val.replace('\n', ' ') for key, val in aliases.items()},
               figsize=(10,8),
               output_path=output_path_cca
               )
    # %% PCA
    # Explore all the factors to determine number of components
    # Factor Loadings: How well do the questions align with underlying constructs?
    # pca_interpretation(frame=df_data,
    #                    columns=questions_score,
    #                    figsize=(12, 12),
    #                    output_path=output_path_pca,
    #                    file_name='all_questions',
    #                    n_components=3)

    df_pca, df_loadings = pca_interpretation(frame=df_data,
                                             columns=questions_score,
                                             figsize=(12, 12),
                                             n_components=4,
                                             file_name='all_questions',
                                             output_path=output_path_pca,
                                             plot=True)

    df_loadings['question'] = df_loadings.index.map(aliases)
    df_loadings['question'] = df_loadings['question'].apply(lambda x: x.replace('\n', ' '))
    df_loadings['number'] = range(1, df_loadings.shape[0]+1)
    df_loadings['hidden_factors'] = df_loadings.index.map(factors_lbls)
    # df_loadings.sort_values(by='PC_0', ascending=False, inplace=True)


    pca_biplot(df_loadings=df_loadings,
                   figsize=(10, 8),
                   output_path=output_path_pca)


    # plot relationship between rows and columns using
    plot_pca_heatmap(df_loadings=df_loadings,
                     figsize=(10, 8),
                     output_path=output_path_pca)

    # PC_0	General Sleepiness Propensity	Tendency to fall asleep in any context
    # PC_1	Contextual Activeness	Sleepiness in active vs. passive states
    # PC_2	Cognitive or Executive Load	Struggling to stay awake in mentally active tasks
    # PC_3	Social/passive nuance	Situational nuances, especially social/passive

    plot_pca_dendrogram(df_loadings=df_loadings,
                        figsize=(8, 6),
                        output_path=output_path_pca)









