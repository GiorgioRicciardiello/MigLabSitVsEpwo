import pathlib
import pandas as pd
from typing import Union, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def osa_categories(ahi):
    """Categorize AHI value into OSA severity levels."""
    ahi_range = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    if pd.isna(ahi):
        return np.nan
    elif ahi < 5:
        return ahi_range['Normal']
    elif 5 <= ahi < 15:
        return ahi_range['Mild']
    elif 15 <= ahi < 30:
        return ahi_range['Moderate']
    else:
        return ahi_range['Severe']


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
    questions_scaled = scaler.fit_transform(frame)
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


def cronbach_alpha(items_scores):
    items_scores = np.array(items_scores)
    item_vars = items_scores.var(axis=1, ddof=1)
    total_var = items_scores.sum(axis=0).var(ddof=1)
    n_items = items_scores.shape[0]
    return n_items / (n_items - 1) * (1 - item_vars.sum() / total_var)