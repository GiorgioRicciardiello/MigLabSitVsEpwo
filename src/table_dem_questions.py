import pathlib
import pandas as pd
import statsmodels.api as sm
from typing import Union, Optional, Tuple
import numpy as np
from config.config import config, mapper
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
import seaborn as sns
import re
from questionnaires.questionnaires import SituationalSleepinessScale, EpworthScale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

col_dem = ['study_id', 'bmi', 'race', 'age']


def plot_responses(frame: pd.DataFrame,
                   columns: list[str],
                   title_global: str = 'Plotting Responses'):
    """Plot the distribution responses of each questionnaire in the same figure"""
    ncols = 3
    nrows = len(columns) // ncols + (len(columns) % ncols > 0)
    plt.figure(figsize=(ncols * 5, nrows * 4))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(title_global,
                 y=0.95)
    for n, col in enumerate(columns):
        ax = plt.subplot(nrows, nrows, n + 1)
        if 'score' in col:
            sns.kdeplot(frame[col],
                        ax=ax,
                        fill=True)
        else:
            sns.histplot(frame[col],
                         kde=True,
                         ax=ax,
                         discrete=True)
        ax.grid(True)
        ax.set_title(f'{col.capitalize()}')
    plt.tight_layout()
    plt.show()


def categorical_var(frame: pd.DataFrame,
                    col: str,
                    category:Optional[str] = None,
                    decimal: Optional[int] = 2,
                    ) -> Union[str, list]:
    """
    Count the number of occurrences in a column, giving he number of evens and the percentage. Used for category columns

    :param frame: dataframe from there to compute the count on the columns
    :param col: column to compute the calculation of the count
    :param category: if we want to count a specific category of the categories
    :param decimal: decimal point to show in the table
    :return:
    """
    if category is not None:
        count = frame.loc[frame[col] == category, col].shape[0]
        cell = f'{count} ({np.round((count / frame.shape[0]) * 100, decimal)}%)'
        return cell
    else:
        # return the count ordered by the index
        count = frame[col].value_counts()
        count = count.sort_index()
        if count.shape[0] == 1:
            # binary data so counting the ones
            cell = f'{count[1]} ({np.round((count[1] / frame.shape[0]) * 100, decimal)}%)'
            return cell
        else:
            cell = [f'{count_} ({np.round((count_ / frame.shape[0]) * 100, decimal)}%)' for count_ in count]
            return cell


def continuous_var(frame: pd.DataFrame,
                   col: str,
                   decimal: Optional[int] = 2) -> str:
    cell = f'{np.round(frame[col].mean(), decimal)} ({np.round(frame[col].std(), decimal)})'
    return cell


def create_index(frame: pd.DataFrame,
                 col: str,
                 prefix: str) -> list[str]:
    """
    Create the indexes that will be used as rows of table one based on the unique values of the the current column.
    We place a prefix to differentiate it from other similar-named rows
    :param frame:
    :param col:
    :param prefix:
    :return:
    """
    unique_vals = frame.dropna(subset=[col])[col].unique()
    unique_vals.sort()
    return [f'{prefix}_{i}' for i in unique_vals]

sns.set_context('talk')
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
    # %% define the questionnaire classes
    ess_quest = EpworthScale()
    sit_quest = SituationalSleepinessScale()
    # %% make table 1
    # Re-map numeric columns to categorical labels
    mapper_inv = {col: {v: k for k, v in mapper.get(col).items()} for col in ['gender', 'race', 'ethnicity']}
    for col in mapper_inv:
        df_data[col] = df_data[col].map(mapper_inv[col])

    index_gender = create_index(df_data, 'gender', 'Gender')
    index_race = create_index(df_data, 'race', 'Race')
    index_ethnicity = create_index(df_data, 'ethnicity', 'Ethn')

    # Combine indices
    index = ['age', 'bmi'] + index_gender + index_race + index_ethnicity

    # Initialize table
    tableone = pd.DataFrame(index=index, columns=['metric'])
    tableone.index.name = 'variable'
    tableone.reset_index(inplace=True)

    # Populate the table
    tableone.loc[tableone['variable'] == 'age', 'metric'] = continuous_var(df_data, 'age')
    tableone.loc[tableone['variable'] == 'bmi', 'metric'] = continuous_var(df_data, 'bmi')

    for gender in index_gender:
        category = gender.split('_')[1]
        tableone.loc[tableone['variable'] == gender, 'metric'] = categorical_var(df_data,
                                                                                 col='gender',
                                                                                 category=category)

    for race in index_race:
        category = race.split('_')[1]
        tableone.loc[tableone['variable'] == race, 'metric'] = categorical_var(df_data,
                                                                               col='race',
                                                                               category=category)

    for ethnicity in index_ethnicity:
        category = ethnicity.split('_')[1]
        tableone.loc[tableone['variable'] == ethnicity, 'metric'] = categorical_var(df_data,
                                                                                    col='ethnicity',
                                                                                    category=category)
    tableone.to_excel(config.get('results_path').joinpath('TableOne.xlsx'), index=False)
    # %% ESS Table 1
    tableone_ess = pd.DataFrame(np.nan,
                                columns=[*ess_quest.levels.keys()],
                                index=[col_ess])
    tableone_ess.reset_index(drop=False, names='Question', inplace=True)

    for question_ in tableone_ess['Question'].values:
        if question_[-1].isdigit():
            tableone_ess.loc[tableone_ess['Question'] == question_, tableone_ess.columns[1:]] = categorical_var(
                frame=df_data, col=question_)
            continue
        if 'score' in question_:
            tableone_ess.loc[tableone_ess['Question'] == question_, tableone_ess.columns[1:]] = continuous_var(
                frame=df_data, col=question_)

    # sort the table and save
    tableone_ess['sort_key'] = tableone_ess['Question'].apply(
        lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
    tableone_ess = tableone_ess.sort_values('sort_key').drop('sort_key', axis=1)
    tableone_ess.to_excel(config.get('results_path').joinpath('TableEpworth.xlsx'), index=False)

    # %% SSS Table 1
    tableone_sit_quest = pd.DataFrame(np.nan,
                                      columns=[*sit_quest.levels.keys()],
                                      index=[col_sss])
    tableone_sit_quest.reset_index(drop=False, names='Question', inplace=True)

    for question_ in tableone_sit_quest['Question'].values:
        if question_[-1].isdigit():
            # for sss6 non answered 4, so we need to select until the column high chance
            responses = categorical_var(frame=df_data, col=question_)
            tableone_sit_quest.loc[tableone_sit_quest['Question'] == question_, tableone_sit_quest.columns[
                                                                                1:len(responses) + 1]] = responses
            continue
        if 'score' in question_:
            tableone_sit_quest.loc[
                tableone_sit_quest['Question'] == question_, tableone_sit_quest.columns[1:6]] = continuous_var(
                frame=df_data, col=question_)
    # sort the table and save
    tableone_sit_quest['sort_key'] = tableone_sit_quest['Question'].apply(
        lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
    tableone_sit_quest = tableone_sit_quest.sort_values('sort_key').drop('sort_key', axis=1)
    tableone_sit_quest.to_excel(config.get('results_path').joinpath('TableOneSitSS.xlsx'), index=False)
    # %% Visualization of the responses
    plot_responses(frame=df_data,
                   columns=col_ess,
                   title_global='Responses Epworth Sleep Scale')

    plot_responses(frame=df_data,
                   columns=col_ess,
                   title_global='Responses Situational Sleep Scale')

