import pathlib
import pandas as pd
import statsmodels.api as sm
import numpy as np
from config.config import config, mapper, mapper_diagnosis
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import seaborn as sns
from questionnaires.questionnaires import SituationalSleepinessScale
import re
from utils.functions import osa_categories

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """sort the ESS and SSS columns based on the last integer ant not the  lexicographically order"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]
def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    leading_columns = ['study_id', 'bmi', 'gender', 'race']
    other_columns = sorted([col for col in df.columns if col not in leading_columns])
    sorted_columns = leading_columns + other_columns
    df = df[sorted_columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


if __name__ == '__main__':
    # %% rename columns
    df_raw = pd.read_excel(config.get('data_raw_path').get('first_cross_sectional'),
                           sheet_name='PpData2')
    mapper_columns = mapper.get('standardization')
    df_raw.columns = map(str.strip, df_raw.columns)
    df_raw.rename(columns=mapper_columns, inplace=True)
    df_raw.columns = map(str.lower, df_raw.columns)
    mapper_columns = {key: val for key, val in mapper_columns.items() if val in df_raw.columns}

    # %% select columns of interest and remove nans rows
    col_extra = ['race', 'ethnicity']
    col_sss = [col for col in df_raw.columns if 'sss' in col and ' ' not in col]
    col_ess = [col for col in df_raw.columns if 'ess' in col and len(col) < 10]
    col_interest = [*mapper_columns.values()] + col_sss + col_ess + col_extra
    df_raw = df_raw[col_interest]
    df_raw.dropna(how='all', inplace=True)
    df_raw.dropna(subset=['study_id'], inplace=True)
    df_raw[col_ess] = df_raw[col_ess].astype(int)
    # %% convert columns in proper format
    df_raw['date'] = pd.to_datetime(df_raw['date'],
                                    errors='coerce')
    df_raw['gender'] = df_raw['gender'].map(mapper.get('gender'))
    df_raw['race'] = df_raw['race'].map(mapper.get('race'))
    df_raw['ethnicity'].replace({'Non hispanic': 'Non-hispanic',
                                 'hispanic':'Hispanic'},
                                inplace=True)
    df_raw['ethnicity'] = df_raw['ethnicity'].map(mapper.get('ethnicity'))
    df_raw['study_id'] = df_raw['study_id'].astype(int)
    df_raw['record_id'] = df_raw['record_id'].astype(int)
    # age
    df_raw['age'] = df_raw['date'].dt.year - df_raw['dob'].dt.year
    # bmi
    df_raw['bmi'] = pd.to_numeric(df_raw['bmi'], errors='coerce')
    # %%
    df_raw.sort_values(by='study_id', inplace=True)
    df_raw.drop(columns=['record_id'],
                inplace=True)
    df_raw.replace('.', np.nan, inplace=True)
    # sort columns alphabetically
    df_raw = sort_columns(df=df_raw)
    # %% convert the 15 minutes responses to the 30 minutes responses
    # the question sss10 should have ask in a 30 min interval not 15
    df_raw['sss10_30min'].mean()
    ratio_sss10 = df_raw['sss10_30min'].mean()/df_raw['sss10_15min'].mean()
    df_raw['sss_30_min_derived'] = np.nan
    df_raw.loc[df_raw['sss10_30min'].isna(), 'sss_30_min_derived'] = df_raw['sss10_15min']*ratio_sss10
    df_raw['sss_30_min_derived'] = df_raw['sss_30_min_derived'].round(1)
    # df_raw[['sss_30_min_derived', 'sss10_30min', 'sss10_15min']]


    def mapping_sss10(row) -> int:
        if pd.isna(row['sss10_30min']):
            if pd.notna(row['sss10_15min']) and 0 < row['sss10_15min'] < 3:
                return row['sss10_15min'] + 1
            else:
                return row['sss10_15min']
        else:
            return row['sss10_30min']


    # Apply the mapping function to create a new column 'sss10_2'
    df_raw['sss10'] = df_raw.apply(mapping_sss10, axis=1)

    # df_raw[['sss_30_min_derived', 'sss10_30min', 'sss10_15min', 'sss10']]
    PLOT = False
    if PLOT:
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        for i, col in enumerate(['sss10_15min', 'sss10_30min', 'sss10']):
            sns.histplot(df_raw[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Density')
            axes[i].grid(alpha=0.7)
        plt.tight_layout()
        plt.show()

    # Perform Kolmogorov-Smirnov test for pairwise comparisons
    col1 = df_raw['sss10_15min'].dropna().astype(int)
    col2 = df_raw['sss10_30min'].dropna().astype(int)
    statistic, p_value = mannwhitneyu(col1.values, col2.values)
    print(f'Mann-U test {col1.name} ({col1.shape[0]}) and {col2.name}({col2.shape[0]}): '
          f'Statistic={statistic}, p-value={p_value}')

    col1 = df_raw['sss10_15min'].dropna().astype(int)
    col2 = df_raw['sss10'].dropna().astype(int)
    statistic, p_value = mannwhitneyu(col1, col2)
    print(f'Mann-U test {col1.name} ({col1.shape[0]}) and {col2.name}({col2.shape[0]}): '
          f'Statistic={statistic}, p-value={p_value}')

    col1 = df_raw['sss10_30min'].dropna().astype(int)
    col2 = df_raw['sss10'].dropna().astype(int)
    statistic, p_value = mannwhitneyu(col1, col2)
    print(f'Mann-U test {col1.name} ({col1.shape[0]}) and {col2.name}({col2.shape[0]}): '
          f'Statistic={statistic}, p-value={p_value}')
    df_raw.drop(columns=['sss10_15min', 'sss10_30min', 'sss_30_min_derived'],
                inplace=True)

    # %% SSS replace nan with not applicable
    col_sss = [col for col in df_raw.columns if 'sss' in col and ' ' not in col]
    df_raw[col_sss] = df_raw[col_sss].fillna(value=4,)
    # %% re-compute the scores for the SSS
    sit_quest = SituationalSleepinessScale()
    sss_score = [score for score in col_sss if 'score' in score]
    # sss_score[0]: sss_score
    # sss_score[1]: sss_score_div_num_quest 
    df_raw[sss_score[0]] = sit_quest.compute_score(responses=df_raw[col_sss])
    df_raw[sss_score[1]] = df_raw[sss_score[0]]/10
    if not 'ess_score_div_num_quest' in df_raw.columns:
        col_ess = [col for col in df_raw.columns if not col != 'ess_score' and 'ess' in col]
        df_raw['ess_score_div_num_quest'] = df_raw['ess_score'] / len(col_ess)
    # %% clean the diagnosis columns with strings
    col_diagnosis = ['insomnia', 'narc_level', 'rls', 'rmeq']
    df_raw['insomnia'].unique()
    mapping_insomnia = {'yes': 1, 'no': 0, 1: 1, 0: 0}
    df_raw['insomnia'] = df_raw['insomnia'].map(mapping_insomnia)

    df_raw['narc_level'].unique()
    mapping_narc_level = {'1 or 3': '1-3',
                          '2 or 3': '2-3',
                          0: 0,
                          1: 1,
                          2: 2,
                          3: 3}
    df_raw['narc_level'] = df_raw['narc_level'].map(mapping_narc_level)
    # set them all as numeric
    df_raw['narc_level'] = df_raw['narc_level'].map(mapper_diagnosis.get('narc_level'))

    df_raw['rls'].unique()
    mapping_rls = {'yes': 1,
                   'no': 0,
                   1: 1,
                   0: 0}
    df_raw['rls'] = df_raw['rls'].map(mapping_rls)

    df_raw['rmeq'].unique()
    mapping_rls = {'yes': 1,
                   'no': 0,
                   1: 1,
                   0: 0}
    df_raw['rls'] = df_raw['rls'].map(mapping_rls)

    df_raw['osa_level_ahi_3per'] = df_raw['ahi_3per'].map(osa_categories)
    df_raw['osa_level_ahi_4per'] = df_raw['ahi_4per'].apply(osa_categories)

    df_raw['osa_level_odi_3per'] = df_raw['odi_3per'].map(osa_categories)
    df_raw['osa_level_odi_4per'] = df_raw['odi_4per'].apply(osa_categories)


    # %% sort the questionnaire columns
    # df_raw.reset_index(inplace=True, drop=True)
    # df_raw = sort_columns(df=df_raw)
    # # sort the questionnaire columns
    # sorted_col_sss = sorted(col_sss, key=natural_sort_key)
    # sorted_col_ess = sorted(col_ess, key=natural_sort_key)
    # # get the ess and sss in a sorted dataframe
    # df_sss = df_raw[sorted_col_sss].copy()
    # # df_sss[sss_score[1]] = df_sss[sss_score[0]]/10
    # df_ess = df_raw[sorted_col_ess].copy()
    # # drop from original the non-organized version
    # df_raw.drop(columns=col_sss + col_ess + [sss_score[1], 'ess_score_div_num_quest'], inplace=True)
    # # aggregate the organized questionnaires
    # df_raw = pd.concat([df_raw, df_ess, df_sss], axis=1)
    # %% save
    df_raw.to_csv(config.get('data_pp_path').get('first_cross_sectional'),
                  index=False)