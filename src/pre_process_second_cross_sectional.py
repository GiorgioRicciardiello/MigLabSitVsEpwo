
# TODO: We will have a new data split to mege, the first is de-identifed and we have repeated data so we check thata
# TODO: Merge the two into a single one
# TODO: comptue teh ss metrics of the new columns we created
# TODO>

import pathlib
import pandas as pd
import statsmodels.api as sm
import numpy as np
from config.config import config, mapper, mapper_diagnosis
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import seaborn as sns
from questionnaires.questionnaires import SituationalSleepinessScale, EpworthScale
import re
from utils.functions import osa_categories

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """sort the ESS and SSS columns based on the last integer ant not the  lexicographically order"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]
def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    leading_columns = ['study_id', 'age', 'bmi', 'gender', 'race', 'dob']
    other_columns = sorted([col for col in df.columns if col not in leading_columns])
    sorted_columns = leading_columns + other_columns
    df = df[sorted_columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


if __name__ == '__main__':
    # %% rename columns
    df_raw = pd.read_excel(config.get('data_raw_path').get('second_cross_sectional'),
                           sheet_name='split 2- raw')
    df_raw = df_raw.loc[df_raw['Duplicate'].isna()]
    df_raw.reset_index(inplace=True, drop=True)
    mapper_columns = mapper.get('standardization')
    df_raw.columns = map(str.strip, df_raw.columns)
    df_raw.rename(columns=mapper_columns, inplace=True)
    df_raw.columns = map(str.lower, df_raw.columns)
    df_raw.drop(columns=df_raw.columns[df_raw.columns.str.contains('unnamed')], inplace=True)

    col_diagnosis = ['osa_level', 'narc_level', 'rls', 'insomnia',
                     'ahi_3per', 'ahi_4per', 'odi_3per', 'odi_4per',
                     'per_stage_one', 'rem_lat_min', 'sleep_lat_min',
                     'sleep_eff_perc', 'plmi', 'rmeq', 'sleep study date']
    df_raw['narc_level'].value_counts()

    # # df_duplicates = df_raw[
    # #     df_raw.apply(lambda row: row.astype(str).str.contains('Duplicate', case=False, na=False).any(), axis=1)]
    # mask_duplicates = df_raw.apply(lambda row: row.astype(str).str.contains('Duplicate', case=False, na=False).any(), axis=0)
    # print(f'Dropping duplicate rows: {mask_duplicates.sum()}')
    #
    # mask_duplicates = df_raw.apply(lambda row: row.astype(str).str.contains('Duplicate', case=False, na=False).any(), axis=1)
    # df_raw = df_raw[[~mask_duplicates]]


    # df = df_raw[~mask_duplicates]
    mapper_columns = {key:val for key, val in mapper_columns.items() if val in df_raw.columns}
    # %% select columns of interest and remove nans rows
    col_race = [col for col in df_raw.columns if 'race__' in col and ' ' not in col]
    col_sss = [col for col in df_raw.columns if 'sss' in col and ' ' not in col]
    col_ess = [col for col in df_raw.columns if 'ess' in col and len(col) < 10]
    col_interest = [*mapper_columns.values()] + col_race + col_sss + col_ess + col_diagnosis
    col_interest = sorted(set(col_interest))
    df_raw = df_raw[col_interest]
    df_raw.replace('.',  np.nan, inplace=True)
    # df_raw.replace({np.nan: np.nan}, inplace=True)
    df_raw.dropna(how='all', inplace=True)
    df_raw.dropna(subset=col_ess, inplace=True)
    df_raw[col_ess] = df_raw[col_ess].astype(int)
    # %% convert columns in proper format
    df_raw['dob'] = pd.to_datetime(df_raw['dob'],
                                    errors='coerce')
    # df_raw['gender'] = df_raw['gender'].map(mapper.get('gender'))
    # Create a dictionary mapping each column to its corresponding race number
    race_mapping = {col: int(col.split('___')[-1]) for col in col_race}
    # Multiply each dummy column by its race number and sum across columns
    df_raw['race'] = df_raw[col_race].dot(pd.Series(race_mapping))
    df_raw['race'].replace({99:np.nan},
                           inplace=True)
    df_raw.drop(columns=[*race_mapping.keys()], inplace=True)
    # df_raw['race'] = df_raw['race'].map(mapper.get('race'))
    df_raw['record_id'] = df_raw['record_id'].astype(int)
    # age
    df_raw['age'] = df_raw['date_consent'].dt.year - df_raw['dob'].dt.year
    # bmi
    df_raw['bmi'] = pd.to_numeric(df_raw['bmi'], errors='coerce')
    # %%
    df_raw['study_id'] = df_raw['record_id']
    df_raw.sort_values(by='study_id', inplace=True)
    df_raw.replace('.', np.nan, inplace=True)
    # sort columns alphabetically
    df_raw = sort_columns(df=df_raw)
    df_raw.drop(columns=['record_id', 'dob'],
                inplace=True)
    # %% compute SS10
    df_raw.rename(columns={'sss10_30min': 'sss10'}, inplace=True)
    df_raw.drop(columns=['sss10_15min'],
                inplace=True)
    # %% SSS replace nan with not applicable
    col_sss = [col for col in df_raw.columns if 'sss' in col and ' ' not in col]
    df_raw[col_sss] = df_raw[col_sss].fillna(value=4)
    # %% re-compute the scores for the SSS
    sit_quest = SituationalSleepinessScale()
    df_raw['sss_score'] = sit_quest.compute_score(responses=df_raw[col_sss])
    df_raw['sss_score_div_num_quest'] = df_raw['sss_score']/len(col_sss)
    # %% ESS score
    col_ess = [col for col in df_raw.columns if 'ess' in col and ' ' not in col]
    ess_score = EpworthScale()
    df_raw['ess_score'] = ess_score.compute_score(responses=df_raw[col_ess])
    df_raw['ess_score_div_num_quest'] = df_raw['ess_score']/len(col_ess)
    # %% clean the diagnosis columns with strings
    col_diagnosis = ['insomnia', 'narc_level', 'rls', 'rmeq']
    df_raw['insomnia'].unique()
    mapping_insomnia = {'yes': 1, 'no': 0, 1: 1, 0: 0}
    df_raw['insomnia'] = df_raw['insomnia'].replace(mapping_insomnia)

    df_raw['narc_level'].unique()

    df_raw['rls'].unique()

    df_raw['rmeq'].unique()


    df_raw['osa_level_ahi_3per'] = df_raw['ahi_3per'].map(osa_categories)
    df_raw['osa_level_ahi_4per'] = df_raw['ahi_4per'].apply(osa_categories)

    df_raw['osa_level_odi_3per'] = df_raw['odi_3per'].map(osa_categories)
    df_raw['osa_level_odi_4per'] = df_raw['odi_4per'].apply(osa_categories)


    # %% sort the questionnaire columns
    df_raw.reset_index(inplace=True, drop=True)
    # sort the questionnaire columns
    sorted_col_sss = sorted(col_sss, key=natural_sort_key)
    sorted_col_ess = sorted(col_ess, key=natural_sort_key)
    # get the ess and sss in a sorted dataframe
    # df_sss = df_raw[sorted_col_sss].copy()
    df_raw['sss_score_div_num_quest'] = df_raw['sss_score']/10
    # drop from original the non-organized version
    # %% save
    df_raw.to_csv(config.get('data_pp_path').get('second_cross_sectional'),
                  index=False)

    counts_dict = {
        'osa_level': df_raw['osa_level'].value_counts(),
        'insomnia': df_raw['insomnia'].value_counts(),
        'narc_level': df_raw['narc_level'].value_counts(),
        'rls': df_raw['rls'].value_counts()
    }

    # Combine into a DataFrame, filling missing values with 0 and converting to integers
    df_counts = pd.DataFrame(counts_dict).fillna(0).astype(int)
    print(df_counts)

















