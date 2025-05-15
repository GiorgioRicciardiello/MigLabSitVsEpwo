
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
    leading_columns = ['study_id', 'record_id', 'full_name', 'age', 'bmi', 'gender', 'race', 'dob']
    leading_columns = [col for col in leading_columns if col in df.columns]
    other_columns = sorted([col for col in df.columns if col not in leading_columns])
    sorted_columns = leading_columns + other_columns
    df = df[sorted_columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


if __name__ == '__main__':
    # %% rename columns
    # df_raw = pd.read_excel(config.get('data_raw_path').get('second_cross_sectional'),
    #                        sheet_name='split 2- raw')

    df_raw = pd.read_excel(config.get('data_raw_path').get('both_cross_sectional'),
                           sheet_name='feb 2025')

    df_raw = df_raw.loc[df_raw['Duplicate'].isna()]
    df_raw.reset_index(inplace=True, drop=True)
    df_raw['email'] = df_raw['email'].astype(str).str.strip().str.lower()
    # Strip spaces from first and last names
    df_raw['first name'] = df_raw['first name'].astype(str).str.strip()
    df_raw['last name'] = df_raw['last name'].astype(str).str.strip()
    df_raw['full_name'] = df_raw['first name'] + ' ' + df_raw['last name']

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

    # %% Generate Study ID for the records
    # df_phi = df_raw[['record_id', 'full_name', 'dob', 'email']]
    df_raw['full_name'] = df_raw.full_name.str.lower()
    df_raw = sort_columns(df_raw)

    df_raw['study_key'] = df_raw['full_name'].astype(str) + '_' + \
                          df_raw['dob'].astype(str)
                # df_phi['email'].astype(str) + '_' + \
    df_raw['study_id'] = pd.factorize(df_raw['study_key'])[0] + 1  # Start IDs at 1
    df_raw = df_raw.drop(columns='study_key')
    repeated_names = df_raw[df_raw['full_name'].duplicated(keep=False)]
    # %% drop 10 conversion, here teh dataset is already at 30 minutes
    assert all(df_raw['sss10_15min'].isna()) == True
    df_raw.drop(columns=['sss10_15min'], inplace=True)
    df_raw.rename(columns={'sss10-30min': 'sss10'}, inplace=True)
    # %% Date
    df_date = pd.read_excel(config.get('data_raw_path').get('second_cross_sectional'),
                           sheet_name='split 1- NA removed')
    df_date = df_date.loc[df_date['Study ID'].isin(df_raw.record_id.unique()), ['Study ID', 'Observation #']]
    df_date.rename(columns={'Observation #': 'date'}, inplace=True)

    df_raw = pd.merge(left=df_raw,
                        right=df_date,
                        left_on='record_id',
                        right_on='Study ID',
                        how='left')
    df_raw.drop(columns='Study ID', inplace=True)
    # %% PHI data for the ASQ merging
    # df_phi = pd.read_excel(config.get('data_raw_path').get('second_cross_sectional'),
    #                        sheet_name='Split 1 Study ID code')
    # df_phi['email'] = df_phi['Best Email Address'].astype(str).str.strip().str.lower()
    # # Strip spaces from first and last names
    # df_phi['First Name'] = df_phi['First Name'].astype(str).str.strip()
    # df_phi['Last Name'] = df_phi['Last Name'].astype(str).str.strip()
    # df_phi = df_phi.loc[df_phi['First Name'] != 'nan', :]
    #
    # # Create full_name with a clean format
    # df_phi['full_name'] = df_phi['First Name'] + ' ' + df_phi['Last Name']
    # df_phi.columns = map(str.strip, df_phi.columns)
    # df_phi.rename(columns=mapper_columns, inplace=True)
    # # df_phi = df_phi[['study_id','record_id', 'full_name']]
    # df_phi = df_phi.loc[~df_phi['full_name'].isna(), :]
    #
    # # assert df_phi.full_name.nunique() == df_phi.shape[0]
    # # study id has not been assigned to the subjects in the dataset, we need to do it
    # # Combine the fields into a single string per row
    # df_phi['email'] = df_phi.email.str.lower()
    # df_phi['study_key'] = df_phi['full_name'].astype(str) + '_' + \
    #                       df_phi['email'].astype(str) + '_' + \
    #                       df_phi['dob'].astype(str)
    #
    # # Convert that combination to a unique number using factorize (efficient and consistent)
    # df_phi['study_id'] = pd.factorize(df_phi['study_key'])[0] + 1  # Start IDs at 1
    # df_phi = df_phi.drop(columns='study_key')
    #
    # # df_phi = sort_columns(df_phi)
    # df_phi = df_phi[['study_id', 'record_id', 'full_name' ]]
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
    col_interest = [*mapper_columns.values()] + col_race + col_sss + col_ess + col_diagnosis + ['full_name']
    col_interest = sorted(set(col_interest))
    df_raw = df_raw[col_interest]
    df_raw.replace('.',  np.nan, inplace=True)
    # df_raw.replace({np.nan: np.nan}, inplace=True)
    df_raw.dropna(how='all', inplace=True)
    df_raw.dropna(subset=col_ess, inplace=True)  # 1 row
    df_raw[col_ess] = df_raw[col_ess].astype(int)

    df_raw = df_raw.dropna(subset=col_sss, how='all')
    # %% SSS correction
    assert int(df_raw[col_sss].isna().sum().sum()) == 1 # one nan in column sss1 (not used for the score)

    assert df_raw[col_sss].max().max() == 6
    # 1. replace the shifte extra number with nans (6 -> nan)
    df_raw[col_sss] = df_raw[col_sss].replace({5: np.nan,
                                               6: np.nan})
    assert df_raw[col_sss].max().max() == 4
    # 2. substract and ignore the nan values
    df_raw[col_sss] = df_raw[col_sss].mask(df_raw[col_sss].notna(), df_raw[col_sss] - 1)

    assert df_raw[col_sss].min().min() == 0
    assert df_raw[col_sss].max().max() == 3

    # TODO: for the SSS we have 4 and 5, these are not applicable responses, in any of the questions
    # TODO: Look at the outlier for the correlation
    # TODO: BLAND ANDMANT PLOT
    # TODO: PC together between the both


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

    df_raw['gender'] = df_raw['gender'].map({1: 1, 2: 0})
    # %%
    # df_raw['study_id'] = df_raw['record_id']
    df_raw.sort_values(by='study_id', inplace=True)
    df_raw.replace('.', np.nan, inplace=True)
    # sort columns alphabetically
    df_raw = sort_columns(df=df_raw)
    # df_raw.drop(columns=['record_id', 'dob'],
    #             inplace=True)
    # %% compute SS10

    # # %% SSS replace nan with not applicable
    # col_sss = [col for col in df_raw.columns if 'sss' in col and ' ' not in col]
    # df_subset = df_raw[col_sss]
    # nan_per_column = df_subset.isna().sum()
    # subjects_with_nan = df_subset.isna().astype(int)
    # df_subset.loc[df_subset.isna().sum(1) > 0, :]
    # print(f'Total subjects with any nan {df_subset.loc[df_subset.isna().sum(1) > 0, :].shape[0]} of {df_raw.shape[0]}')
    # summary_df = pd.DataFrame({
    #     'Column': nan_per_column.index,
    #     'Number of NaNs': nan_per_column.values,
    #     'Subjects with NaNs': subjects_with_nan.astype(bool).sum(axis=0).values
    # }).set_index('Column')
    # print(f'Summary of NaNs per column\n{summary_df}')
    # %% re-compute the scores for the SSS
    sit_quest = SituationalSleepinessScale()
    df_raw['sss_score'] = sit_quest.compute_score(responses=df_raw[col_sss])
    df_raw['sss_score_div_num_quest'] = sit_quest.compute_score_normalized(responses=df_raw[col_sss])
    # %% ESS score
    col_ess = [col for col in df_raw.columns if 'ess' in col and ' ' not in col]
    ess_score = EpworthScale()
    df_raw['ess_score'] = ess_score.compute_score(responses=df_raw[col_ess])
    df_raw['ess_score_div_num_quest'] = ess_score.compute_score_normalized(responses=df_raw[col_ess])
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
    df_raw['batch'] = 'second'
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

















