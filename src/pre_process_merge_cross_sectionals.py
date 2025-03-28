import pandas as pd
from config.config import config, mapper, mapper_diagnosis
import numpy as np

if __name__ == '__main__':
    # %% rename columns
    df_first = pd.read_csv(config.get('data_pp_path').get('first_cross_sectional'))
    df_second = pd.read_csv(config.get('data_pp_path').get('second_cross_sectional'))

    # shift the study_id of the second follow up so it does not match those form the first
    # the study id of the second one were assigned based on unique names and dob from the second follow up, they are
    # independent from the first
    df_second['study_id'] = df_second['study_id'].apply(lambda x: x + df_first.study_id.max())


    # Find the common columns between the two dataframes
    common_columns = df_first.columns.intersection(df_second.columns)

    # Subset both dataframes to include only the common columns
    df_first_common = df_first[common_columns]
    df_second_common = df_second[common_columns]


    # check if there are full names in common
    common_names = list(set(df_first_common['full_name']) & set(df_second_common['full_name']))  # 3 common names only
    # assing the study_id that is unique for each subject for those subject that did the quetionnaire at the two follow ups
    for common_name in common_names:
        previous_id = df_first_common.loc[df_first_common['full_name'] == common_name, 'study_id'].unique()[0]
        print(f'{common_name}: {previous_id}')
        df_second_common.loc[df_second_common['full_name'] == common_name, 'study_id'] = previous_id

    common_ids = set(df_first_common['study_id']) & set(df_second_common['study_id'])
    assert len(common_ids) == len(common_names)

    # Concatenate the dataframes vertically and reset the index
    df_combined = pd.concat([df_first_common, df_second_common], axis=0)
    df_combined.reset_index(drop=True, inplace=True)
    # remove the narcolepsy diagnosis that are unclear from chart (4 observations from the first cross sectional)
    df_combined = df_combined.loc[df_combined['narc_level'].isin({0,1,2,3, np.nan}), :]
    df_combined['study_id'] = df_combined['study_id'].astype(int)

    # Replace NaNs in race with the most frequent category (mode)
    most_common_race = df_combined['race'].mode()[0]
    df_combined['race'] = df_combined['race'].fillna(most_common_race)

    most_common_gender = df_combined['gender'].mode()[0]
    df_combined['gender'] = df_combined['gender'].fillna(most_common_gender)

    # Make sure race and gender are treated as categorical with 0 as the reference
    df_combined['race'] = pd.Categorical(df_combined['race'], categories=sorted(df_combined['race'].unique()))
    df_combined['gender'] = pd.Categorical(df_combined['gender'], categories=sorted(df_combined['gender'].unique()))
    df_combined.to_csv(config.get('data_pp_path').get('pp_data'), index=False)

    counts_dict = {
        'osa_level': df_combined['osa_level'].value_counts(),
        'insomnia': df_combined['insomnia'].value_counts(),
        'narc_level': df_combined['narc_level'].value_counts(),
        'rls': df_combined['rls'].value_counts()
    }

    # Combine into a DataFrame, filling missing values with 0 and converting to integers
    df_counts = pd.DataFrame(counts_dict).fillna(0).astype(int)
    print(df_counts)


    # TODO:
    # 1. csv with study id and demo to match with the ASQ COMPLETED
    # 2. ASQ pull
    # 3. ASQ matches by name and dob, hope to find all
    # 4. Remove the duplicates COMPLETED
    # 4.1. second follow use email, first_name, last_name because same people might be in both, assing the study id to the second one
    # 4.2. create the study_id for the second follow up using the unique email first name and last name
    # 5. From the ASQ take all the records with the SSS asnwered, remove duplicates iwth the exissting and add more subejcts that only did the ASQ




