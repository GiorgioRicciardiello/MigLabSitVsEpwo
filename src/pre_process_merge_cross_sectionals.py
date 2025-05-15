import pandas as pd
from config.config import config, mapper, mapper_diagnosis
import numpy as np
from tabulate import tabulate
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

    #%% Clean the SSS by removing subjects who responses >4 in any of the SSS questions
    # Identify columns
    col_sss_no_score = [col for col in df_combined.columns if 'sss' in col and 'score' not in col]
    col_ess = [col for col in df_combined.columns if 'ess' in col and len(col) < 10]

    # Subset of SSS-related columns (excluding 'score')
    df_subset = df_combined[col_sss_no_score]

    # NaN statistics
    nan_per_column = df_subset.isna().sum()
    n_total_subjects = df_combined.shape[0]
    n_subjects_with_any_nan = (df_subset.isna().sum(axis=1) > 0).sum()

    print(f"\n📊 Total subjects: {n_total_subjects}")
    print(
        f"❗ Subjects with any NaN in SSS (non-score) columns: {n_subjects_with_any_nan} ({n_subjects_with_any_nan / n_total_subjects:.2%})")

    # Summary of NaNs per column
    summary_df = pd.DataFrame({
        'Number of NaNs': nan_per_column,
        'Percentage (%)': (nan_per_column / n_total_subjects * 100).round(2)
    }).sort_values(by='Number of NaNs', ascending=False)

    print("\n🔍 NaN Summary per SSS Column:")
    print(tabulate(summary_df))

    # Save full NaN summary
    df_nana_counts = df_combined.isna().sum().reset_index()
    df_nana_counts.columns = ['Column', 'Number of NaNs']
    df_nana_counts['Percentage of NaNs'] = (df_nana_counts['Number of NaNs'] / n_total_subjects * 100).round(2)
    df_nana_counts.to_csv(config.get('results_path').joinpath('nan_counted_merged_before_dropping_on_sss.csv'),
                          index=False)

    # Count how many entries in SSS columns are "not applicable" (>= 4)
    greater_than_4_counts = (df_combined[col_sss_no_score] >= 4).sum()
    print("\n🛑 Count of 'not applicable' responses (value >= 4) per SSS column:")
    print(greater_than_4_counts)

    # Removing the following cell because we want to preserve the nan values
    # Drop rows with any 'not applicable' response or NaN in SSS columns
    # n_before_cut = df_combined.shape[0]
    # df_combined = df_combined[~(df_combined[col_sss_no_score] >= 4).any(axis=1)]
    # df_combined = df_combined[~df_combined[col_sss_no_score].isna().any(axis=1)]
    # n_after_cut = df_combined.shape[0]
    #
    # print(f"\n✅ Rows remaining after cleaning: {n_after_cut} (removed {n_before_cut - n_after_cut})")

    # 📊 Total subjects: 527
    # ❗ Subjects with any NaN in SSS (non-score) columns: 87 (16.51%)
    # 🔍 NaN Summary per SSS Column:
    #        Number of NaNs  Percentage (%)
    # sss8               61           11.57
    # sss6               17            3.23
    # sss7               14            2.66
    # sss5               12            2.28
    # sss10              11            2.09
    # sss1                7            1.33
    # sss9                7            1.33
    # sss4                4            0.76
    # sss11               4            0.76
    # sss3                2            0.38
    # sss2                1            0.19
    # 🛑 Count of 'not applicable' responses (value >= 4) per SSS column:
    # sss1     31
    # sss11    47
    # sss2     22
    # sss3      3
    # sss4     13
    # sss5     37
    # sss6      9
    # sss7     17
    # sss8     34
    # sss9     91
    # sss10    63
    # dtype: int64
    # ✅ Rows remaining after cleaning: 334 (removed 193)

    # %% save the dataset
    df_combined.to_csv(config.get('data_pp_path').get('pp_data'), index=False)
    print(f"Data saved in {config.get('data_pp_path').get('pp_data')}\n\tDim: {df_combined.shape}")
    counts_dict = {
        'osa_level': df_combined['osa_level'].value_counts(),
        'insomnia': df_combined['insomnia'].value_counts(),
        'narc_level': df_combined['narc_level'].value_counts(),
        'rls': df_combined['rls'].value_counts()
    }

    # Combine into a DataFrame, filling missing values with 0 and converting to integers
    df_counts = pd.DataFrame(counts_dict).fillna(0).astype(int)
    print(df_counts)

    df_combined.groupby(by='batch').count()
    # total samples : 334
    #      osa_level  insomnia  narc_level  rls
    # 0.0         61       176         278  256
    # 1.0         95       116           3   30
    # 2.0         72         0           3    0
    # 3.0         62         0           5    0


    # TODO:
    # 1. csv with study id and demo to match with the ASQ COMPLETED
    # 2. ASQ pull
    # 3. ASQ matches by name and dob, hope to find all
    # 4. Remove the duplicates COMPLETED
    # 4.1. second follow use email, first_name, last_name because same people might be in both, assing the study id to the second one
    # 4.2. create the study_id for the second follow up using the unique email first name and last name
    # 5. From the ASQ take all the records with the SSS asnwered, remove duplicates iwth the exissting and add more subejcts that only did the ASQ




