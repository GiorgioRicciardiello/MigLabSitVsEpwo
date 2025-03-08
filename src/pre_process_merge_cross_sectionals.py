import pandas as pd
from config.config import config, mapper, mapper_diagnosis


if __name__ == '__main__':
    # %% rename columns
    df_first = pd.read_csv(config.get('data_pp_path').get('first_cross_sectional'))
    df_second = pd.read_csv(config.get('data_pp_path').get('second_cross_sectional'))

    # Find the common columns between the two dataframes
    common_columns = df_first.columns.intersection(df_second.columns)

    # Subset both dataframes to include only the common columns
    df_first_common = df_first[common_columns]
    df_second_common = df_second[common_columns]

    # Concatenate the dataframes vertically and reset the index
    df_combined = pd.concat([df_first_common, df_second_common], ignore_index=True)

    # remove the narcolepsy diagnosis that are unclear from chart (4 observations from the first cross sectional)
    df_combined = df_combined.loc[df_combined['narc_level'].isin({0,1,2,3}), :]

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

