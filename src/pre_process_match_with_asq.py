import pathlib
import pandas as pd
import numpy as np
from config.config import config, mapper, mapper_diagnosis
import matplotlib.pyplot as plt
from library.fuzzy_search import FuzzySearch, NameDateProcessor
import logging

def fuzzy_search_wrapper(df_students: pd.DataFrame,
                         df_academic_calendar: pd.DataFrame,
                         df_asq: pd.DataFrame,
                         scorer: str) -> pd.DataFrame:
    """
    Fuzzy search wrapper tailor for matching the ASQs with the students. For reducing the possible of matches within
    the fuzzy filter threshold we use the academic calendar to limit the ASQ for the period of the course the students
    were enrolled. This also optimizes the search by reducing the search of possible matches at each iteration.
    :param df_students:
    :param df_academic_calendar:
    :param df_asq:
    :param scorer:
    :return:
        dataframe with the results of the matches and the score
    """
    df_matches = pd.DataFrame()
    for idx, calendar in df_academic_calendar.iterrows():
        # calendar = df_academic_calendar.loc[4, :]
        year = int(calendar['Academic Year'].split('-')[0])
        df_enrolled = df_students.loc[(df_students['year'] == year) & (df_students['quarter'] == calendar.Term), :]
        if df_enrolled.shape[0] == 0:
            logging.info(f'No students in the {calendar.Term} - {year} quarter term')
            continue
        # we could use the start_time or completed or created_at
        df_asq_quarter = df_asq.loc[(df_asq.created_at >= calendar['Start Date']) &
                                    (df_asq.created_at <= calendar['End Date']), :]

        logging.info(
            f'Searching for {df_enrolled.shape[0]} students in {df_asq_quarter.shape[0]} ASQs for '
            f'the {calendar.Term} - {year} quarter term')

        fuzzy_search = FuzzySearch(asq_df=df_asq_quarter,
                                   subjects_df=df_enrolled,
                                   scorer=scorer)

        matches = fuzzy_search.search_by_name_matches(fuzzy_filter=60)
        matches['quarter'] = f'{calendar.Term}-{year}'
        df_matches = pd.concat([df_matches, matches])
    df_matches.rename(columns={'subject_idx': 'idx_student_db',
                               'subject_id': 'idx_asq_db'
                               }, inplace=True)
    df_matches.sort_values(by='score', inplace=True, ascending=False)
    return df_matches


if __name__ == '__main__':
    # %% Read the data
    df_combined = pd.read_csv(config.get('data_pp_path').get('pp_data'))
    df_asq = pd.read_csv(config.get('data_pp_path').get('asq_data'))

    # %% Structure the ASQ for the matches
    df_asq = df_asq[~df_asq['name'].isna()]
    df_asq['name'] = df_asq['name'].astype(str)
    df_asq = df_asq[df_asq['name'] != 'O-l']
    df_asq['completed'] = pd.to_datetime(df_asq.completed)
    df_asq['created_at'] = pd.to_datetime(df_asq['created_at'], format='mixed', utc=True)
    # we only need ASQ records from 2023 to 2025
    df_asq = df_asq.loc[~df_asq['start_time'].isna(), :]
    df_asq['start_time_year'] = df_asq['start_time'].apply(lambda x: int(x.split('-')[0]))
    df_asq = df_asq.loc[df_asq['start_time_year'] >= 2023, :]
    print(f'ASQ records: {df_asq.shape}')
    # %%
    df_combined.rename(columns={'full_name': 'name'}, inplace=True)
    scorer = 'R'
    fuzzy_search = FuzzySearch(asq_df=df_asq,
                               subjects_df=df_combined.copy(),
                               scorer=scorer)

    matches = fuzzy_search.search_by_name_dob_matches(fuzzy_filter=80)
