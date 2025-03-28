"""
Library used for the processing and matching of the two datasets.
The dataset are matched by full name and date of birth
"""
import pathlib
from pathlib import Path
import re
import pandas as pd
from typing import Optional, Tuple, Union, Any
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm import tqdm
from pandas import Series, DataFrame

def process_name(name:pd.Series):
    """
    Function to process 'name' column
    Remove spaces and split into first and last names
    :param name:
    :return:
    """
    # Remove spaces and split into first and last names
    if isinstance(name, str):
        # Define a regular expression pattern to match special characters
        # pattern = r'[!@#$%^&*()_+={}\[\]:;"\'<>,.?/\\|\xa0]'
        # Define a regular expression pattern to match non-alphabetic characters
        # pattern = r'[!@#$%^&*()_+={}\[\]:;"\'<>,.?/\\|\xa0℅]'
        # pattern = r'[^a-zA-Z ]'
        pattern = r'[^a-zA-Z -]'  # keep the - in the names
        # Use the re.sub() function to remove non-alphabetic characters
        name = re.sub(pattern, "", name)
        name = name.lstrip()
        name = name.strip()
        name = name.replace("  ", " ")
        name = name.lower()
        # Split the name into parts and capitalize the first letter of each part
        name_parts = name.split()
        name_parts = [part.capitalize() for part in name_parts]

        # Join the parts to form the modified name
        modified_name = " ".join(name_parts)

        return modified_name
    else:
        return name


class NameDateProcessor:
    def __init__(self):
        """
        The class has two methods:
        1. Encode the first middle and last name into single column and encode
        2. change the format of date columns

        Example usage:
        name_processor = NameProcessor()
        processed_dataframe = name_processor.encode_names(your_dataframe)
        """
        self.col_name_standard_name = "name"

    def _process_name(self, name: pd.Series) -> Union[str, Series]:
        """
        Method to process 'name' column
        Remove spaces and split into first and last names
        :param name:
        :return:
        """
        # Remove spaces and split into first and last names
        if isinstance(name, str):
            # Define a regular expression pattern to match non-alphabetic characters
            pattern = r'[^a-zA-Z ]'
            # Use the re.sub() function to remove non-alphabetic characters
            name = re.sub(pattern, "", name)
            name = name.lstrip()
            name = name.strip()
            name = name.replace("  ", " ")
            name = name.lower()
            # Split the name into parts and capitalize the first letter of each part
            name_parts = name.split()
            name_parts = [part.capitalize() for part in name_parts]

            # Join the parts to form the modified name
            modified_name = " ".join(name_parts)

            return modified_name
        else:
            return name

    def encode_names(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Create the name column or rename it to the standard convention.
        Create the column name with the combination of name_first, name_middle, and name_last. Middle name is optional
        :param frame:
        :return:
        """
        if all(column in frame.columns for column in ['name_first', 'name_middle', 'name_last']):
            frame[['name_first', 'name_middle', 'name_last']] = frame[
                ['name_first', 'name_middle', 'name_last']].fillna('')

            # Concatenate columns efficiently and join with space
            frame["name"] = (frame["name_first"] + " " + frame["name_middle"] + " " + frame["name_last"]).apply(
                lambda name: " ".join(name.split())).apply(self._process_name)

            frame.drop(columns=['name_first', 'name_middle', 'name_last'], inplace=True)

            return frame

        elif all(column in frame.columns for column in ['name_first', 'name_last']):
            frame[['name_first', 'name_last']] = frame[
                ['name_first', 'name_last']].fillna('')

            # Concatenate columns efficiently and join with space
            frame["name"] = (frame["name_first"] + " " + frame["name_last"]).apply(
                lambda name: " ".join(name.split())).apply(self._process_name)

            frame.drop(columns=['name_first', 'name_last'], inplace=True)

            return frame

    def encode_date_columns(self, frame: pd.DataFrame, dob_column:str) -> pd.DataFrame:
        """
        Encode the date of birth column, it accounts for the missing values (keeps them) and set a string or datetime
        format to the date of birth. Then rename to the standard name for dob
        :param frame:
        :return:
        """
        format = 'time'
        result = frame[dob_column].dropna().copy()
        result = pd.to_datetime(result,
                                errors='coerce').dt.strftime('%Y-%m-%d')
        if format == 'string':
            frame[dob_column] = result.combine_first(frame[dob_column])
        else:
            frame[dob_column] = result.combine_first(pd.to_datetime(frame[dob_column],
                                                                    errors='coerce'))
        return frame

class FuzzySearch:
    def __init__(self,
                 asq_df: Union[pd.DataFrame, pathlib.Path],
                 subjects_df: Union[pd.DataFrame, pathlib.Path],
                 scorer:Optional[str]='TSeR'
                 ):
        """
        Search for matched between the subjects_df and the asq_df
        :param asq_df:
        :param subjects_df:
        """
        self.asq_df = self._read_csv_or_dataframe(asq_df)
        self.subjects_df = self._read_csv_or_dataframe(subjects_df)
        self.dob_variations = ['dob', 'date of birth', 'date_of_birth', 'date-of-birth']
        self.col_name_standard_name = 'name'
        self.col_dob_standard_name = 'date_of_birth'
        self.scorer_dict = {'R': fuzz.ratio,
                       'PR': fuzz.partial_ratio,
                       'TSeR': fuzz.token_set_ratio,
                       'TSoR': fuzz.token_sort_ratio,
                       'PTSeR': fuzz.partial_token_set_ratio,
                       'PTSoR': fuzz.partial_token_sort_ratio,
                       'WR': fuzz.WRatio,
                       'QR': fuzz.QRatio,
                       'UWR': fuzz.UWRatio,
                       'UQR': fuzz.UQRatio}
        if scorer not in self.scorer_dict.keys():
            raise ValueError(f'Scorer must be one of {self.scorer_dict.keys()}')
        self.scorer = self.scorer_dict.get(scorer)


    @staticmethod
    def _read_csv_or_dataframe(data: Union[pd.DataFrame, pathlib.Path]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, (str, Path)):
            if isinstance(data, Path):
                if data.suffix == '.csv':
                    return pd.read_csv(data)
                else:
                    return pd.read_excel(data)
            else:
                return pd.read_csv(data)
        else:
            raise ValueError("Input must be a DataFrame, a path to a CSV file, or a pathlib.Path.")

    @staticmethod
    def _get_column_intersection(df_one: pd.DataFrame,
                                 df_two: pd.DataFrame) -> list:
        """
        return the intersection of two dataframes columns"
        :param df_one:
        :param df_two:
        :return:
        """
        return list(set(df_one.columns) & set(df_two.columns))

    @staticmethod
    def _check_columns_exist(dataframe: pd.DataFrame,
                             columns_to_check: Union[list, str]) -> bool:
        """Check if the columns we will work with are in the input"""
        if isinstance(columns_to_check, str):
            columns_to_check = [columns_to_check]
        missing_columns = [col for col in columns_to_check if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the dataframe: {missing_columns}")
        else:
            print("All required columns are present in the dataframe.")
            return True

    def assign_mrn_by_name_dob_matches(self, fuzzy_filter: int = 92) -> Union[pd.DataFrame, bool]:
        """
        Insert patient MRN in the ASQ from the subject dataframe
        :param fuzzy_filter:
        :return:
            pd.Dataframe with similar names
        """
        columns = [self.col_name_standard_name, self.col_dob_standard_name]
        # asq = self.asq_df.copy()
        # subjects_df = self.subjects_df.copy()
        self._pre_process_search_by_name_dob_matches(frames=[self.asq_df, self.subjects_df])

        if not self._check_columns_exist(dataframe=self.asq_df, columns_to_check=columns):
            print(F"ASQ Dataframe must have the columns {columns}")
            return False
        columns.append('mrn')
        if not self._check_columns_exist(dataframe=self.subjects_df, columns_to_check=columns):
            print(F"Subject Dataframe must have the columns {columns}")
            return False

        # implement the fuzzy search
        # asq = self.asq_df.copy()
        # subjects_df = self.subjects_df.copy()
        similar_names = []
        # fuzzy_result_df = pd.DataFrame(columns=['asq_name', 'subject_name', 'score', 'asq_dob',
        #                                         'subject_mrn', 'asq_mrn'],
        #                                # index=range(0, self.asq_df.shape[0])
        #                                )
        for idx_, subject in tqdm(self.subjects_df.iterrows(),
                                  total=len(self.subjects_df),
                                  desc="Matching Subjects Name & Dob with ASQ Records"):
            # subject is the patient we are searching in the asq database
            # subject = self.subjects_df.loc[0, :]
            # filter by  date of birth
            asq_dob_matches = self.asq_df[
                self.asq_df[self.col_dob_standard_name] == subject[self.col_dob_standard_name]]
            # in the asq rows with same dob, do a fuzzy search for the name match
            matches = process.extract(subject[self.col_name_standard_name],
                                      asq_dob_matches[self.col_name_standard_name],
                                      scorer=fuzz.token_set_ratio,
                                      limit=asq_dob_matches.shape[0])
            similar_names.extend([{'asq_name': fuzzy_[0],
                                   'subject_name': subject[self.col_name_standard_name],
                                   'score': fuzzy_[1],
                                   'asq_dob': self.asq_df.loc[fuzzy_[2], self.col_dob_standard_name],
                                   'subject_dob': subject[self.col_dob_standard_name],
                                   'subject_mrn': subject['mrn'],
                                   'asq_mrn': self.asq_df.loc[fuzzy_[2], 'mrn'],
                                   'subject_idx': idx_,
                                   'asq_survey_id': self.asq_df.loc[fuzzy_[2], 'survey_id'],
                                   # 'asq_epic_id': self.asq_df.loc[fuzzy_[2], 'epic_id'],
                                   # 'asq_completed': self.asq_df.loc[fuzzy_[2], 'completed'],
                                   } for fuzzy_ in matches])
        # Create a DataFrame with similar names and similarity scores
        similar_names_df = pd.DataFrame(similar_names)
        # filter by those higher than 92
        similar_names_df = similar_names_df[similar_names_df['score'] >= fuzzy_filter]
        # fuzzy_result_df = fuzzy_result_df.append(self._fuzzy_rule(fuzzy_matches=similar_names_df),
        #                                          ignore_index=True)
        return similar_names_df

    @staticmethod
    def _fuzzy_rule(fuzzy_matches: pd.DataFrame) -> pd.DataFrame:
        """
        We should return a dataframe with only one row, the best match
        :param fuzzy_matches:
        :return:
        """
        if fuzzy_matches.asq_name.unique().shape[0] == 1:
            # we have the same name match in all we return
            fuzzy_matches = fuzzy_matches.sort_values(by='score',
                                                      ascending=False)
            return fuzzy_matches.loc[0, :]
        elif fuzzy_matches.asq_name.unique().shape[0] > 1:
            # different names in the fuzzy match, we select the one with highet score
            top_score = fuzzy_matches[fuzzy_matches['score'] == fuzzy_matches['score'].max()]
            return top_score

    def _pre_process_search_by_name_dob_matches(self,
                                                frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Implement a pre-processing operation to the name and date of birth columns
        :param frames:
        :return: list[pd.DataFrame] list of pre-processed dataframes
        """
        for frame in frames:
            # frame = frames[1]
            if any('unnamed:' in column.lower() for column in frame.columns):
                unnamed_drop = [column for column in frame.columns if 'unnamed:' in column.lower()]
                frame.drop(columns=unnamed_drop, inplace=True)
            self._encode_names(frame=frame)
            self._encode_date_of_birth(frame=frame)
        return frames

    def _pre_process_search_by_name_matches(self,
                                                frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Implement a pre-processing operation to the name and date of birth columns
        :param frames:
        :return: list[pd.DataFrame] list of pre-processed dataframes
        """
        for frame in frames:
            # frame = frames[1]
            if any('unnamed:' in column.lower() for column in frame.columns):
                unnamed_drop = [column for column in frame.columns if 'unnamed:' in column.lower()]
                frame.drop(columns=unnamed_drop, inplace=True)
            self._encode_names(frame=frame)
        return frames

    def _encode_names(self, frame: pd.DataFrame):
        """
        Create the name column or rename it to the standard convention.
        Creste the column name with the combination of name_fisrt, name_middle and name_last. Middle name is optional
        :param frame:
        :return:
        """
        # if any(column.casefold() == 'name' for column in frame.columns):
        #     name_drop = [column for column in frame.columns if column.casefold() == 'name'][0]
        #     frame.rename(columns={name_drop: self.col_name_standard_name}, inplace=True)

        if all(column in frame.columns for column in ['name_first', 'name_middle', 'name_last']):
            frame[['name_first', 'name_middle', 'name_last']] = frame[
                ['name_first', 'name_middle', 'name_last']].fillna('')

            frame["name"] = (frame["name_first"] + " " + frame["name_middle"] + " " + frame["name_last"]).apply(
                process_name)
            frame.drop(columns=['name_first', 'name_middle', 'name_last'], inplace=True)
            frame["name"] = frame["name"].apply(process_name)

        elif all(column in frame.columns for column in ['name_first', 'name_last']):
            frame[['name_first', 'name_last']] = frame[
                ['name_first', 'name_last']].fillna('')
            frame["name"] = (frame["name_first"] + " " + frame["name_last"]).apply(
                process_name)
            frame.drop(columns=['name_first', 'name_last'], inplace=True)
            frame["name"] = frame["name"].apply(process_name)

        else:
            name_drop = frame.filter(like='name', axis=1).columns
            name_drop = [non_name for non_name in name_drop if non_name != 'name']
            # frame.rename(columns={name_drop[0]: self.col_name_standard_name}, inplace=True)
            frame.drop(columns=name_drop,
                       inplace=True)
    def _encode_date_of_birth(self,
                              frame: pd.DataFrame):
        """
        Encode the date of birth column, it accounts for the missing values (keeps them) and set a string or datetime
        format to the date of birth. Then rename to the standard name for dob
        :param frame:
        :return:
        """
        format = 'time'
        dob_column: str = self._keep_first_occurrence(
            frame=frame,
            column_aliases=self.dob_variations)

        result = frame[dob_column].dropna().copy()
        result = pd.to_datetime(result,
                                errors='coerce').dt.strftime('%Y-%m-%d')
        if format == 'string':
            frame[dob_column] = result.combine_first(frame[dob_column])
        else:
            frame[dob_column] = result.combine_first(pd.to_datetime(frame[dob_column],
                                                                    errors='coerce'))
        frame.rename(columns={dob_column: self.col_dob_standard_name},
                     inplace=True)

    @staticmethod
    def _keep_first_occurrence(frame: pd.DataFrame, column_aliases: list) -> str:
        """
        Sme column with different aliases e.g., dob and date of birth in the same dataframe
        :param frame: datrafem to search the alises columns
        :param column_aliases: list of aliases possible for the same column
        :return:
        """
        #  TODO: test this  name_drop = frame.filter(like='name', axis=1).columns for the dob matches
        column_aliases = set(column_aliases)  # Convert to set to ensure unique variations
        columns_to_keep = []
        for alias in column_aliases:
            for column in frame.columns:
                if alias.casefold() in column.casefold():
                    columns_to_keep.append(column)
                    # break  # Break out of the inner loop after finding the first occurrence
        # keep the first one
        column = columns_to_keep[0]
        # remove all others if present
        if len(columns_to_keep) > 1:
            frame.drop(columns=columns_to_keep[1::],
                       inplace=True)
        return column


    def search_by_name_dob_matches(self, method: str = 'fuzzy', fuzzy_filter:Optional[int]=95) -> pd.DataFrame:
        """
        Wrapper function to search subjects in the main asq by name and dob
        :param method:
        :return:
        """
        self._pre_process_search_by_name_dob_matches(frames=[self.asq_df, self.subjects_df])
        if method == 'exact':
            return self._exact_search_name_dob()
        elif method == 'fuzzy':
            return self._fuzzy_search_name_dob(fuzzy_filter=fuzzy_filter)


    def search_by_name_matches(self,
                               fuzzy_filter:Optional[int]=95) -> pd.DataFrame:
        """Use only the name foe the fuzzy search"""
        self._pre_process_search_by_name_matches(frames=[self.asq_df, self.subjects_df])
        return self._fuzzy_search_name(fuzzy_filter=fuzzy_filter)


    def _exact_search_name_dob(self) -> pd.DataFrame:
        """
        Search for the patient using the date of birth and then the exact name match
        :return:
        """
        result_exact_search = []
        for idx_, subject in tqdm(self.subjects_df.iterrows(),
                                  total=len(self.subjects_df),
                                  desc="Matching Subjects Name & Dob with ASQ Records"):
            # subject is the patient we are searching in the asq database
            # subject = self.subjects_df.loc[0, :]
            # filter by  date of birth
            asq_dob_matches = self.asq_df[
                self.asq_df[self.col_dob_standard_name] == subject[self.col_dob_standard_name]]
            asq_dob_names_matches = asq_dob_matches[asq_dob_matches[self.col_name_standard_name] ==
                                                    subject[self.col_name_standard_name]]
            result_exact_search.append(asq_dob_names_matches.to_dict(orient='records'))

        # Flatten the list of dictionaries and create the DataFrame directly
        return pd.DataFrame([item for sublist in result_exact_search for item in sublist])

    def _fuzzy_search_name_dob(self, fuzzy_filter:int):
        """
        Search for the patient using the date of birth and a fuzzy search for the name match
        :param fuzzy_filter: int, filter to implement in the fuzzy score
        :return:
        """
        similar_dob_name = []
        for idx_, subject in tqdm(self.subjects_df.iterrows(),
                                  total=len(self.subjects_df),
                                  desc="Matching Subjects Name & Dob with ASQ Records"):
            # subject is the patient we are searching in the asq database
            # subject = self.subjects_df.loc[0, :]
            # filter by  date of birth
            asq_dob_matches = self.asq_df[
                self.asq_df[self.col_dob_standard_name] == subject[self.col_dob_standard_name]]
            # in the asq rows with same dob, do a fuzzy search for the name match
            matches = process.extract(subject[self.col_name_standard_name],
                                      asq_dob_matches[self.col_name_standard_name],
                                      scorer=fuzz.token_set_ratio,
                                      limit=asq_dob_matches.shape[0])
            similar_dob_name.extend([{'asq_name': fuzzy_[0],
                                   'subject_name': subject[self.col_name_standard_name],
                                   'score': fuzzy_[1],
                                   'asq_dob': self.asq_df.loc[fuzzy_[2], self.col_dob_standard_name],
                                   'subject_dob': subject[self.col_dob_standard_name],
                                   'subject_idx': idx_,
                                   'asq_survey_id': self.asq_df.loc[fuzzy_[2], 'survey_id'],
                                   # 'asq_epic_id': self.asq_df.loc[fuzzy_[2], 'epic_id'],
                                   # 'asq_completed': self.asq_df.loc[fuzzy_[2], 'completed'],
                                   } for fuzzy_ in matches])
        # Create a DataFrame with similar names and similarity scores
        similar_names_df = pd.DataFrame(similar_dob_name)
        # filter by those higher than 92
        similar_names_df = similar_names_df[similar_names_df['score'] >= fuzzy_filter]
        # fuzzy_result_df = fuzzy_result_df.append(self._fuzzy_rule(fuzzy_matches=similar_names_df),
        #                                          ignore_index=True)
        return similar_names_df

    def _fuzzy_search_name(self, fuzzy_filter:int):
        """
        Search for the patient using fuzzy search on the name
        https://medium.com/analytics-vidhya/matching-messy-pandas-columns-with-fuzzywuzzy-4adda6c7994f
        :param fuzzy_filter: int, filter to implement in the fuzzy score
        :return:
        """
        similar_dob_name = []
        for idx_, subject in tqdm(self.subjects_df.iterrows(),
                                  total=len(self.subjects_df),
                                  desc="Matching Subjects Name & Dob with ASQ Records"):
            # subject is the patient we are searching in the asq database
            # subject = self.subjects_df.loc[0, :]
            # filter by  date of birth
            # asq_dob_matches = self.asq_df[
            #     self.asq_df[self.col_dob_standard_name] == subject[self.col_dob_standard_name]]
            # in the asq rows with same dob, do a fuzzy search for the name match

            matches = process.extract(subject[self.col_name_standard_name],
                                      self.asq_df[self.col_name_standard_name],
                                      scorer=self.scorer,
                                      limit=self.subjects_df.shape[0])
            similar_dob_name.extend([{'asq_name': fuzzy_[0],
                                   'subject_name': subject[self.col_name_standard_name],
                                   'score': fuzzy_[1],
                                   'asq_dob': self.asq_df.at[fuzzy_[2], self.col_dob_standard_name],
                                   'asq_created_at': self.asq_df.at[fuzzy_[2], 'created_at'],
                                      'asq_start_time': self.asq_df.at[fuzzy_[2], 'start_time'],
                                      'asq_completed': self.asq_df.at[fuzzy_[2], 'completed'],
                                   'subject_idx': idx_,
                                    'subject_id': subject.ID,
                                   'asq_survey_id': self.asq_df.at[fuzzy_[2], 'survey_id'],
                                   # 'asq_epic_id': self.asq_df.loc[fuzzy_[2], 'epic_id'],
                                   # 'asq_completed': self.asq_df.loc[fuzzy_[2], 'completed'],
                                   } for fuzzy_ in matches])
        # Create a DataFrame with similar names and similarity scores
        similar_names_df = pd.DataFrame(similar_dob_name)
        # filter by those higher than 92
        similar_names_df = similar_names_df[similar_names_df['score'] >= fuzzy_filter]
        # fuzzy_result_df = fuzzy_result_df.append(self._fuzzy_rule(fuzzy_matches=similar_names_df),
        #                                          ignore_index=True)
        return similar_names_df

    def get_scorers(self)->dict:
        return self.scorer_dict