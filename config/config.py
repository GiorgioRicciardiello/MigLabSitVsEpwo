"""
Configuration file of the project

"""
import pathlib
from typing import Union

import numpy as np
import pandas as pd

#%% define the directories
# Define root path
root_path = pathlib.Path(__file__).resolve().parents[1]
# Define raw data path
data_raw_path = root_path.joinpath('data', 'rawdata', 'SSS_data_download_18Apr2024.xlsx')
# Define pre-processed data path
data_pre_proc_path = root_path.joinpath('data', 'ppdata', 'pp_data.csv')
# Define results path
results_path = root_path.joinpath('results')

config = {
    'root_path': root_path,
    'data_raw_path': data_raw_path,
    'data_pp_path': data_pre_proc_path,
    'results_path': results_path,

}

mapper = {
    'gender':{'Female': 0,
              'Male': 1},
    'race': {'white': 0,
               'Asian': 1,
               'Black': 2,
               'Mixed': 3,
               'Not reported': 4,
               'other': 5,
               'Pacific Islander': 6},

    'ethnicity':{'Non-hispanic': 0,
                    'Hispanic':1,
                 },

    'standardization': {
        'Study ID': 'study_id',
        'Record ID': 'record_id',
        'Survey Timestamp': 'date',
        'Date of Birth': 'dob',
        'Sex': 'gender',
        'SSS10-30mina': 'sss10_30min',
        'SSS10-15mins': 'sss10_15min',
        'Score SSS': 'sss_score',
        'SSS divided by #questions answered': 'sss_scores_div_num_quest',
        'ESS total': 'ess_score',
        'ESS Divided': 'ess_score_div_num_quest',
        'BMI': 'bmi',
        'OSA 1-Mild, 2-Moderate, 3-Severe)': 'osa_level',
        'Narcolepsy (1- NT1, 2-NT2, 3-IH)': 'narc_level',
        'RLS': 'rls',
        'Insomnia': 'insomnia',
        'AHI 3%': 'ahi_3per',
        'AHI 4%': 'ahi_4per',
        'ODI 3%': 'odi_3per',
        'ODI 4%': 'odi_4per',
        '% in stage 1': 'per_stage_one',
        'Rem latency (min)': 'rem_lat_min',
        'Sleep latency (min)': 'sleep_lat_min',
        'Sleep Efficiency (%)': 'sleep_eff_perc',
        'PLMI': 'plmi',
        'rMEQ': 'rmeq'
    },
}


# mapping of the diagnosis
mapper_diagnosis = {
    'narc_level': {
        0:0,
        1:1,
        2:2,
        3:3,
        '1-3':4,
        '2-3':5,
    },
    'osa_levels': {0: 'Normal',
                   1: 'Mild',
                   2: 'Moderate',
                   3: 'Severe'},
    'insomnia': {
        1:'yes',
        0:'no'
    }
}




