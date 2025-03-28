"""
Definition of the sleep scale questionnaire to be used in the study
"""
import pandas as pd
from typing import Dict

class SituationalSleepinessScale:
    """
    Cut off in the score at 10
    SSS < 10 -> Not sleepiness problem
    """
    def __init__(self,):
        self.levels = {'No chance': 0, 
                       'Slight chance': 1, 
                       'Moderate chance': 2, 
                       'High chance':3, 
                       'Not applicable':4,
                       'Prefer not to answer':5}
        self.questionnaire = {
            # TODO: try to figure a way to put the SS1 RENAME AS SSG because it is a general question so the numbering
            #  should go until 10
            'sss1': {
                'label': 'Over the past two weeks, how likely is it that you would unintentionally fall asleep or doze off?',
                'levels': [*self.levels.keys()]},
            'sss2': {
                'label': 'Sitting at a desk/table working on a computer or tablet',
                'levels': [*self.levels.keys()]},
            'sss3': {
                'label': 'Talking to someone on the phone',
                'levels': [*self.levels.keys()]},
            'sss4': {
                'label': 'In a meeting with several people',
                'levels': [*self.levels.keys()]},
            'sss5': {
                'label': 'Listening to someone talking in a class, lecture or at church',
                'levels': [*self.levels.keys()]},
            'sss6': {
                'label': 'Playing cards or a board game with others',
                'levels': [*self.levels.keys()]},
            'sss7': {
                'label': 'Driving a car',
                'levels': [*self.levels.keys()]},
            'sss8': {
                'label': 'Playing a videogame',
                'levels': [*self.levels.keys()]},
            'sss9': {
                'label': 'Lying down to rest',
                'levels': [*self.levels.keys()]},
            'sss10': {
                'label': 'Traveling as a passenger in a bus, train or car for more than 30 minutes',
                'levels': [*self.levels.keys()]},
            'sss11': {
                'label': 'Watching a film at home or the cinema',
                'levels': [*self.levels.keys()]},
            'sss_score': {
                'label': 'SSS Score ',
                'levels': [1,2,3,4,5,6,7,8,9,10,11 ]}
        }
        self.latent_factor = {
            'sss1': 'general',  # general likelihood of dozing off
            'sss2': 'active',  # working on a computer – requires attention
            'sss3': 'active',  # talking on the phone – interaction involved
            'sss4': 'passive',  # in a meeting – likely sitting and listening
            'sss5': 'passive',  # lecture/church – mostly listening
            'sss6': 'active',  # playing a game with others – interactive
            'sss7': 'active',  # driving – highly active and alert
            'sss8': 'active',  # videogames – engaging and interactive
            'sss9': 'passive',  # lying down – minimal engagement
            'sss10': 'passive',  # traveling as passenger – sedentary and passive
            'sss11': 'passive',  # watching a film – relaxed, seated
            'sss_score': 'computed'  # summary score, not a state
        }

    def compute_score(self,
                      responses: pd.DataFrame) -> pd.Series:
        """Compute the  SSS Score. We do not want to use ss1 for the score"""
        question = [quest for quest in list(self.questionnaire.keys()) if quest not in ['sss_score', 'sss1']]
        df_responses = responses[question].copy()
        # not applicable and prefer not to answer cannot be considered for the score computation
        df_responses.replace(to_replace=4, value=0, inplace=True)
        return df_responses.sum(1)

    def apply_cut_off(self, col_score:pd.Series) -> int:
        if col_score > 10:
            return 1
        else:
            return 0

    def get_labels(self) -> dict:
        """Shorter version of the question name to use in the plots and figures"""
        return {
            # TODO: try to figure a way to put the SS1 RENAME AS SSG because it is a general question so the numbering
            #  should go until 10
            'sss1': 'Unintentional sleep\nlikelihood',
            'sss2': 'On computer/tablet',
            'sss3': 'Talking on\nphone',
            'sss4': 'Meeting',
            'sss5': 'Listening\nSpeaker',
            'sss6': 'Board game',
            'sss7': 'Driving',
            'sss8': 'Videogame',
            'sss9': 'Lying\ndown(SSS)',
            'sss10': 'Traveling\npassenger',
            'sss11': 'Watching\nmovie',
            'sss_score': 'SSS Score',
        }
    def get_latent_factor(self) -> Dict[str, str]:
        return self.latent_factor


class EpworthScale:
    """
    Cut off in the score at 10
    ESS < 10 -> Not sleepiness problem
    """
    def __init__(self):
        self.levels = {'Would never doze':0,
                       'Slight chance of dozing':1,
                       'Moderate chance of dozing':2,
                       'High chance of dozing':3}

        self.questionnaire = {
            'ess1': {'label': 'Sitting/reading',
                     'levels': [*self.levels.keys()]},
            'ess2': {'label': 'Watching TV',
                     'levels': [*self.levels.keys()]},
            'ess3': {'label': 'Sitting inactive',
                     'levels': [*self.levels.keys()]},
            'ess4': {'label': 'Car passenger',
                     'levels': [*self.levels.keys()]},
            'ess5': {'label': 'Lying down (ESS)',
                     'levels': [*self.levels.keys()]},
            'ess6': {'label': 'Talking to someone',
                     'levels': [*self.levels.keys()]},
            'ess7': {'label': 'Sitting after lunch',
                     'levels': [*self.levels.keys()]},
            'ess8': {'label': 'Car in traffic',
                     'levels': [*self.levels.keys()]}
        }

        self.latent_factor = {
            'ess1': 'passive',  # Sitting/reading – low stimulation
            'ess2': 'passive',  # Watching TV – passive entertainment
            'ess3': 'passive',  # Sitting inactive in public – minimal stimulation
            'ess4': 'passive',  # Car passenger – no activity required
            'ess5': 'passive',  # Lying down – extremely low engagement
            'ess6': 'active',  # Talking to someone – interactive
            'ess7': 'passive',  # Sitting after lunch – digestion + rest = low alertness
            'ess8': 'active',  # Car in traffic while stopped – attentional demand (even as driver)
            'ess_score': 'computed',
        }

    def compute_score(self, responses: pd.DataFrame) -> pd.Series:
        """Compute the Epworth Sleepiness Scale (ESS) Score"""
        question = [quest for quest in list(self.questionnaire.keys()) if quest not in ['ess_score']]
        return responses[question].sum(1)

    def apply_cut_off(self, col_score:pd.Series) -> int:
        if col_score > 10:
            return 1
        else:
            return 0
    def get_labels(self) -> dict:
        """Shorter version of the question name to use in the plots and figures"""
        return {
            'ess1': 'Sitting/reading',
            'ess2': 'Watching TV',
            'ess3': 'Sitting\ninactive',
            'ess4': 'Car\npassenger',
            'ess5': 'Lying\ndown(ESS)',
            'ess6': 'Talking to\nsomeone',
            'ess7': 'Sitting after\nlunch',
            'ess8': 'Car in\ntraffic',
            'ess_score': 'ESS Score',
        }
    def get_latent_factor(self) -> Dict[str, str]:
        return self.latent_factor