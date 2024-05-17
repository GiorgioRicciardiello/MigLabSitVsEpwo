"""
Definition of the sleep scale questionnaire to be used in the study
"""
import pandas as pd

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
            'ess1': {'label': 'Sitting and reading',
                     'levels': [*self.levels.keys()]},
            'ess2': {'label': 'Watching TV',
                     'levels': [*self.levels.keys()]},
            'ess3': {'label': 'Sitting, inactive in a public place (e.g., a theater or a meeting)',
                     'levels': [*self.levels.keys()]},
            'ess4': {'label': 'As a passenger in a car for an hour without a break',
                     'levels': [*self.levels.keys()]},
            'ess5': {'label': 'Lying down to rest in the afternoon when circumstances permit',
                     'levels': [*self.levels.keys()]},
            'ess6': {'label': 'Sitting and talking to someone',
                     'levels': [*self.levels.keys()]},
            'ess7': {'label': 'Sitting quietly after lunch without alcohol',
                     'levels': [*self.levels.keys()]},
            'ess8': {'label': 'In a car, while stopped for a few minutes in traffic',
                     'levels': [*self.levels.keys()]}
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
