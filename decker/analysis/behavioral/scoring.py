import pandas as pd
pd.options.mode.chained_assignment = None
import os 
from scipy.stats import ttest_1samp, ttest_ind
import numpy as np

class Data(object):
    """Class for getting all the files you wish to analyze and putting them in a single object
    
    Parameters
    ----------
    path: str, default: current path
        Absolute path to the folder which holds data files.
        Must be in .csv format
    """

    def __init__(self, path=os.path.dirname(os.path.abspath(__name__))):
        self.path = path
        _data = os.listdir(self.path)
        self.files = pd.DataFrame(data={"pwd": path,"files": _data})

    def parse_files(self):
        """Find all of the files you wish to score
        
        Parameters
        ----------
        """

        self.files = self.files[self.files["files"].str.endswith('csv')].reset_index(drop=True)
        
        # return
        self.numfiles = len(self.files)

        return self

    def rm_subs(self, subids: tuple):
        """Remove subjects' files from object

        Parameters
        ----------
        subids: tuple
             Tuple of subject IDs that match the filenames. 
        """

        self.files = self.files.drop(self.files[self.files["files"].str.startswith(subids)].index).reset_index(drop=True)
        self.numfiles = len(self.files)

        return self

    def score(self):
        """Scoring individual participant data"""

        # create empty df for scoring
        anskey = pd.DataFrame(columns=['correct_key_resp', 'actual_key_resp', 'assign_codes', 'score'])

        str_scores = []
        rand_scores = []

        for i, j in self.files.iterrows():

            _i = pd.read_csv(f'{j["pwd"]}/{j["files"]}')

            order = []
            if _i['blocks'][5] == 'block1.csv':
                order = 1
            elif _i['blocks'][5] == 'block12.csv':
                order = 2

            anskey['actual_key_resp'] = list(_i['key_resp.keys'][5:29].dropna())

            # set answer key based on order set a few chunks earlier
            answers =  ['z', 'v', 'v',  'z', 'm', 'z', 'z','v', 'm','v', 'v', 'm']
            if order == 1:
                anskey['correct_key_resp'] = answers
            elif order == 2:
                anskey['correct_key_resp'] = answers[::-1]

            anskey["assign_codes"] = anskey.apply(lambda x: 1 if x["actual_key_resp"] == x["correct_key_resp"] else 0, axis=1)

            # get score
            anskey['score'][0] = (sum(anskey['assign_codes']))/12;
            score = anskey['score'][0];

            #get cond
            cond = j["files"].split('_')[1].split('.')[0]
            if cond == 'structured':
                str_scores.append(score)
            elif cond == 'random':
                rand_scores.append(score)

        str_scores = pd.DataFrame(data={'structured': str_scores})
        rand_scores = pd.DataFrame(data={'random': rand_scores})
        self.scores = pd.concat([rand_scores, str_scores], axis=1)

        return self
    
    def indiv_score(self, subid):
        """Get score and individual responses for a single subject
        
        Parameters
        ----------
        subid: str
            single subject ID that matches filename
        """

        anskey = pd.DataFrame(columns=['correct_key_resp', 'actual_key_resp', 'assign_codes', 'score'])

        file_idx = np.where(self.files["files"].str.startswith(subid))
                
        _i = pd.read_csv(f'{self.files["pwd"][file_idx[0][0]]}/{self.files["files"][file_idx[0][0]]}')

        order = []
        if _i['blocks'][5] == 'block1.csv':
            order = 1
        elif _i['blocks'][5] == 'block12.csv':
            order = 2

        anskey['actual_key_resp'] = list(_i['key_resp.keys'][5:29].dropna())

        # set answer key based on order set a few chunks earlier
        answers =  ['z', 'v', 'v',  'z', 'm', 'z', 'z','v', 'm','v', 'v', 'm']
        if order == 1:
            anskey['correct_key_resp'] = answers
        elif order == 2:
            anskey['correct_key_resp'] = answers[::-1]

        anskey["assign_codes"] = anskey.apply(lambda x: 1 if x["actual_key_resp"] == x["correct_key_resp"] else 0, axis=1)

        # get score
        anskey['score'][0] = (sum(anskey['assign_codes']))/12;

        return anskey
    
    def cat_score(self):
        """Get accuracy for other stimuli categories (e.g., foils or part words)
        
        """
        cat_anskey_columns = ['foil_correct_key_resp', 'part_correct_key_resp', 'actual_key_resp', 'foil_codes', 'part_codes', 'foil_score', 'part_score']
        cat_anskey = pd.DataFrame(columns=cat_anskey_columns)

        category_scores = {
            'rand_foil_scores': [],
            'rand_part_scores': [],
            'str_foil_scores': [],
            'str_part_scores': []
        }

        foil_answers =  ['v', 'm', 'm', 'v', 'z', 'm', 'm', 'm', 'v', 'z', 'z', 'z']
        part_answers = ['m', 'z', 'z', 'm', 'v', 'v', 'v', 'z', 'z', 'm', 'm', 'v']

        for i, j in self.files.iterrows():
            _i = pd.read_csv(f'{j["pwd"]}/{j["files"]}')
            order = 1 if _i['blocks'][5] == 'block1.csv' else 2

            cat_anskey['actual_key_resp'] = list(_i['key_resp.keys'][5:29].dropna())

            if order == 1:
                cat_anskey['foil_correct_key_resp'] = foil_answers
                cat_anskey['part_correct_key_resp'] = part_answers
            elif order == 2:
                cat_anskey['foil_correct_key_resp'] = foil_answers[::-1]
                cat_anskey['part_correct_key_resp'] = part_answers[::-1]

            cat_anskey["foil_codes"] = cat_anskey.apply(lambda x: int(x["actual_key_resp"] == x['foil_correct_key_resp']), axis=1)
            cat_anskey["part_codes"] = cat_anskey.apply(lambda x: int(x["actual_key_resp"] == x['part_correct_key_resp']), axis=1)

            cat_anskey['foil_score'] = sum(cat_anskey['foil_codes']) / 12
            cat_anskey['part_score'] = sum(cat_anskey['part_codes']) / 12

            cond = j["files"].split('_')[1].split('.')[0]
            if cond == 'structured':
                category_scores['str_foil_scores'].append(cat_anskey['foil_score'].iloc[0])
                category_scores['str_part_scores'].append(cat_anskey['part_score'].iloc[0])
            elif cond == 'random':
                category_scores['rand_foil_scores'].append(cat_anskey['foil_score'].iloc[0])
                category_scores['rand_part_scores'].append(cat_anskey['part_score'].iloc[0])

        self.category_scores = pd.DataFrame(category_scores)

        return self

        
class Stats(Data):
    """Class to compute a 1 sample, 1 tailed t-test or two-samples independant, two tailed t-test
    """
    def __init__(self, scores):
        super().__init__() # gets 'self' from previous class

        self.scores=scores

    def compute(self, test, **kwargs):
        '''Method for calculating scores
        
        Parameters
        ----------
        test: str or int
            Choose which statistical test you want to computer. 
            1 or '1samp' = 1 sample, 1 tailed t-test
            2 or 'ind' = two samples, independet, two-tailed t-test

        Keyword Arguments
        -----------------
        mu: int
            Population parameter you wish to test agains. Only necessary if test = 1samp
        '''
                
        if test == 1 or test == '1samp':
            self.test = '1samp'    
        elif test == 2 or test =='ind':
            self.test = 'ind'

        self.mu = kwargs.get('mu')
        if self.mu is None and self.test == '1samp':
            raise KeyError('When conducting a 1 sample t-test, you must provide the population parameter (mu). \n +\
                           In the case of this project, mu= (1/3)')

        if self.test == '1samp':
             # compute test
            self.statistic = ttest_1samp(self.scores['structured'], popmean=self.mu, alternative='greater', nan_policy='omit')

        elif self.test == 'ind':
            # compute test
            self.statistic = ttest_ind(a=self.scores['structured'], b=self.scores['random'], equal_var=True, alternative='greater', nan_policy='omit')

        return self