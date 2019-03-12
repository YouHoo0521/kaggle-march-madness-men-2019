import pandas as pd
import numpy as np
import os
from src.utils import get_project_root


DATAFILES_BASEDIR = os.path.join(get_project_root(), 'input/datafiles/')


def get_train_data_v1(season=None):
    ##################################################
    # read data
    ##################################################
    RegularSeasonCompactResults = pd.read_csv(
        os.path.join(DATAFILES_BASEDIR, 'RegularSeasonCompactResults.csv'))
    NCAATourneyCompactResults = pd.read_csv(
        os.path.join(DATAFILES_BASEDIR, 'NCAATourneyCompactResults.csv'))
    NCAATourneySeeds = pd.read_csv(
        os.path.join(DATAFILES_BASEDIR, 'NCAATourneySeeds.csv'))
    ##################################################
    # process data
    ##################################################
    NCAATourneySeeds['seednum'] = NCAATourneySeeds['Seed'].str.slice(1, 3).astype(int)
    RegularSeasonCompactResults['tourney'] = 0
    NCAATourneyCompactResults['tourney'] = 1
    # combine regular and tourney data
    data = pd.concat([RegularSeasonCompactResults, NCAATourneyCompactResults])
    if season:
        data = data[data.Season == season]  # filter season
    ##################################################
    # team1: team with lower id
    data['team1'] = (data['WTeamID'].where(data['WTeamID'] < data['LTeamID'],
                                           data['LTeamID']))
    # team2: team with higher id
    data['team2'] = (data['WTeamID'].where(data['WTeamID'] > data['LTeamID'],
                                           data['LTeamID']))
    data['team1'] = (data['WTeamID'].where(data['WTeamID'] < data['LTeamID'],
                                           data['LTeamID']))
    data['score1'] = data['WScore'].where(data['WTeamID'] < data['LTeamID'], data['LScore'])
    data['score2'] = data['WScore'].where(data['WTeamID'] > data['LTeamID'], data['LScore'])
    data['loc'] = (data['WLoc']
                   .where(data['WLoc'] != 'H', data['WTeamID'])
                   .where(data['WLoc'] != 'A', data['LTeamID'])
                   .where(data['WLoc'] != 'N', 0))  # 0 if no home court
    data['team1win'] = np.where(data['WTeamID'] == data['team1'], 1, 0)
    ##################################################
    # get tourney seeds
    data = (data
            .pipe(pd.merge, NCAATourneySeeds,
                  left_on=['Season', 'team1'], right_on=['Season', 'TeamID'],
                  how='left')
            .pipe(pd.merge, NCAATourneySeeds,
                  left_on=['Season', 'team2'], right_on=['Season', 'TeamID'],
                  how='left', suffixes=('1', '2'))
            )
    data['seeddiff'] = data['seednum2'] - data['seednum1']
    data = data.drop(['TeamID1', 'TeamID2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc'], axis=1)
    data.columns = data.columns.str.lower()
    data['ID'] = (data[['season', 'team1', 'team2']].astype(str)
                  .apply(lambda x: '_'.join(x), axis=1))
    return data


if __name__ == '__main__':
    pass
