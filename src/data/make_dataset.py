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


def get_boxscore_dataset_v1(season=None):
    '''
    Extend train_data_v1 with seasonwise mean/std boxscore columns for each team and opponent
    '''
    data = get_train_data_v1(season=season) # main data
    ##################################################
    # regular season boxscore data
    ##################################################
    RegularSeasonDetailedResults = pd.read_csv(
        os.path.join(DATAFILES_BASEDIR, 'RegularSeasonDetailedResults.csv'))
    ##################################################
    # column processing
    ##################################################
    cols = RegularSeasonDetailedResults.columns
    w_cols = (cols.str.slice(0, 1) == 'W') & (~cols.isin(['WLoc']))
    l_cols = cols.str.slice(0, 1) == 'L'
    box_colnames = cols[w_cols].str.slice(1)  # remove 'W' and 'L'
    # for reversing W columns with L cols
    reverse_dict = dict(zip(list('W' + box_colnames) + list('L' + box_colnames),
                            list('L' + box_colnames) + list('W' + box_colnames)))
    # for converting W and L boxstats to team and opponent boxstats
    rename_dict = dict(zip(list('W' + box_colnames) + list('L' + box_colnames),
                           list(box_colnames + '_team') + list(box_colnames + '_opp')))
    ##################################################
    # stack the original and reversed dataframes
    ##################################################
    RegularSeasonDetailedResultsStacked = pd.concat(
        [RegularSeasonDetailedResults,
         RegularSeasonDetailedResults.rename(reverse_dict, axis=1)],
        sort=True).rename(rename_dict, axis=1)
    n = RegularSeasonDetailedResults.shape[0]
    RegularSeasonDetailedResultsStacked['win'] = np.array([True] * n + [False] * n)
    ##################################################
    # calculate boxscore stats
    ##################################################
    df_boxstat = (RegularSeasonDetailedResultsStacked[list(rename_dict.values()) + ['Season']]
                  .groupby(['Season', 'TeamID_team'])
                  .agg(['mean', 'std']))
    df_boxstat.columns = ['_'.join(col).strip() for col in df_boxstat.columns.values]
    df_boxstat.columns = df_boxstat.columns.str.lower()
    drop_cols = df_boxstat.columns[df_boxstat.columns.str.contains('teamid_opp')]
    df_boxstat = df_boxstat.drop(drop_cols, axis=1)
    df_boxstat.index.names = ['Season', 'TeamID']
    ##################################################
    # merge with main data
    ##################################################
    data = (data
            .pipe(pd.merge, df_boxstat,
                  left_on=['season', 'team1'], right_index=True,
                  how='left')
            .pipe(pd.merge, df_boxstat,
                  left_on=['season', 'team2'], right_index=True,
                  how='left', suffixes=('1', '2'))
            )
    return data


if __name__ == '__main__':
    pass
