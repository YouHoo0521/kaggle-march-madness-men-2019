import pystan
import pickle
import argparse
import os
from src.utils import get_project_root  # see src/ folder in project repo
from src.data import make_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss


MODEL_BASEDIR = os.path.join(get_project_root(), 'models')


def create_poisson_model():
    model_code = '''
    /*
    pairwise poisson regression model of count boxscores

    */

    data {
      int<lower=0> N_teams;
      int<lower=0> N;  // number of games in regular season
      int<lower=0> N_tourney;  // number of games in tournament
      int<lower=1, upper=N_teams> j_team[N + N_tourney];  // index for team 1
      int<lower=1, upper=N_teams> k_team[N + N_tourney];  // index for team 2
      int<lower=0> y_att[N];  // count outcome

      real alpha0[N_teams];  // prior for offense effects
      real beta0[N_teams];   // prior for defense effects

      real weights[N];  // weights on observations

    }

    transformed data {

    }

    parameters {
      real alpha[N_teams];  // team effect
      real beta[N_teams];  // team scorediff effect
      real<lower=0> sigma_alpha;  // std for team levels
      real<lower=0> sigma_beta;  // std for scorediff effect
    }

    transformed parameters {

    }

    model {
      vector[N] eta;  // linear predictor

      sigma_alpha ~ normal(0, 20);
      sigma_beta ~ normal(0, 20);
      alpha ~ normal(alpha0, sigma_alpha); // team levels
      beta ~ normal(beta0, sigma_beta); // team levels

      for(n in 1:N) {
        eta[n] = alpha[j_team[n]] - beta[k_team[n]];
        // likelihoods
        target +=  poisson_log_lpmf(y_att[n] | eta[n]) * weights[n];
      }

    }

    generated quantities {
      int<lower=0> y_att_new[N_tourney];
      vector[N_tourney] eta_new; // linear effect
      for(n in 1:N_tourney) {
        eta_new[n] = alpha[j_team[N+n]] - beta[k_team[N+n]];
      }
      y_att_new = poisson_log_rng(eta_new);
    }
    '''
    sm = pystan.StanModel(model_code=model_code)
    return sm


def create_binomial_model():
    model_code = '''
    /*
    pairwise binomial model of count boxscores

    */

    data {
      int<lower=0> N_teams;
      int<lower=0> N;  // number of games in regular season
      int<lower=0> N_tourney;  // number of games in tournament
      int<lower=1, upper=N_teams> j_team[N + N_tourney];  // index for team 1
      int<lower=1, upper=N_teams> k_team[N + N_tourney];  // index for team 2
      int<lower=0> y_att[N];   // attempt count
      int<lower=0> y_made[N];  // made count

      real alpha0[N_teams];  // prior for offense effects
      real beta0[N_teams];   // prior for defense effects

      real weights[N];  // weights on observations

    }

    transformed data {

    }

    parameters {
      real alpha[N_teams];  // team effect
      real beta[N_teams];  // team scorediff effect
      real<lower=0> sigma_alpha;  // std for team levels
      real<lower=0> sigma_beta;  // std for scorediff effect
    }

    transformed parameters {

    }

    model {
      vector[N] eta;  // linear predictor

      sigma_alpha ~ normal(0, 20);
      sigma_beta ~ normal(0, 20);
      alpha ~ normal(alpha0, sigma_alpha); // team levels
      beta ~ normal(beta0, sigma_beta); // team levels

      for(n in 1:N) {
        eta[n] = alpha[j_team[n]] - beta[k_team[n]];
        // likelihoods
        target +=  binomial_logit_lpmf(y_made[n] | y_att[n], eta[n]) * weights[n];
      }

    }

    generated quantities {
      vector[N_tourney] eta_new; // linear effect
      vector[N_tourney] pi_new;
      for(n in 1:N_tourney) {
        eta_new[n] = alpha[j_team[N+n]] - beta[k_team[N+n]];
      }
      pi_new = inv_logit(eta_new);
    }
    '''
    sm = pystan.StanModel(model_code=model_code)
    return sm


def create_poisson_data(y_att='fga', year=2015):
    data = make_dataset.get_train_data_v1(year, detailed=True).reset_index()
    common_cols = ['index', 'tourney', 'daynum', 'ID']
    team1_cols = ['team1', 'team2', 'confabbrev2', 'fga_team1', 'fgm_team1', 'fga3_team1', 'fgm3_team1', 'fta_team1', 'ftm_team1']
    team2_cols = ['team2', 'team1', 'confabbrev1', 'fga_team2', 'fgm_team2', 'fga3_team2', 'fgm3_team2', 'fta_team2', 'ftm_team2']
    new_cols = ['team', 'opp', 'conf_opp', 'fga', 'fgm', 'fga3', 'fgm3', 'fta', 'ftm']
    # duplicate and stack data
    data = pd.DataFrame(np.vstack(
        [data.loc[data['tourney'] == 0, common_cols + team1_cols].values,
         data.loc[data['tourney'] == 0, common_cols + team2_cols].values,
         data.loc[data['tourney'] == 1, common_cols + team1_cols].values,
         data.loc[data['tourney'] == 1, common_cols + team2_cols].values]),
			columns=common_cols + new_cols)
    data['fga2'] = data['fga'] - data['fga3']
    data['fgm2'] = data['fgm'] - data['fgm3']
    N = (data.tourney == 0).sum()
    N_tourney = (data.tourney == 1).sum()
    # teams
    teams = set(data['team'].unique())
    team_f2id = dict(enumerate(teams, 1))  # start from 1 for stan's one-based indexing
    team_id2f = {v:k for k, v in team_f2id.items()}
    N_teams = len(teams)
    # weights and priors
    weights = np.ones(N)
    alpha0 = np.ones(N_teams) #  * (np.log(data['fga'].mean())  + 2)
    beta0 = np.ones(N_teams)
    # data dict for stan
    stan_data = {
        'N_teams': N_teams,
        'N': N,
        'N_tourney': N_tourney,
        'j_team': data['team'].map(team_id2f).values,
        'k_team': data['opp'].map(team_id2f).values,
        'y_att': data.loc[data.tourney == 0, y_att].astype(int),
        'weights': weights,
        'alpha0': alpha0,
        'beta0': beta0,
    }
    misc = {'data': data, 'team_f2id': team_f2id}
    return stan_data, misc


def create_binomial_data(y_made='fgm', y_att='fga', year=2015):
    data = make_dataset.get_train_data_v1(year, detailed=True).reset_index()
    common_cols = ['index', 'tourney', 'daynum', 'ID']
    team1_cols = ['team1', 'team2', 'confabbrev2', 'fga_team1', 'fgm_team1', 'fga3_team1', 'fgm3_team1', 'fta_team1', 'ftm_team1']
    team2_cols = ['team2', 'team1', 'confabbrev1', 'fga_team2', 'fgm_team2', 'fga3_team2', 'fgm3_team2', 'fta_team2', 'ftm_team2']
    new_cols = ['team', 'opp', 'conf_opp', 'fga', 'fgm', 'fga3', 'fgm3', 'fta', 'ftm']
    # duplicate and stack data
    data = pd.DataFrame(np.vstack(
        [data.loc[data['tourney'] == 0, common_cols + team1_cols].values,
         data.loc[data['tourney'] == 0, common_cols + team2_cols].values,
         data.loc[data['tourney'] == 1, common_cols + team1_cols].values,
         data.loc[data['tourney'] == 1, common_cols + team2_cols].values]),
			columns=common_cols + new_cols)
    data['fga2'] = data['fga'] - data['fga3']
    data['fgm2'] = data['fgm'] - data['fgm3']
    N = (data.tourney == 0).sum()
    N_tourney = (data.tourney == 1).sum()
    # teams
    teams = set(data['team'].unique())
    team_f2id = dict(enumerate(teams, 1))  # start from 1 for stan's one-based indexing
    team_id2f = {v:k for k, v in team_f2id.items()}
    N_teams = len(teams)
    # weights and priors
    weights = np.ones(N)
    alpha0 = np.ones(N_teams) #  * (np.log(data['fga'].mean())  + 2)
    beta0 = np.ones(N_teams)
    # data dict for stan
    stan_data = {
        'N_teams': N_teams,
        'N': N,
        'N_tourney': N_tourney,
        'j_team': data['team'].map(team_id2f).values,
        'k_team': data['opp'].map(team_id2f).values,
        'y_att': data.loc[data.tourney == 0, y_att].astype(int),
        'y_made': data.loc[data.tourney == 0, y_made].astype(int),
        'weights': weights,
        'alpha0': alpha0,
        'beta0': beta0,
    }
    misc = {'data': data, 'team_f2id': team_f2id}
    return stan_data, misc


def fit_poisson_model(year=2015, y_att='fga',
                      model_fname='tmp_model_poisson.pkl', fit_fname='tmp_fit_poisson.pkl',
                      num_chains=4, num_iter=500):
    # compmile model
    stan_data, misc = create_poisson_data(y_att=y_att, year=year)
    if os.path.exists(model_fname):
        with open(model_fname, 'rb') as f:
            pickle_data = pickle.load(f)
            sm_poisson = pickle_data['sm_poisson']
    else:
        sm_poisson = create_poisson_model()
        with open(model_fname, "wb") as f:
            pickle.dump({'sm_poisson': sm_poisson}, f, protocol=-1)
    # fit model
    fit_poisson = sm_poisson.sampling(data=stan_data, iter=num_iter, chains=num_chains)
    with open(fit_fname, "wb") as f:
        pickle.dump({'sm_poisson': sm_poisson, 'fit_poisson': fit_poisson}, f, protocol=-1)
    return sm_poisson, fit_poisson


def fit_binomial_model(year=2015, y_made='fgm', y_att='fga',
                       model_fname='tmp_model_binomial.pkl', fit_fname='tmp_fit_binomial.pkl',
                       num_chains=4, num_iter=500):
    # compmile model
    stan_data, misc = create_binomial_data(y_made=y_made, y_att=y_att, year=year)
    if os.path.exists(model_fname):
        with open(model_fname, 'rb') as f:
            pickle_data = pickle.load(f)
            sm_binomial = pickle_data['sm_binomial']
    else:
        sm_binomial = create_binomial_model()
        with open(model_fname, "wb") as f:
            pickle.dump({'sm_binomial': sm_binomial}, f, protocol=-1)
    # fit model
    fit_binomial = sm_binomial.sampling(data=stan_data, iter=num_iter, chains=num_chains)
    with open(fit_fname, "wb") as f:
        pickle.dump({'sm_binomial': sm_binomial, 'fit_binomial': fit_binomial}, f, protocol=-1)
    return sm_binomial, fit_binomial


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='tmp')
    parser.add_argument('--year', type=int, default=2015)
    parser.add_argument('--num_iter', type=int, default=500)
    parser.add_argument('--num_chains', type=int, default=4)
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    # create model and fit directories
    model_directory = os.path.join(MODEL_BASEDIR, args.model_name)
    fit_directory = os.path.join(model_directory, '{}'.format(args.year))
    os.makedirs(fit_directory, exist_ok=True)
    # pickle files for model and fit
    model_fname_poisson = os.path.join(model_directory, 'poisson_model.pkl')
    # pickle files for model and fit
    model_fname_binomial = os.path.join(model_directory, 'binomial_model.pkl')
    pred_fname = os.path.join(fit_directory, 'pred.csv')
    sm_poisson_dict = {}
    fit_poisson_dict = {}
    sm_binomial_dict = {}
    fit_binomial_dict = {}
    if not args.predict:
        for y_att, y_made in zip(['fga', 'fga3', 'fta'], ['fgm', 'fgm3', 'ftm']):
            fit_fname_poisson = os.path.join(fit_directory, 'poisson_fit_{}.pkl'.format(y_att))
            fit_fname_binomial = os.path.join(fit_directory, 'binomial_fit_{}.pkl'.format(y_att))
            print('Fitting poisson model for: {}'.format(y_att))
            sm_poisson_dict[y_att], fit_poisson_dict[y_att] = fit_poisson_model(
                year=args.year,
                y_att=y_att,
                model_fname=model_fname_poisson,
                fit_fname=fit_fname_poisson,
                num_chains=args.num_chains,
                num_iter=args.num_iter)
            print('Fitting binomial model for: {}'.format(y_att))
            sm_binomial_dict[y_att], fit_binomial_dict[y_att] = fit_binomial_model(
                year=args.year,
                y_att=y_att,
                y_made=y_made,
                model_fname=model_fname_binomial,
                fit_fname=fit_fname_binomial,
                num_chains=args.num_chains,
                num_iter=args.num_iter)
    else:
        for y_att in ['fga', 'fga3', 'fta']:
            fit_fname_poisson = os.path.join(fit_directory, 'poisson_fit_{}.pkl'.format(y_att))
            fit_fname_binomial = os.path.join(fit_directory, 'binomial_fit_{}.pkl'.format(y_att))
            with open(fit_fname_poisson, 'rb') as f:
                pickle_data = pickle.load(f)
                sm_poisson_dict[y_att] = pickle_data['sm_poisson']
                fit_poisson_dict[y_att] = pickle_data['fit_poisson']
            with open(fit_fname_binomial, 'rb') as f:
                pickle_data = pickle.load(f)
                sm_binomial_dict[y_att] = pickle_data['sm_binomial']
                fit_binomial_dict[y_att] = pickle_data['fit_binomial']
        pass
    # process results
    la_poisson_dict = {}
    la_binomial_dict = {}
    for y_att in ['fga', 'fga3', 'fta']:
        la_poisson_dict[y_att] = fit_poisson_dict[y_att].extract()
        la_binomial_dict[y_att] = fit_binomial_dict[y_att].extract()
    data = make_dataset.get_train_data_v1(args.year, detailed=True)
    fg2_made_sim = (la_poisson_dict['fga']['y_att_new'] - la_poisson_dict['fga3']['y_att_new']) * la_binomial_dict['fga']['pi_new']
    fg3_made_sim = la_poisson_dict['fga3']['y_att_new'] * la_binomial_dict['fga3']['pi_new']
    ft_made_sim = la_poisson_dict['fta']['y_att_new'] * la_binomial_dict['fta']['pi_new']
    points_sim = fg2_made_sim * 2 + fg3_made_sim * 3 + ft_made_sim

    points_pred = np.mean(points_sim, axis=0)
    points_actual = np.concatenate([data.loc[data['tourney'] == 1,'score1'].values,
                                    data.loc[data['tourney'] == 1,'score2'].values])
    points_error = points_pred - points_actual
    print(np.quantile(points_error, q=[0.10, 0.25, 0.5, 0.75, 0.90]))
    n_pred = int(points_sim.shape[1]/2)
    prob_win = np.mean(points_sim[:,:n_pred] > points_sim[:,n_pred:], axis=0)
    win_actual = (data.loc[data['tourney'] == 1, 'team1win'] == 1).values
    correct = win_actual == (prob_win > 0.5)
    accuracy = np.mean(correct)
    loss = log_loss(win_actual, prob_win)
    print("year = {}\taccuracy = {}\t loss = {}".format(args.year, accuracy, loss))
    # save prediction
    df_pred = pd.DataFrame({'ID':data.loc[data['tourney'] == 1, 'ID'], 'Pred':prob_win})
    df_pred.to_csv(pred_fname, index=False)
