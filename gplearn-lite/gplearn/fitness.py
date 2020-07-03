"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numbers

import numpy as np
from scipy.stats import rankdata
import scipy.io as sio
import statsmodels.api as sm

__all__ = ['make_fitness']


class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

        neutral_fac_mat = sio.loadmat('neutral_fac20.mat')
        self.neutral_fac = neutral_fac_mat['neutral_fac']

    def __call__(self, *args):
        if '_rank_ic' in str(self.function) or '_top_return' in str(self.function):
            return self.function(*args, self.neutral_fac)
        else:
            return self.function(*args)


def make_fitness(function, greater_is_better):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    '''
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')
    '''
    return _Fitness(function, greater_is_better)

def _rank_ic(y, y_pred, p, w, neutral_fac):
    from scipy import stats
    import pandas as pd
    import math

    df_y_pred = pd.DataFrame()
    df_y_pred['y_pred'] = y_pred
    df_y_check = df_y_pred.dropna()
    if df_y_check.shape[0] == 0:
        return 0

    factor_style = pd.DataFrame()
    factor_style['log_mkt'] = neutral_fac[0][0][0][:, p-1]
    factor_style['turn20'] = neutral_fac[0][0][2][:, p-1]
    factor_style['std20'] = neutral_fac[0][0][3][:, p-1]
    factor_style['return20'] = neutral_fac[0][0][4][:, p-1]
    factor_style['std10_close'] = neutral_fac[0][0][5][:, p-1]
    factor_indus_dummy = pd.get_dummies(neutral_fac[0][0][1][:, p-1])
    factor_neutral = pd.concat([factor_style, factor_indus_dummy], axis=1)
    columns = factor_neutral.columns

    tmp_concat = pd.concat([df_y_pred, factor_neutral], axis=1)
    tmp_concat = tmp_concat.dropna()
    tmp_corr = tmp_concat.corr().iloc[0, 1:]
    # 如果相关系数大于0.9，删除该风格或行业因子
    factor_neutral.drop(columns[list(np.where(tmp_corr.abs() > 0.9)[0])], axis=1, inplace=True)

    #print(factor_neutral.shape)
    model = sm.OLS(df_y_pred,factor_neutral, missing='drop')
    results = model.fit()

    #factor_output = results.resid
    # 取残差为中性化后的因子
    factor_output = pd.Series(index=df_y_pred.index)
    factor_output[results.fittedvalues.index] = df_y_pred.iloc[results.fittedvalues.index, 0] - results.fittedvalues

    df = pd.DataFrame()
    df['y'] = y
    df['y_pred'] = factor_output
    df = df.dropna()
    #corr = stats.spearmanr(df['y'], df['y_pred'])[0]
    #corr = np.corrcoef(df.iloc[:,0], df.iloc[:,1])[0,1]
    corr = stats.pearsonr(df['y'], df['y_pred'])[0]
    #corr = _weighted_pearson(df.iloc[:, 0], df.iloc[:, 1], w[:df.shape[0]])

    if math.isnan(corr):
        corr = 0
    return corr


def _top_return(y, y_pred, p, w, neutral_fac):
    import pandas as pd

    df_y_pred = pd.DataFrame()
    df_y_pred['y_pred'] = y_pred
    df_y_check = df_y_pred.dropna()
    if df_y_check.shape[0] == 0:
        return 0

    factor_style = pd.DataFrame()
    factor_style['log_mkt'] = neutral_fac[0][0][0][:, p - 1]
    factor_style['turn20'] = neutral_fac[0][0][2][:, p - 1]
    factor_style['std20'] = neutral_fac[0][0][3][:, p - 1]
    factor_style['return20'] = neutral_fac[0][0][4][:, p - 1]
    factor_style['std10_close'] = neutral_fac[0][0][5][:, p - 1]
    factor_indus_dummy = pd.get_dummies(neutral_fac[0][0][1][:, p - 1])
    factor_neutral = pd.concat([factor_style, factor_indus_dummy], axis=1)

    tmp_concat = pd.concat([df_y_pred, factor_neutral], axis=1)
    tmp_concat = tmp_concat.dropna()
    tmp_corr = tmp_concat.corr().iloc[0, 1:]
    # 如果相关系数大于0.9，删除该风格或行业因子
    factor_neutral.drop(list(np.where(tmp_corr.abs() > 0.9)[0]), axis=1, inplace=True)

    # print(factor_neutral.shape)
    model = sm.OLS(df_y_pred, factor_neutral, missing='drop')
    results = model.fit()

    # factor_output = results.resid
    # 取残差为中性化后的因子
    factor_output = pd.Series(index=df_y_pred.index)
    factor_output[results.fittedvalues.index] = df_y_pred.iloc[results.fittedvalues.index, 0] - results.fittedvalues

    df = pd.DataFrame()
    y = (y - np.mean(y)) / np.std(y)
    df['y'] = y
    df['y_pred'] = factor_output
    df = df.dropna()
    df = df.sort_values(ascending=False, by='y_pred')
    return np.mean(df.iloc[:100, 0])

def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.

def _weighted_spearman(y, y_pred, w):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)


def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _log_loss(y, y_pred, w):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score, weights=w)


weighted_pearson = make_fitness(function=_weighted_pearson,
                                greater_is_better=True)
weighted_spearman = make_fitness(function=_weighted_spearman,
                                 greater_is_better=True)
mean_absolute_error = make_fitness(function=_mean_absolute_error,
                                   greater_is_better=False)
mean_square_error = make_fitness(function=_mean_square_error,
                                 greater_is_better=False)
root_mean_square_error = make_fitness(function=_root_mean_square_error,
                                      greater_is_better=False)
log_loss = make_fitness(function=_log_loss, greater_is_better=False)

rank_ic = make_fitness(function=_rank_ic, greater_is_better=True)

top_return = make_fitness(function=_top_return, greater_is_better=True)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss,
                'rank_ic' : rank_ic,
                'top_return': top_return}
