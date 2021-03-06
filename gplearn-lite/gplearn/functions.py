"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    type : int
        The type of function
    """

    def __init__(self, function, name, arity, _type):
        self.function = function
        self.name = name
        self.arity = arity
        self.type = _type

    def __call__(self, *args):
        return self.function(*args)


def make_function(function, name, arity, _type='normal'):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    """
    '''
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)
    '''
    return _Function(function, name, arity, _type)

'''
def _ts_argmax(x1, window):
    data = x1[:,-window:]
    argmax = np.nanargmax(data, axis = 1)+1.0
    result = argmax
    for i in range(1, x1.shape[1]-window):
        data = x1[:, -window-i:-i]
        argmax = np.nanargmax(data, axis = 1)+1.0
        result = np.column_stack([argmax, result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _ts_argmin(x1, window):
    data = x1[:,-window:]
    argmax = np.nanargmin(data, axis = 1)+1.0
    result = argmax
    for i in range(1, x1.shape[1]-window):
        data = x1[:, -window-i:-i]
        argmax = np.nanargmin(data, axis = 1)+1.0
        result = np.column_stack([argmax, result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result
'''

def _ts_argmax(x1, window):
    data = x1[:,-window:]
    isnan = np.isnan(data[:,-1])
    argmax = np.argmax(data, axis = 1)+1.0
    argmax[isnan] = np.nan
    result = argmax
    for i in range(1, x1.shape[1]-window):
        data = x1[:, -window-i:-i]
        isnan = np.isnan(data[:,-1])
        argmax = np.argmax(data, axis = 1)+1.0
        argmax[isnan] = np.nan
        result = np.column_stack([argmax, result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _ts_argmin(x1, window):
    data = x1[:,-window:]
    isnan = np.isnan(data[:,-1])
    argmax = np.argmin(data, axis = 1)+1.0
    argmax[isnan] = np.nan
    result = argmax
    for i in range(1, x1.shape[1]-window):
        data = x1[:, -window-i:-i]
        isnan = np.isnan(data[:,-1])
        argmax = np.argmin(data, axis = 1)+1.0
        argmax[isnan] = np.nan
        result = np.column_stack([argmax, result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _delay(x1, window):
    result = x1[:,-window]
    for i in range(1, x1.shape[1]-window):
        result = np.column_stack([x1[:,-window-i], result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _delta(x1, window):
    result = x1[:,-1] / x1[:,-window] - 1
    for i in range(1, x1.shape[1]-window):
        result = np.column_stack([x1[:,-1-i] / x1[:,-window-i] - 1, result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _ts_stddev(x1, window):
    result = np.nanstd(x1[:, -window:], axis=1, ddof=1)
    for i in range(1, x1.shape[1]-window):
        result = np.column_stack([np.nanstd(x1[:, -window-i:-i], axis=1, ddof=1), result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _ts_sum(x1, window):
    result = np.nansum(x1[:, -window:], axis=1)
    for i in range(1, x1.shape[1]-window):
        result = np.column_stack([np.nansum(x1[:, -window-i:-i], axis=1), result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _ts_max(x1, window):
    result = np.nanmax(x1[:, -window:], axis=1)
    for i in range(1, x1.shape[1]-window):
        result = np.column_stack([np.nanmax(x1[:, -window-i:-i], axis=1), result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _ts_min(x1, window):
    result = np.nanmin(x1[:, -window:], axis=1)
    for i in range(1, x1.shape[1]-window):
        result = np.column_stack([np.nanmin(x1[:, -window-i:-i], axis=1), result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _ts_nanmean(x1, window):
    result = np.nanmean(x1[:, -window:], axis=1)
    for i in range(1, x1.shape[1]-window):
        result = np.column_stack([np.nanmean(x1[:, -window-i:-i], axis=1), result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _ts_prod(x1, window):
    result = np.nanprod(x1[:, -window:], axis=1)
    for i in range(1, x1.shape[1]-window):
        result = np.column_stack([np.nanprod(x1[:, -window-i:-i], axis=1), result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _ts_rank(x1, window):
    data = pd.DataFrame(x1[:,-window:])
    result = data.rank(axis = 1).iloc[:,-1].values/window
    for i in range(1, x1.shape[1]-window):
        data = pd.DataFrame(x1[:, -window-i:-i])
        result = np.column_stack([data.rank(axis = 1).iloc[:,-1].values/window, result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    return result

def _rank(x1):
    data = pd.DataFrame(x1)
    rank = data.rank().values
    return rank/np.nanmax(rank)

def _ts_covariance(x1, x2, window):
    data1 = pd.DataFrame(x1[:,-window:].T)
    data2 = pd.DataFrame(x2[:,-window:].T)
    result = data1.covwith(data2).values
    for i in range(1, x1.shape[1]-window):
        data1 = pd.DataFrame(x1[:, -window-i:-i].T)
        data2 = pd.DataFrame(x2[:, -window-i:-i].T)
        result = np.column_stack([data1.covwith(data2).values,result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    data = pd.DataFrame(result)
    rank = data.rank().values
    return rank/np.nanmax(rank)

def _ts_correlation(x1, x2, window):
    data1 = pd.DataFrame(x1[:,-window:].T)
    data2 = pd.DataFrame(x2[:,-window:].T)
    result = data1.corrwith(data2).values
    for i in range(1, x1.shape[1]-window):
        data1 = pd.DataFrame(x1[:, -window-i:-i].T)
        data2 = pd.DataFrame(x2[:, -window-i:-i].T)
        result = np.column_stack([data1.corrwith(data2).values,result])
    for i in range(x1.shape[1]-window, x1.shape[1]):
        result = np.column_stack([result[:,0], result])
    data = pd.DataFrame(result)
    rank = data.rank().values
    return rank/np.nanmax(rank)

def _protected_rank_add(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    data = pd.DataFrame(x1)
    rank = data.rank().values
    rank1 = rank / np.nanmax(rank)
    data = pd.DataFrame(x2)
    rank = data.rank().values
    rank2 = rank / np.nanmax(rank)

    return  np.add(rank1, rank2)

def _protected_rank_sub(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    data = pd.DataFrame(x1)
    rank = data.rank().values
    rank1 = rank / np.nanmax(rank)
    data = pd.DataFrame(x2)
    rank = data.rank().values
    rank2 = rank / np.nanmax(rank)

    return  np.subtract(rank1, rank2)

def _protected_rank_mul(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    data = pd.DataFrame(x1)
    rank = data.rank().values
    rank1 = rank / np.nanmax(rank)
    data = pd.DataFrame(x2)
    rank = data.rank().values
    rank2 = rank / np.nanmax(rank)

    return  np.multiply(rank1, rank2)

def _protected_rank_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    data = pd.DataFrame(x1)
    rank = data.rank().values
    rank1 = rank / np.nanmax(rank)
    data = pd.DataFrame(x2)
    rank = data.rank().values
    rank2 = rank / np.nanmax(rank)

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(rank2) > 0.001, np.divide(rank1, rank2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))

def _protected_add(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    return  np.add(x1, x2)

def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


add2 = make_function(function=_protected_add, name='add', arity=2)
sub2 = make_function(function=np.subtract, name='sub', arity=2)
mul2 = make_function(function=np.multiply, name='mul', arity=2)
div2 = make_function(function=_protected_division, name='div', arity=2)
rank_add2 = make_function(function=_protected_rank_add, name='rank_add', arity=2)
rank_sub2 = make_function(function=_protected_rank_sub, name='rank_sub', arity=2)
rank_mul2 = make_function(function=_protected_rank_mul, name='rank_mul', arity=2)
rank_div2 = make_function(function=_protected_rank_division, name='rank_div', arity=2)
sqrt1 = make_function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = make_function(function=_protected_log, name='log', arity=1)
neg1 = make_function(function=np.negative, name='neg', arity=1)
inv1 = make_function(function=_protected_inverse, name='inv', arity=1)
abs1 = make_function(function=np.abs, name='abs', arity=1)
max2 = make_function(function=np.maximum, name='max', arity=2)
min2 = make_function(function=np.minimum, name='min', arity=2)
sin1 = make_function(function=np.sin, name='sin', arity=1)
cos1 = make_function(function=np.cos, name='cos', arity=1)
tan1 = make_function(function=np.tan, name='tan', arity=1)
sig1 = make_function(function=_sigmoid, name='sig', arity=1)
ts_corr3 = make_function(function=_ts_correlation, name='ts_corr', arity=3, _type='ts')
ts_cov3 = make_function(function=_ts_covariance, name='ts_cov', arity=3, _type='ts')
ts_sum2 = make_function(function=_ts_sum, name='ts_sum', arity=2, _type='ts')
ts_stddev2 = make_function(function=_ts_stddev, name='ts_stddev', arity=2, _type='ts')
ts_max2 = make_function(function=_ts_max, name='ts_max', arity=2, _type='ts')
ts_min2 = make_function(function=_ts_min, name='ts_min', arity=2, _type='ts')
ts_nanmean2 = make_function(function=_ts_nanmean, name='ts_nanmean', arity=2, _type='ts')
ts_prod2 = make_function(function=_ts_prod, name='ts_prod', arity=2, _type='ts')
ts_rank2 = make_function(function=_ts_rank, name='ts_rank', arity=2, _type='ts')
rank1 = make_function(function=_rank, name='rank', arity=1)

delay2 = make_function(function=_delay, name='delay', arity=2, _type='ts')
delta2 = make_function(function=_delta, name='delta', arity=2, _type='ts')
ts_argmax2 = make_function(function=_ts_argmax, name='ts_argmax', arity=2, _type='ts')
ts_argmin2 = make_function(function=_ts_argmin, name='ts_argmin', arity=2, _type='ts')

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'rank_add': rank_add2,
                 'rank_sub': rank_sub2,
                 'rank_mul': rank_mul2,
                 'rank_div': rank_div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 'ts_corr': ts_corr3,
                 'ts_cov':ts_cov3,
                 'ts_stddev':ts_stddev2,
                 'ts_sum':ts_sum2,
                 'ts_max':ts_max2,
                 'ts_min':ts_min2,
                 'ts_nanmean':ts_nanmean2,
                 'ts_prod':ts_prod2,
                 'ts_rank':ts_rank2,
                 'rank':rank1,
                 'delay':delay2,
                 'delta':delta2,
                 'ts_argmax':ts_argmax2,
                 'ts_argmin':ts_argmin2}
