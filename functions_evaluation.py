
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy import stats
import numpy as np
import pandas as pd

def root_mean_squared_error(y_true, y_pred):
    return MSE(y_true, y_pred)**(1/2)

def relative_mean_absolute_error(y_true, y_pred, period):
    naive_real = y_true.copy()
    n_prices_day = 24
    if period == 'Period2':
        dates = pd.date_range(start='2021-01-01', periods=y_true.shape[0], freq='1H')  
    elif period == 'Period1':
        dates = pd.date_range(start='2019-07-01', periods=y_true.shape[0], freq='1H')  

    naive_real = pd.DataFrame(naive_real, index=dates)

    index = naive_real.index[n_prices_day * 7:]

    naive_pred = pd.DataFrame(index=index, columns=naive_real.columns)
    naive_pred.loc[:, :] = naive_real.loc[naive_pred.index - pd.Timedelta(days=7)].values

    naive_real = naive_real.loc[naive_pred.index]
    MAE_naive = MAE(naive_real, naive_pred)
    return np.mean(np.abs(y_true - y_pred) / MAE_naive) 

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))

def get_date_range(period, y_true):
    if period == 'Period2':
        dates = pd.date_range(start='2020-01-01 00:00:00', periods=y_true.shape[0], freq='1H', tz='CET')  
    elif period == 'Period1':
        dates = pd.date_range(start='2019-07-01 00:00:00', periods=y_true.shape[0], freq='1H', tz='CET') 
    return dates

def diebold_mariano(y_true, y_pred1, y_pred2, period, norm = 1):
    date_range = get_date_range(period, y_true)
    values_d = {"y_true": y_true, "y_pred1": y_pred1, "y_pred2": y_pred2, "DateTime": date_range, "Hour": date_range.hour}
    df_preds = pd.DataFrame(values_d)
    errors_pred_1 = y_true - y_pred1
    errors_pred_2 = y_true - y_pred2
    DM_stats = []
    for i in range(24):
        df_preds_h = df_preds[df_preds["Hour"] == i]
        errors_pred_1 = df_preds_h["y_true"] - df_preds_h["y_pred1"]
        errors_pred_2 = df_preds_h["y_true"] - df_preds_h["y_pred2"]
        if norm == 1:
            diff = np.abs(errors_pred_1.values) - np.abs(errors_pred_2.values)
        elif norm == 2:
            diff = errors_pred_1.values**2 - errors_pred_2.values**2
        # Computing the test statistic
        N = len(diff)
        mean_d = np.mean(diff)
        var_d = np.var(diff, ddof=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)
        DM_stats.append(DM_stat)
    p_value = 1 - stats.norm.cdf(DM_stats)
    return p_value