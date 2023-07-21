import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
from datetime import timedelta

import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats import median_abs_deviation as MAD


def get_train_test(df, start, end=None, datetime_variable = 'ValueDateTimeOffset', form="%Y-%m-%d %H:%M:%S.%f %z"):
    dates = pd.to_datetime(df[datetime_variable], format=form, utc=True).dt.tz_convert('CET').dt.date

    if end:
        test = df[(dates>=datetime.date(*start)) &  (dates<datetime.date(*end))]
        train = df[dates<datetime.date(*start)] 
    
    else:
        test = df[dates>=datetime.date(*start)]
        train = df[dates<datetime.date(*start)]
    
    return train, test 

def apply_func(df, func, cols_to_apply):
    transformed_df = df.copy()
    for name, values in df.iteritems():
        if name in cols_to_apply:
            transformed_df[name] = func(values)
    return transformed_df


def mad_normalize_data(df_train, df_test, col, derived_columns):
    normalized_df_train = df_train.copy()
    normalized_df_test = df_test.copy()

    median_p = np.median(df_train[col])
    mad_price_p = MAD(df_train[col])

    for name in derived_columns:
        median = np.median(df_train[name])
        mad_price = MAD(df_train[name])
        normalized_df_train[name] = (1/mad_price**2)*(normalized_df_train[name] - median)
        normalized_df_test[name] = (1/mad_price**2)*(normalized_df_test[name] - median)

    return median_p, mad_price_p, normalized_df_train, normalized_df_test 

def standardizes_ind_cols(df_train, df_test, individual_cols):
    normalized_df_train = df_train.copy()
    normalized_df_test = df_test.copy()
    
    for name in individual_cols:
        mean = np.mean(df_train[name])
        sd = np.std(df_train[name])
        if np.round(mean)==0 and np.round(sd)==0 and 'solar' in name:
            pass
        else: 
            normalized_df_train[name] = (normalized_df_train[name]-mean)/sd**2
            normalized_df_test[name] = (normalized_df_test[name]-mean)/sd**2
    
    return normalized_df_train, normalized_df_test

def transformed_data(df_train, df_test, Y, X_no_dummies): 
    # Transform Y and Lags
    cols_from_price = [p for p in X_no_dummies if Y[0] in p]
    median, mad_price, normalized_df_train, normalized_df_test  = mad_normalize_data(df_train, df_test, Y, cols_from_price + Y)
    
    # Apply arcsinh
    normalized_df_train = apply_func(normalized_df_train, np.arcsinh, cols_from_price + Y)
    normalized_df_test = apply_func(normalized_df_test , np.arcsinh, cols_from_price + Y)

    # Standardize other variables
    cols_other = [x for x in X_no_dummies if Y[0] not in x]
    normalized_df_train, normalized_df_test = standardizes_ind_cols(normalized_df_train, normalized_df_test , cols_other)

    return median, mad_price, normalized_df_train, normalized_df_test 

def rolling_window(model, df_train, df_test, X, Y, X_no_dummies, train_size=100, test_size=30, get_coef=False):
    test_dates = pd.to_datetime(df_test['DateValueCET']).dt.date
    test_start = min(test_dates)
    test_end = max(test_dates)
    df_to_train = df_train.copy()
    pred_out = np.array([])
    
    iterations = np.ceil(((test_end-test_start).days+1)/test_size)
    print(f'Training {iterations} models', end='; \n')

    k = 1
    coef_count = np.zeros(len(X))
    while test_start <= test_end:
        train_dates = pd.to_datetime(df_to_train['DateValueCET']).dt.date
        train_start = test_start - timedelta(days=train_size)
        test_interval = test_start + timedelta(days=test_size)

        # Create train and test set
        df_to_train = df_to_train[train_dates >= train_start].copy()
        df_to_test = df_test[(test_dates >= test_start) & (test_dates < test_interval)].copy()

        if len(df_to_test) != 0:
            # Transform data and fit model
            median, mad_price, df_to_train_norm, df_to_test_norm = transformed_data(df_to_train, df_to_test, Y, X_no_dummies)
            model.fit(df_to_train_norm[X], df_to_train_norm[Y].values.ravel())
            if get_coef:
                coef = model.coef_
                coef_count[coef!=0] += 1

            print(k, end='\r')
            k += 1

            # Evaluate on test set
            pred_transformed = model.predict(df_to_test_norm[X])

            # Save predictions
            pred = np.sinh(pred_transformed)*mad_price**2 + median
            pred_out = np.append(pred_out, pred)

        test_start = test_interval
        df_to_train = pd.concat([df_to_train, df_to_test])

    if get_coef:
        return pred_out, coef_count
    return pred_out

def rolling_window_grid(grid, df_train, df_test, X, Y, X_no_dummies, train_size=100, test_size=3, hyp_opt_freq = 30, get_counts = False):
    test_dates = pd.to_datetime(df_test['DateValueCET']).dt.date
    test_start = min(test_dates)
    test_end = max(test_dates)
    df_to_train = df_train.copy()
    pred_out = np.array([])
    
    iterations = np.ceil(((test_end-test_start).days + 1)/test_size)
    print(f'Training {iterations} models', end='; \n')
    if get_counts:
        counts = np.zeros((int(iterations), len(X)))
        scores = np.zeros(int(iterations))
        cv_res = []
    k = 1
    while test_start <= test_end:
        train_dates = pd.to_datetime(df_to_train['DateValueCET']).dt.date
        train_start = test_start - timedelta(days=train_size)
        test_interval = test_start + timedelta(days=test_size)

        # Create train and test set
        df_to_train = df_to_train[train_dates >= train_start].copy()
        df_to_test = df_test[(test_dates >= test_start) & (test_dates < test_interval)].copy()

        # Transform data and fit model
        median, mad_price, df_to_train_norm, df_to_test_norm = transformed_data(df_to_train, df_to_test, Y, X_no_dummies)

        if k % hyp_opt_freq == 0 or k == 1:
            grid.fit(df_to_train_norm[X], df_to_train_norm[Y].values.ravel())
            if get_counts:
                cv_res.append(grid.cv_results_)
        else:
            grid.best_estimator_.fit(df_to_train_norm[X], df_to_train_norm[Y].values.ravel())
        
        if get_counts:
            counts[k-1] = grid.best_estimator_.feature_importances_
            score = grid.best_estimator_.score(df_to_train_norm[X], df_to_train_norm[Y])
            scores[k-1] = score
        print(k, end='\r')
        k += 1

        # Evaluate on test set
        pred_transformed = grid.predict(df_to_test_norm[X])

        # Save predictions
        pred = np.sinh(pred_transformed)*mad_price**2 + median
        pred_out = np.append(pred_out, pred)

        test_start = test_interval
        df_to_train = pd.concat([df_to_train, df_to_test])

    if get_counts:
        return pred_out, counts, scores, cv_res
    return pred_out

def rolling_window_hourly(model, df_train, df_test, X, Y, dummies, train_size=100, test_size=30, print_res=True, get_coef=False):
    preds = []

    counts = np.zeros((24, len(X[0])))
    for i in range(24):
        print(f'Hour {i}', end='; ')
        X_no_dummies = [x for x in X[i] if x not in dummies]
        current_df = df_train.loc[df_train['HourCET'] == i]
        current_df_test = df_test.loc[df_test['HourCET'] == i]

        if get_coef:
            pred, coef_count = rolling_window(model, current_df, current_df_test, X[i], Y, X_no_dummies, train_size, test_size, get_coef)
            counts[i] = coef_count

        else:
            pred = rolling_window(model, current_df, current_df_test, X[i], Y, X_no_dummies, train_size, test_size, get_coef)

        current_index = current_df_test.index
        preds.append(pd.Series(pred.reshape(1,-1)[0], current_index))
        
        if print_res:
            print('MAE', MAE(y_true = current_df_test[Y], y_pred = pred))
        else:
            print('')
    
    preds = pd.concat([*preds]).sort_index()
    print('Final results')

    plt.plot(range(len(df_test[Y])), df_test[Y])
    plt.plot(range(len(preds)), preds)
    plt.show()

    if print_res:
        print('MAE', MAE(y_true = df_test[Y], y_pred = preds))
        print('MSE', MSE(y_true = df_test[Y], y_pred = preds))
    
    if get_coef:
        return preds, counts
    
    return preds

def backward_selection(df_train, X, Y, dummies, train_size, cov_type="nonrobust", print_res=True):
    train_dates = pd.to_datetime(df_train['DateValueCET']).dt.date
    train_end = max(train_dates)
    train_start = train_end - timedelta(days=train_size)
    df_train = df_train[train_dates > train_start].copy()
    X_no_dummies = [x for x in X if x not in dummies]
    _, _, df_train, _ = transformed_data(df_train, df_train, Y, X_no_dummies)
    df_dropped = df_train[X].copy()

    ex = sm.add_constant(df_dropped)
    step = 0
    while(True):
        step += 1 
        result = sm.OLS(df_train[Y], ex).fit(cov_type=cov_type)
        ar2 = result.rsquared
        pval = result.pvalues
        throw_away_index = np.argmax(result.pvalues)
        if pval[throw_away_index] >= 0.1:
            ex = ex.drop(pval.index[throw_away_index], axis=1)
            if print_res:
                print(f'step {step}', end= ': ')
                print(f'Adjusted R2: {ar2}', end= ', ')
                print(f'Candidate: {pval.index[throw_away_index]}', end= ', ')
                print(f'p-value: {pval[throw_away_index]}')
                print('===============')
        else:
            print('Final eval', end=': ')
            print(f'Adjusted R2: {ar2}')
            return ex.columns.values