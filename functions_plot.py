import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime as dt
import pytz

from os import listdir
from os.path import isfile, join

from texttable import Texttable
import latextable

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

from functions_evaluation import symmetric_mean_absolute_percentage_error as sMAPE
from functions_evaluation import relative_mean_absolute_error as rMAE
from functions_evaluation import mean_absolute_percentage_error as MAPE
from functions_evaluation import diebold_mariano
from functions_evaluation import get_date_range

def create_name(file_name, models, nolatex=True):
    years = {
    'Year1': str(365),
    'Year2': str(365*2),
    'Year3': str(365*3),
    'Year4': str(365*4),
    'Week12': str(7*12),
    }  
    file_name, _ = file_name.split('.')
    model, _, training = file_name.split('_')
    if model in models:
        if nolatex:
            return models[model]
            #return models[model] + years[training]
        return r'$\text{' + models[model] + '}' + '_{' + years[training] + '}$'

def create_table(period, models, nbest):
    header = [r'\textbf{Model}', r'\textbf{MAE}', r'\textbf{rMAE}', r'\textbf{MSE}', r'\textbf{MAPE}', r'\textbf{sMAPE}']
    rows = [header]
    path = 'results'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    names = [create_name(file, models) for file in files if create_name(file, models) != None and period in file]
    order = sorted(range(len(names)), key=lambda x: names[x].lower())
    

    for file in files:
        if period in file:
            with open (join(path, file), 'rb') as fp:
                y_pred, y_true  = pickle.load(fp)
            name = create_name(file, models, nolatex=False)
            if name != None:
                new_row = [name, MAE(y_pred=y_pred, y_true=y_true), 
                        rMAE(y_pred=y_pred, y_true=y_true, period=period), 
                        MSE(y_pred=y_pred, y_true=y_true), 
                        MAPE(y_pred=y_pred, y_true=y_true), 
                        sMAPE(y_pred=y_pred, y_true=y_true)]
                rows.append(new_row)

    ncol = len(header)
    nrow = len(rows)

    values = np.asarray(rows)[1:,1:].astype(float)
    best_models = np.argsort(values, axis=0)
    colorings = np.linspace(start=5, stop=20, num=nbest)[::-1]

    for j in range(1,ncol):
        for i in range(1,nrow):
            if header[j] == r'\textbf{MAPE}':
                to_string = '$' + f'{rows[i][j]:.3}' + '$'
                before_e, after_e = to_string.split('e')
                to_string = before_e + '\mathrm{e}' + after_e
                rows[i][j] = to_string
                
            else:
                rows[i][j] = '$' + str(np.round(rows[i][j], 3)) + '$'

    
    for j in range(1,ncol):
        for nb in range(nbest):
            i = best_models[nb][(j-1)]
            c = str(int(colorings[nb]))
            rows[i+1][j] = r'\cellcolor{grey!' + c + '}' + rows[i+1][j]

    rows = [rows[0]] + [rows[i+1] for i in order]

    table = Texttable()
    table.set_cols_align(['l']+ ["c"] * (len(header)-1))
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    if period == 'Period1':
        cap = 'Results for 2019-01-07 to 2021-07-01'
        lab = 'table:res_period1'
    elif period == 'Period2':
        cap = 'Results for 2021-01-01 to 2023-01-01'
        lab = 'table:res_period2'
        
    print(latextable.draw_latex(table, caption=cap, label=lab))

def create_prediction_plot(period, models, day, models_to_print):
    path = 'results'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    st = dt.datetime(*day, 0, 0).astimezone(pytz.timezone('CET'))
    en = dt.datetime(*day, 23, 0).astimezone(pytz.timezone('CET'))    
    for file in files:
        if period in file:
            with open (join(path, file), 'rb') as fp:
                y_pred, y_true  = pickle.load(fp)
            
            dates = get_date_range(period, y_true) 
            
            df_dates = pd.DataFrame(y_pred, index=dates)
            df_dates['Hour'] = df_dates.index.hour
            agg = np.mean([[row for row in df_dates[0][df_dates['Hour'] == h]] for h in range(24)],axis=1)

            name = create_name(file, models)
            if name!=None and name in models_to_print:
                #to_plot = df_dates[st:en]
                plt.plot(range(24), agg, label=name)

    df_dates = pd.DataFrame(y_true, index=dates)
    df_dates['Hour'] = df_dates.index.hour
    agg = np.mean([[row for row in df_dates[0][df_dates['Hour'] == h]] for h in range(24)],axis=1)
    
    plt.plot(range(24), agg, label='True values')      
    plt.legend(loc='upper left')
    plt.xlabel('Hour')
    plt.ylabel('Price(EUR)')
    plt.title(f'Results for {st.date()}', fontweight="bold")

def create_diebold_mariano_plot(period, models, norm):
    rows = []
    path = 'results'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    names = [create_name(file, models) for file in files if create_name(file, models) != None and period in file]
    order = sorted(range(len(names)), key=lambda x: names[x].lower())
    for file in files:
        if period in file:
            with open (join(path, file), 'rb') as fp:
                y_pred, y_true  = pickle.load(fp)
            name = create_name(file, models)
            if name != None:
                new_row = [name, y_pred]
                rows.append(new_row)
    n_models = len(rows)
    rows = [rows[i] for i in order]
    out = np.zeros([n_models, n_models])
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                test = diebold_mariano(y_true, rows[i][1], rows[j][1], period=period, norm = norm)
                out[i,j] = np.sum(test<=0.05)
    plt.matshow(out, cmap='summer')

    for (i, j), z in np.ndenumerate(out):
        plt.text(j, i, str(int(z)), ha='center', va='center')

    names = [row[0] for row in rows]

    if period == 'Period1':
        cap = 'Numbers of significant Diebold Mariano tests\nfor 2019-01-07 to 2021-07-01'
    elif period == 'Period2':
        cap = 'Numbers of significant Diebold Mariano tests\nfor 2021-01-01 to 2023-01-01'

    x_ticks = plt.xticks(range(len(rows)), names, fontsize=10, rotation=90)
    y_ticks = plt.yticks(range(len(rows)), names, fontsize=10)
    plt.title(cap, fontweight="bold")

def create_diebold_mariano_hourly_plot(period, models, norm, models_to_print):
    rows = []
    path = 'results'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        if period in file:
            with open (join(path, file), 'rb') as fp:
                y_pred, y_true  = pickle.load(fp)
            name = create_name(file, models)
            if name != None and name in models_to_print:
                new_row = [name, y_pred]
                rows.append(new_row)
    
    n_models = len(rows)
    out = []
    names = []
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                test = diebold_mariano(y_true, rows[i][1], rows[j][1], period = period, norm = norm)
                out.append(test)
                names.append(f'{rows[j][0]} agianst {rows[i][0]}')
    
    out_map = [[1 if value>0.05 else 0 for value in row] for row in out]

    plt.matshow(out, cmap='summer_r')

    for (i, j), z in np.ndenumerate(out):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

    if period == 'Period1':
        cap = 'Result from Diebold Mariano tests for 2019-01-07 to 2021-07-01'
    elif period == 'Period2':
        cap = 'Result from Diebold Mariano tests for 2021-01-01 to 2023-01-01'

    x_ticks = plt.xticks(range(24), fontsize=10)
    y_ticks = plt.yticks(range(len(names)), names, fontsize=10)
    plt.title(cap, fontweight="bold")


def create_coef_plot(dfs, grouped_variables, ticks, files):
    variables=[item for sublist in grouped_variables for item in sublist]
    for df, filename in zip(dfs, files):
        n = max(df.max())
        m = len(ticks)
        default_figsize = plt.rcParams["figure.figsize"]
        plt.rcParams['figure.figsize'] = (20,4)

        fig, axs = plt.subplots(1,m)
        for i, ax in enumerate(axs):
            variables = grouped_variables[i]
            im = ax.matshow(df[variables], cmap='summer', vmin=0, vmax=n, aspect="auto")
            ax.set_title(ticks[i])
            ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
            
        
        for ax in fig.get_axes():
            ax.label_outer()

        fig.subplots_adjust(wspace=0.015, hspace=0)
        fig.colorbar(im, ax=axs.ravel().tolist())
        fig.title()
        plt.savefig('plots' + '\\' + filename + '.png')
        plt.rcParams["figure.figsize"] = default_figsize


def create_cumulative_error_plot(period, models, models_to_print, dt_start, dt_end, norm = 2):
    path = 'results'
    files = [f for f in listdir(path) if isfile(join(path, f))]
 
    for file in files:
        if period in file:
            with open (join(path, file), 'rb') as fp:
                y_pred, y_true  = pickle.load(fp)
            
            dates = get_date_range(period, y_true)
            
            df_dates = pd.DataFrame({'y_pred':y_pred, 'y_true':y_true}, index=dates)
            name = create_name(file, models)
            if name!=None and name in models_to_print:
                to_plot = np.cumsum(abs(df_dates['y_true']-df_dates['y_pred'])**norm)/np.arange(len(df_dates))
                plt.plot(to_plot.loc[dt_start: dt_end], label=name)

    if period == 'Period1':
        cap = 'Cumulative absolute error for 2019-01-07 to 2021-07-01'
    elif period == 'Period2':
        cap = 'Cumulative absolute error for 2021-01-01 to 2023-01-01'

    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Cumulative absolute error')
    plt.title(cap, fontweight="bold")

def create_relative_cumulative_loss_plot(period, models, models_to_print, relative_benchmark  = "b1AR", norm = 2):
    path = 'results'
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for file in files:
        if period in file:
            name = create_name(file, models)
            if name != None and name in relative_benchmark:
                with open (join(path, file), 'rb') as fp:
                    y_pred_bench, y_true = pickle.load(fp)

    for file in files:
        if period in file:
            with open (join(path, file), 'rb') as fp:
                y_pred, y_true  = pickle.load(fp)

            dates = get_date_range(period, y_true)

            df_dates = pd.DataFrame({'y_pred':y_pred, 'y_pred_bench':y_pred_bench, 'y_true':y_true}, index=dates)
            name = create_name(file, models)
            if name!=None and name in models_to_print:
                if norm == 2:
                    cLoss_pred = np.cumsum((df_dates['y_true']-df_dates['y_pred'])**2)/np.arange(1, len(df_dates['y_true'])+1)
                    cLoss_bench = np.cumsum((df_dates['y_true']-df_dates['y_pred_bench'])**2)/np.arange(1, len(df_dates['y_true'])+1)
                elif norm == 1:
                    cLoss_pred = np.cumsum(np.abs(df_dates['y_true']-df_dates['y_pred']))/np.arange(1, len(df_dates['y_true'])+1)
                    cLoss_bench = np.cumsum(np.abs(df_dates['y_true']-df_dates['y_pred_bench']))/np.arange(1, len(df_dates['y_true'])+1)
                to_plot = 1 - cLoss_pred/cLoss_bench
                plt.plot(to_plot.index.values, to_plot.values, label=name)
    if period == 'Period1':
        cap = 'Cumulative loss relative to ' + relative_benchmark + ' for 2019-01-07 to 2021-07-01'
    elif period == 'Period2':
        cap = 'Cumulative loss relative to ' + relative_benchmark + ' for 2021-01-01 to 2023-01-01'

    plt.ylim((-1, 1))
    plt.legend(loc='center left')
    plt.xlabel('Time')
    plt.ylabel('Relative Cumulative loss')
    plt.title(cap, fontweight="bold")