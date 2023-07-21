from datetime import timedelta
from datetime import datetime as dt
import random
import pandas as pd

def TestLags(df, size_of_test=1000):
    def perform_test(df, prev_date, lag_val):
        prev_val = df[df['ValueDateTimeOffset'] == prev_date]['PriceMWh'].values
        if prev_val.size > 0:
            if prev_val[0] != lag_val[0]:
                return False

    if size_of_test == 'All': # Warning: takes very long!
        dates = df['ValueDateTimeOffset']
    else:
        index = random.sample(range(0, len(df)), size_of_test)
        dates = pd.to_datetime(df['ValueDateTimeOffset'], format='%Y-%m-%d %H:%M:%S%z', utc=True).iloc[index]
        
    tests = [True, True, True]
    
    for date in dates:
        cur_row = df[df['ValueDateTimeOffset'] == date]
        lag1 = cur_row['PriceMWh_lag1'].values
        lag2 = cur_row['PriceMWh_lag2'].values
        lag7 = cur_row['PriceMWh_lag7'].values

        # test for Lag1
        prev_date1 = date - timedelta(1)
        if perform_test(df, prev_date1, lag1) == False:
            tests[0] = False

        # test for Lag2
        prev_date2 = date - timedelta(2)
        if perform_test(df, prev_date2, lag2) == False:
            tests[1] = False
                
        # test for Lag7
        prev_date7 = date - timedelta(7)
        if perform_test(df, prev_date7, lag7) == False:
            tests[2] = False
    
    if all(tests):
        print('TestLag Success!')
    else:
        print('TestLag Failed!')
        print(f'Lag1 {"Success" if tests[0] else "Failed"}')
        print(f'Lag2 {"Success" if tests[1] else "Failed"}')
        print(f'Lag7 {"Success" if tests[2] else "Failed"}')

#Previous day lag
def TestPrevDayLag(df, prefix, test_col, delta, size_of_test=50):
    if size_of_test == 'All': # Warning: takes very long!
        dates = df['ValueDateTimeOffset']
    else:
        index = random.sample(range(0, len(df)), size_of_test)
        dates = df['ValueDateTimeOffset'].iloc[index]

    test = True
    
    for date in dates:
        cur_row = df[df['ValueDateTimeOffset'] == str(date)]
        prev_date = (dt.strptime(date, "%Y-%m-%d %H:%M:%S.%f %z") - timedelta(days=delta)).date()

        for hour in range(0,24):
            prev_val = df[(df['DateValueCET']==str(prev_date)) & (df['HourCET']==hour)][test_col].values
            hour_val = cur_row[f'{prefix}_h{hour}'].values
      
            if len(prev_val) != 0:
                if prev_val[0] != hour_val[0]:
                    print(date, hour)
                    test = False
            else:
                print('One not checked', date)
                print(prev_val, hour)
            
    if test:
        print('TestPrevDayLag Success!')
    else:
        print('TestPrevDayLag Failed!') 

