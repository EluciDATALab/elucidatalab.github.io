import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays

from datetime import datetime, timedelta
from IPython.display import Markdown, display

import multiprocessing
from joblib import Parallel, delayed

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from elucidata.tools.visualization import vis_plotly as vp
from elucidata.tools.visualization import vis_plotly_tools as vpt
from elucidata.tools.visualization import vis_plotly_widgets as vpw
from elucidata.tools.visualization import vis_plotly_widgets_tools as vpwt

from ipywidgets import interact, interact_manual, interactive, fixed
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

target = 'consumption'
features = ['Pastday', 'Pastweek', 'Hour', 'Weekday', 'Month', 'Holiday', 'Temperature']


def plot_real_estimated_power(df_real_values,array_predicted_values, ax=None):
    """
    plot the real and estimated values of the global active power. The values are resampled on daily basis

    df_real_values: dataframe containing the real values
    array_predicted_values: array containing the predicted values
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    df_copy = df_real_values.copy()
    df_copy['Predicted'] = array_predicted_values
    df_copy = df_copy.resample('D').median().copy()
    df_copy.loc[:,'consumption'].plot(figsize=(16,5), ax=ax)
    df_copy.loc[:,'Predicted'].plot(figsize=(16,5), legend=True, ax=ax)
    ax.legend(['Observed', 'Predicted'])
    ax.set_title('Evolution global active power')
    ax.set_ylabel('Active Power [Wh]');
    return ax


def plot_MAE(scores_baseline,scores_rfr,scores_svr,title):
    """
    plot the MAE of each model.

    scores_baseline: MAE of the dummy model
    scores_rfr: MAE of the random forest model
    scores_svr: MAE of the support vector regression model
    """

    plt.figure(figsize=(14,5))
    plt.plot(scores_baseline)
    plt.plot(scores_rfr)
    plt.plot(scores_svr)
    plt.xlabel('Month')
    plt.ylabel('MAE')
    plt.legend(['Dummy', 'Random Forest Regression', 'Support Vector Regression'], loc='upper right')
    plt.title(title)

def plot_month_with_weekends_and_holidays_highlighted(df, start, end):
    fr_holidays = holidays.France() #Change variable name in fr_holidays
    df['Weekday'] = df.index.dayofweek

    fig, ax = plt.subplots(nrows=1, ncols=1, sharey='row', figsize=(15,5))

    df[start:end]['Global_active_power'].resample('4H').sum().plot(ax=ax,title='Evolution global active power')

    for date in df[start:end].index:
        if df.loc[date]['Weekday'] == 5 or df.loc[date]['Weekday'] == 6:
            plt.axvspan(date, date + timedelta(hours=1), color='orange', alpha=0.5, lw=None)
        if date in fr_holidays:
            plt.axvspan(date, date + timedelta(hours=1), color='red', alpha=0.5, lw=None)
    ax.set_ylabel('consumption energy [Wh]');

def plot_year_with_temperature(df, start, end):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharey='row', figsize=(15,5))

    plot_df = df[start:end].resample('D').agg({'Global_active_power':'sum','temperature':'mean'})

    ax1.set_ylabel('Active Power [Wh]');
    plot_df['Global_active_power'].plot(ax=ax1, color='tab:blue')

    ax2 = ax1.twinx()

    ax2.set_ylabel('temperature')
    plot_df['temperature'].plot(ax=ax2, color='tab:orange')

    ax1.set_title('Evolution global active power & temperature')

    fig.tight_layout()
    plt.show()

def plot_day_with_submeters(df, start, end):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharey='row', figsize=(15,10))

    df[start:end]['Global_active_power'].plot(title='Evolution global active power', ax=ax[0])
    ax[0].set_ylabel('consumption energy [kWmin]');

    df[start:end][['Sub_metering_1','Sub_metering_2','Sub_metering_3']].plot(title='Evolution energy consumption per sub metering', ax=ax[1]);
    ax[1].set_ylabel('consumption energy [Wh]');

def plot_weekly_data(df):
    start = '2009-12-01 00:00:00'
    end = '2009-12-07 23:23:59'


    plot_df = df[start:end]['Global_active_power']
    fig = vp.plot_multiple_time_series([plot_df],
                                       show_time_slider=False, show_time_frame=False)
    fig.update_layout(yaxis_title='Global active power [Wh]',
              xaxis_title='Date',
              title='Evolution global active power',
              font={'size': 18},
              showlegend=True)
    fig.show()


def plot_correlation(df):

    def make_plot(resample_rate):

        if resample_rate is None:
            resampled_df = df.copy()
        else:
            resampled_df = df.resample(resample_rate).agg({'Global_active_power':'sum','temperature':'mean'})
        corr = resampled_df.corr()
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig = px.imshow(corr, color_continuous_scale='RdBu', zmin=-1, zmax=1)
        fig.show()

    controller= vpwt.get_controller({'widget': 'Dropdown',
                                     'options': [('No resampling', None),
                                                 ('1 Hour', 'H'),
                                                 ('8 Hours', '8H'),
                                                 ('1 Day', 'D'),
                                                 ('1 Week', '7D'),
                                                 ('30 days', '30D')],
                                     'value': 'H', 'description': 'Resample rate',
                                     'style': {'description_width': 'initial'}})

    interact(make_plot, resample_rate=controller)


def read_consumption_data():
    parse = lambda x, y: datetime.strptime(x+' '+y, '%d/%m/%Y %H:%M:%S')
    data = pd.read_csv('household_power_consumption.txt', sep=';', na_values=['?'], index_col=0,
                       parse_dates=[['Date', 'Time']], date_parser=parse)
    data = data.loc['2007-01-01':]
    return data

def read_climate_data():
    # climate_file_name = 'temperature_data.hdf5'
    #ext_data = pd.read_hdf(climate_file_name, 'temp')
    climate_file_name = 'temperature_data.csv'
    ext_data = pd.read_csv(climate_file_name)
    ext_data["b'time_UTC'"] = pd.to_datetime(ext_data["b'time_UTC'"])
    ext_data = ext_data.set_index("b'time_UTC'")
    cols = ['temperature','dew_point','humidity','sea_lvl_pressure','visibility','wind_dir_degrees']
    ext_data[cols] = ext_data[cols].replace('N/A',np.nan)
    ext_data = ext_data[cols].apply(pd.to_numeric)
    ext_data = ext_data.loc['2007-01-01':]

    return ext_data

def printmd(string):
    display(Markdown(string))

def get_months(feats):
    start = feats.index[0]
    end = feats.index[-1]

    daterange = pd.date_range(start, end,freq='M')
    months = [(d.year,d.month) for d in pd.date_range(start, end,freq='M')]
    return months

def predictions_window(model, feats, start_month_train_idx, window_size_train, offset_test=1, window_size_test=1):

    months = get_months(feats)

    start_month_train = months[start_month_train_idx]
    end_month_train = months[start_month_train_idx + window_size_train]

    start_month_test = months[start_month_train_idx + window_size_train + offset_test]
    end_month_test = months[start_month_train_idx + window_size_train + offset_test + window_size_test]

    # Add time specifier in the end date, otherwise, the full first day of the end month is included
    # Then, exclude the final element so that the data ends at 23h the day before
    train_data = feats["{}-{}-1".format(*start_month_train):"{}-{}-1 00:00:00".format(*end_month_train)][:-1]
    test_data = feats["{}-{}-1".format(*start_month_test):"{}-{}-1 00:00:00".format(*end_month_test)][:-1]

    model.fit(train_data[features], train_data[target])
    test_predict = model.predict(test_data[features])

    return test_predict


def score_window(model, feats, start_month_train_idx, window_size_train, offset_test=1, window_size_test=1):
    """
    Get the mean absolute error of a model trained a certain time window and evaluated on some month(s)
    after that window.

    model: instance of an sklearn regressor
    start_month_train_idx: index of the starting month of the training window
    window_size_train: number of months in the training window
    offset_test: how many months between the last month in the training window and the first month
        of the test window. Default: 1.
    window_size_test: number of months in the testing window
    """
    months = get_months(feats)

    start_month_train = months[start_month_train_idx]
    end_month_train = months[start_month_train_idx + window_size_train]

    start_month_test = months[start_month_train_idx + window_size_train + offset_test]
    end_month_test = months[start_month_train_idx + window_size_train + offset_test + window_size_test]

    # Add time specifier in the end date, otherwise, the full first day of the end month is included
    # Then, exclude the final element so that the data ends at 23h the day before
    train_data = feats["{}-{}-1".format(*start_month_train):"{}-{}-1 00:00:00".format(*end_month_train)][:-1]
    test_data = feats["{}-{}-1".format(*start_month_test):"{}-{}-1 00:00:00".format(*end_month_test)][:-1]

    model.fit(train_data[features], train_data[target])
    test_predict = model.predict(test_data[features])
    return mean_absolute_error(test_data[target], test_predict)

def get_true_values(feats, start_month_train_idx, window_size_train, offset_test=1, window_size_test=1):

    #months = get_months(feats)

    #start_month_test = months[start_month_train_idx + window_size_train + offset_test + 1]
    #end_month_test = months[start_month_train_idx + window_size_train + offset_test + 1 + window_size_test]

    #test_data = feats["{}-{}-1".format(*start_month_test):"{}-{}-1 00:00:00".format(*end_month_test)][:-1]
    test_idxes = get_datetimes(feats, start_month_train_idx, window_size_train,
                               offset_test=offset_test, window_size_test=window_size_test)
    test_data = feats.loc[test_idxes]
    return test_data[target]


def get_datetimes(feats, start_month_train_idx, window_size_train, offset_test=1, window_size_test=1):

    months = get_months(feats)

    start_month_train = months[start_month_train_idx]
    end_month_train = months[start_month_train_idx + window_size_train]

    start_month_test = months[start_month_train_idx + window_size_train + offset_test]
    end_month_test = months[start_month_train_idx + window_size_train + offset_test + window_size_test]

    # Add time specifier in the end date, otherwise, the full first day of the end month is included
    # Then, exclude the final element so that the data ends at 23h the day before

    test_data = feats["{}-{}-1".format(*start_month_test):"{}-{}-1 00:00:00".format(*end_month_test)][:-1]

    return test_data.index

def parallel_evaluation(model, feats, window_size_train, offset_test=1, window_size_test=1):
    num_cores = multiprocessing.cpu_count()
    months = get_months(feats)
    if window_size_train == 'fullperiod':
        max_window_size = len(months) - offset_test - 1 - window_size_test
        scores = Parallel(n_jobs=num_cores)(delayed(score_window)(model, feats, 0, w, offset_test, window_size_test)
                                                for w in range(1, max_window_size))
    else:
        max_idx_month = len(months) - offset_test - 1 - window_size_test - window_size_train
        scores = Parallel(n_jobs=num_cores)(delayed(score_window)(model, feats, m, window_size_train, offset_test, window_size_test)
                                                for m in range(0, max_idx_month))

    return scores


def plot_effects_resampling(data, show_time_frame=False):
    def rolling(x, c, window):
        # Use a rolling window
        x = x[c]
        return x.rolling(f'{window}T').mean()
    methods = {'Rolling': rolling}

    sampled_data = data.loc['2008-04-01 00:00:00': '2008-04-30 23:59:59'][['Global_active_power']]#.sample(100000).sort_index()

    transform_plot, method_picker,  controllers = \
        vpw.plot_transform_time_series(sampled_data, methods,
                                       controllers=[{'widget': 'Dropdown',
                                                     'options': [('30 minutes', 30), ('1 Hour', 60),
                                                                 ('4 Hours', 240)],
                                                     'value': 30,
                                                     'description': 'Window size for resampling',
                                                     'style': {'description_width': 'initial'}}],
                                        kwargs_ts={'show_time_frame': show_time_frame, 'show_time_slider': False})
    interact(transform_plot, method=fixed('Rolling'), column=fixed('Global_active_power'), win=controllers[1]);

def plot_temperature_power_one_year_old(new_data):

    plot_df = new_data[['Global_active_power', 'temperature']].resample('D').agg({'Global_active_power':'sum',
                                                                                  'temperature':'mean'})
    plot_df = plot_df.dropna()
    scaler = MinMaxScaler()
    plot_df[['Global_active_power', 'temperature']] = scaler.fit_transform(plot_df[['Global_active_power', 'temperature']])
    fig = vp.plot_multiple_time_series([plot_df['Global_active_power'], plot_df['temperature']],
                                       show_time_slider=False, show_time_frame=False)#, same_plot=False)
    fig.show()

def plot_temperature_power_one_year(new_data, show_time_frame=False):

    def normalize_data(x, c, normalize):
        print(c, normalize)
        x = x.copy()
        if normalize:
            scaler = MinMaxScaler()
            x[c] = scaler.fit_transform(np.array(x[c]).reshape(-1, 1))

        return x

    plot_df = new_data[['Global_active_power', 'temperature']].resample('D').agg({'Global_active_power':'mean',
                                                                                  'temperature':'mean'})
    plot_df = plot_df.dropna()

    both_columns = ['Global_active_power', 'temperature']

    controller_norm = vpwt.get_controller({'widget': 'Checkbox', 'value': False, 'description': 'Normalize data?'})
    controller_columns = vpwt.get_controller({'widget': 'RadioButtons',
                                              'options': [('Plot active power & temperature', both_columns),
                                                          ('Only plot active power', ['Global_active_power']),
                                                          ('Only plot temperature', ['temperature'])],
                                             'value': both_columns, 'description': ' '})

    def make_plot(normalize):
        df = plot_df[both_columns].copy()
        if normalize:
            scaler = MinMaxScaler()
            df[both_columns] = scaler.fit_transform(df)
        series_to_plot = [df[col] for col in both_columns]
        fig = vp.plot_multiple_time_series(series_to_plot,
                                           show_time_slider=False, show_time_frame=False)#, same_plot=False)
        fig.show()
    # sampled_data = data.loc['2008-04-01 00:00:00': '2008-04-30 23:59:59'][['Global_active_power']]#.sample(100000).sort_index()

    interact(make_plot, normalize=controller_norm);



def plot_consumption_with_holidays_weekends_old2(new_data):
    resample_rate_hour = 4
    df = new_data[['Global_active_power']].resample(f'{resample_rate_hour}H').agg({'Global_active_power':'sum'})
    years = np.arange(2007, 2010)

    def make_plot(**controllers):
        # print(controllers)
        years = [int(descr[-4:]) for descr, val in controllers.items() if val]
        series_to_plot = []
        shapes = []

        colors = vpt.get_colors('Dark2', [0.5 for _ in years])

        for _i, year in enumerate(years):

            start_lastweek = f'{year}-12-24'
            end_lastweek = f'{year}-12-31'
            df_lastweek= df.loc[start_lastweek:end_lastweek]
            last_sunday = None
            _j = 0
            #
            while last_sunday is None:
                date = df_lastweek.index[_j]
                if date.dayofweek == 6:
                    last_sunday = date

                _j += 1
            start_day = last_sunday - pd.Timedelta('27D')
            christmas_idx = (24 - (last_sunday.day - 28))
            # print(last_sunday, last_sunday.day, christmas_idx)

            plot_df = df[start_day:last_sunday + pd.Timedelta('23H') + pd.Timedelta('59m')].reset_index(drop=True)
            plot_df['index'] = np.arange(0, 28, resample_rate_hour / 24)
            plot_df = plot_df.set_index('index')


            d = dict(type="rect",
                xref="x",
                # y-reference is assigned to the plot paper [0,1]
                yref="paper",
                x0=christmas_idx,
                y0=0,
                x1=christmas_idx+1,# + timedelta(hours=4),
                y1=1,
                fillcolor=colors[_i],
                opacity=0.25,
                layer="below",
                line_width=1)
            shapes.append(d)
            # print(plot_df)
            series_to_plot.append(plot_df['Global_active_power'])

        series, _, trace_specs, trace_cols =  \
            vpt.define_figure_specs(series_to_plot, None, {},
                                ['alpha', 'width', 'dash', 'color'])
        
        trace_specs['scatter_kwargs'] = {}
        plot_assistant = {'series': series, 'series_names': years,
                  'trace_cols': trace_cols, 'trace_specs': trace_specs,
                  'mode': ['lines'], 'scatter_kwargs':{}}
        # print(trace_cols, trace_specs)

        n_series = len(series)
        fig = go.Figure([vp._make_timeseries(k, plot_assistant) for k in np.arange(n_series)])

        fig.update_layout(yaxis_title='Global active power [Wh]',
                  xaxis_title='Day number',
                  title='Last 4 weeks of the year (with days of the week aligned)',
                  font={'size': 18},# 'color': "#7f7f7f"},
                  showlegend=True)
        #fig = vp.plot_multiple_time_series(series_to_plot, show_time_slider=False,
        #                                   show_time_frame=False)

        # shapes = []
        # fr_holidays = holidays.France()
        for idx in [5, 12, 19, 26]:#np.arange(0, 28, 1/6):
            #day_of_week = int(np.floor(idx)) % 7
            #if day_of_week in (5, 6):
                #ax.axvspan(date, date + timedelta(hours=1), color='orange', alpha=0.5, lw=None)
                d = dict(type="rect",
                    xref="x",
                    # y-reference is assigned to the plot paper [0,1]
                    yref="paper",
                    x0=idx,
                    y0=0,
                    x1=idx+2,# + timedelta(hours=4),
                    y1=1,
                    fillcolor="yellow",
                    opacity=0.5,
                    layer="below",
                    line_width=1)
                shapes.append(d)

        fig.update_layout(
            shapes=shapes
        )

        fig['layout']['xaxis'].update(range=[0, 28])
        fig['layout']['yaxis'].update(range=[0, 20000])

        fig.show()


    checked = {2007: True, 2008: False, 2009: False}
    controllers = {f'plot_{year}': vpwt.get_controller({'widget': 'Checkbox', 'value': checked[year],
                                                        'description': f'{year}'}) for year in years}


    interact(make_plot, **controllers)

def plot_consumption_with_holidays_weekends(new_data):
    resample_rate_hour = 4
    df = new_data[['Global_active_power']].resample(f'{resample_rate_hour}H').agg({'Global_active_power':'sum'})
    years = np.arange(2007, 2010)

    def make_plot(year):
        # print(controllers)
        series_to_plot = []
        shapes = []

        colors = vpt.get_colors('Dark2', [0.5 for _ in years])


        start_dec = f'{year}-12-01 00:00:00'
        end_dec = f'{year}-12-31 23:59:59'


        plot_df = df.loc[start_dec:end_dec] # .reset_index(drop=True)
        # plot_df['index'] = np.arange(0, 28, resample_rate_hour / 24)
        # plot_df = plot_df.set_index('index')

        fr_holidays = holidays.France()


        for day in plot_df.index:
            if day.dayofweek == 5 and day.hour == 0:
                d = dict(type="rect",
                    xref="x",
                    # y-reference is assigned to the plot paper [0,1]
                    yref="paper",
                    x0=day,
                    y0=0,
                    x1=day+ timedelta(hours=48),
                    y1=1,
                    fillcolor='yellow',
                    opacity=0.25,
                    layer="below",
                    line_width=1)
                shapes.append(d)
            if day.date() in fr_holidays and day.hour == 0:
                d = dict(type="rect",
                    xref="x",
                    # y-reference is assigned to the plot paper [0,1]
                    yref="paper",
                    x0=day,
                    y0=0,
                    x1=day+ timedelta(hours=24),
                    y1=1,
                    fillcolor='orange',
                    opacity=0.5,
                    layer="below",
                    line_width=1)
                shapes.append(d)
        # print(plot_df)
        series_to_plot.append(plot_df['Global_active_power'])

        series, _, trace_specs, trace_cols =  \
            vpt.define_figure_specs(series_to_plot, None, {},
                                ['alpha', 'width', 'dash', 'color'])
        
        trace_specs['scatter_kwargs'] = {}
        
        
        plot_assistant = {'series': series, 'series_names': [year],
                  'trace_cols': trace_cols, 
                  'mode': ['lines'], 'scatter_kwargs':[{}], 'trace_specs': trace_specs}
        # print(trace_cols, trace_specs)

        n_series = len(series)
        fig = go.Figure([vp._make_timeseries(k, plot_assistant) for k in np.arange(n_series)])

        fig.update_layout(yaxis_title='Global active power [Wh]',
                  xaxis_title='Day number',
                  title=f'Power consumption of December {year}',
                  font={'size': 18},# 'color': "#7f7f7f"},
                  showlegend=True)
        #fig = vp.plot_multiple_time_series(series_to_plot, show_time_slider=False,
        #                                   show_time_frame=False)

        # shapes = []
        '''
        for idx in [5, 12, 19, 26]:#np.arange(0, 28, 1/6):
            #day_of_week = int(np.floor(idx)) % 7
            #if day_of_week in (5, 6):
                #ax.axvspan(date, date + timedelta(hours=1), color='orange', alpha=0.5, lw=None)
                d = dict(type="rect",
                    xref="x",
                    # y-reference is assigned to the plot paper [0,1]
                    yref="paper",
                    x0=idx,
                    y0=0,
                    x1=idx+2,# + timedelta(hours=4),
                    y1=1,
                    fillcolor="orange",
                    opacity=0.5,
                    layer="below",
                    line_width=1)
                shapes.append(d)
        '''

        fig.update_layout(
            shapes=shapes
        )

        #fig['layout']['xaxis'].update(range=[0, 28])
        fig['layout']['yaxis'].update(range=[0, 20000])

        fig.show()


    checked = {2007: True, 2008: False, 2009: False}
    controller_year = vpwt.get_controller({'widget': 'RadioButtons', 'options': [2007, 2008, 2009],
                                            'value': 2007, 'description': 'Year'})


    interact(make_plot, year=controller_year)


def plot_yearly_data(new_data):
    resampling_rate_days = 1

    years = np.arange(2007, 2010)

    def make_plot(resampling_rate):#, **controllers):
        resampling_rate_days = int(resampling_rate[:-1])
        df = new_data[['Global_active_power']].copy()
        # years = [int(descr[-4:]) for descr, val in controllers.items() if val]
        series_to_plot = []

        def is_leap_year(year):
            return (year % 4 == 0 and not year % 100 == 0) or year % 1000 == 0

        colors = vpt.get_colors('Dark2', [0.5 for _ in years])

        for _i, year in enumerate(years):

            start_year= f'{year}-01-01 00:00:00'
            end_year = f'{year}-12-31 23:59:59'
            plot_df = df.loc[start_year:end_year].resample(resampling_rate).agg({'Global_active_power':'sum'})

            number_of_days = 365
            if is_leap_year(year):
                number_of_days += 1

            days = np.arange(0, number_of_days, resampling_rate_days)
            month = days / 30 + 1
            plot_df['month'] = month
            plot_df = plot_df.set_index('month')
            # print(plot_df)
            series_to_plot.append(plot_df['Global_active_power'])

        series, _, trace_specs, trace_cols =  \
            vpt.define_figure_specs(series_to_plot, None, {},
                                ['alpha', 'width', 'dash', 'color'])
        trace_specs['scatter_kwargs'] = {}

        _years = [str(year) for year in years]
        plot_assistant = {'series': series, 'series_names': _years,
                  'trace_cols': trace_cols, 'trace_specs': trace_specs,
                  'mode': ['lines']*len(years)}

        n_series = len(series)
        fig = go.Figure([vp._make_timeseries(k, plot_assistant) for k in np.arange(n_series)])

        if resampling_rate_days > 1:
            suffix = 's'
        else:
            suffix = ''
        fig.update_layout(yaxis_title=f'Power consumed in {resampling_rate_days} day{suffix}  [Wh]',
                  xaxis_title='Month',
                  title=' ',
                  font={'size': 18},# 'color': "#7f7f7f"},
                  showlegend=True)

        fig.show()


    #checked = {2007: True, 2008: False, 2009: False}
    #controllers_year_selection = {f'plot_{year}': vpwt.get_controller({'widget': 'Checkbox', 'value': checked[year],
    #                                                    'description': f'{year}',
    #                                                    'style': {'description_width': 'auto'}}) for year in years}

    controller_resampling = vpwt.get_controller({'widget': 'Dropdown',
                                                 'options': [('1 day', '1D'), ('3 days', '3D'), ('1 week', '7D')],
                                                 'value': '1D',
                                                 'description': 'Resampling rate',
                                                 'style': {'description_width': 'initial'}})
    interact(make_plot, resampling_rate=controller_resampling)#, **controllers_year_selection)


def plot_auto_correlation(data):
    power = data['Global_active_power']

    steps = 30
    hours_a_day = 24
    corr = np.zeros(steps - 1)
    x = np.arange(1, steps)
    for i in x:
        corr[i-1] = power.autocorr(i*hours_a_day)

    fig = px.line(x=x, y=corr)
    fig.update_layout(yaxis_title='Correlation',
              xaxis_title='Days',
              xaxis = {'tickmode': 'linear', 'tick0': 1, 'dtick': 1},
              # title='Evolution global active power',
              font={'size': 18},# 'color': "#7f7f7f"},
              showlegend=True)
    fig['layout']['yaxis'].update(range=[0.25, 0.5])
    fig.show()
