
import signals.utils as su
import pandas as pd
import numpy as np
import get_price.get_first_rate_intraday_data as id
import contract_utilities.contract_meta_info as cmi
from os.path import exists

def prepare_2H_data(**kwargs):

    file_name ='cnn_data.pkl'

    if exists(file_name):
        return pd.read_pickle(file_name)

    rolling_schema_output = su.get_futures_rolling_schema(ticker_head='CL',date_from=20100101, date_to=20171030)
    liquid_contract_frame = rolling_schema_output['liquid_contract_frame']

    ticker_list = liquid_contract_frame['ticker'].unique()
    data_list = []

    for ticker_i in ticker_list:

        contract_specs_output = cmi.get_contract_specs(ticker_i)

        data_out_5M = id.get_presaved_data(ticker=ticker_i, interval='5M')

        settle_dates_i = liquid_contract_frame.loc[liquid_contract_frame['ticker']==ticker_i,'settle_date']
        dates_after_i = liquid_contract_frame.loc[liquid_contract_frame['settle_date']>max(settle_dates_i),'settle_date']
        dates_before_i = liquid_contract_frame.loc[liquid_contract_frame['settle_date']<min(settle_dates_i),'settle_date']

        date_from_i = min(settle_dates_i)
        date_to_i = max(settle_dates_i)

        if len(dates_before_i)>0:
            date_from_i = dates_before_i.iloc[-3]

        if len(dates_after_i)>0:
            date_to_i = dates_after_i.iloc[2]

        data_out_5M = data_out_5M.loc[(data_out_5M.index>=date_from_i)&(data_out_5M.index<=date_to_i),:]
        data_out_5M['ticker'] = ticker_i
        data_out_5M['cont_indx'] = contract_specs_output['cont_indx']

        for i in range(1, 24):
            data_out_5M[['open_' + str(i),'high_' + str(i),'low_' + str(i),'close_' + str(i),'volume_' + str(i)]] = \
            data_out_5M[['open','high','low','close','volume']].shift(i)

        data_2H = data_out_5M.resample('2H').last()
        data_2H.dropna(subset='close', inplace=True)
        data_2H['percent_diff'] = data_2H['close'].diff()/data_2H['close'].shift(1)
        data_2H['std'] = data_2H['percent_diff'].rolling(10).std()
        data_2H['percent_diff1'] = data_2H['percent_diff'].shift(-1)

        if len(dates_before_i)>0:
            date_from_i = dates_before_i.iloc[-1]

        if len(dates_after_i)>0:
            date_to_i = dates_after_i.iloc[0]

        data_2H = data_2H.loc[(data_2H.index >= date_from_i) & (data_2H.index <= date_to_i), :]

        data_list.append(data_2H)

    raw_data = pd.concat(data_list)
    raw_data['settle_date'] = pd.to_datetime(raw_data.index.date)
    raw_data['datetime_'] = pd.to_datetime(raw_data.index)
    raw_data.sort_values(['datetime_', 'cont_indx'], ascending=[True, False], inplace=True)
    raw_data.drop_duplicates(subset='datetime_', keep='first', inplace=True)
    raw_data.dropna(subset=['std', 'percent_diff1'], inplace=True)

    raw_data['label'] = 1
    raw_data.loc[raw_data['percent_diff1']>0.55*raw_data['std'],'label'] = 2
    raw_data.loc[raw_data['percent_diff1']<-0.55*raw_data['std'],'label'] = 0

    raw_data.to_pickle('cnn_data.pkl')

    return raw_data


def prepare_rolling_simulation_indices(**kwargs):

    entire_data = kwargs['entire_data']

    training_size = 60*52*2
    validation_size = 4*60
    test_size = 2*60
    total_size = training_size + validation_size + test_size

    train_start_index = list(range(0, len(entire_data) - total_size, test_size))
    train_end_index = [x + training_size for x in train_start_index]
    validation_start_index = train_end_index
    validation_end_index = [x + validation_size for x in validation_start_index]
    test_start_index = validation_end_index
    test_end_index = [x + test_size for x in test_start_index]
    test_end_index[-1] = -1

    return {'train_start_index_list': train_start_index,'train_end_index_list': train_end_index,
            'validation_start_index_list': validation_start_index, 'validation_end_index_list': validation_end_index,
            'test_start_index_list': test_start_index, 'test_end_index_list': test_end_index}


def get_features(**kwargs):

    df = kwargs['df']

    column_list = []
    primary_column_names = ['open', 'high', 'low', 'close', 'volume']
    for col_i in primary_column_names:
        column_list = column_list + [col_i] + [col_i + '_' + str(x) for x in range(1, 24)]

    return df[column_list]


def reshape_data(data_input):

    return np.reshape(data_input, (data_input.shape[0], 5, 24))






    
        

