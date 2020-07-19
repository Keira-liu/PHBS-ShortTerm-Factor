# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:44:40 2020

@author: bolan
"""

#import scipy.io as scio
import h5py
import pandas as pd
import numpy as np
import os

path = r'C:\Users\52463\Documents\GitHub\PHBS-ShortTerm-Factor\data'
#data = scio.loadmat(path)
marketInfo_securities_china = h5py.File(os.path.join(path,'marketInfo_securities_china.mat'))
data=marketInfo_securities_china['aggregatedDataStruct']

start_date = 20190101
curr_date = 20191231

stock_ticker = np.transpose(data['stock']['description']['tickers']['officialTicker'])
#把股票代码格式从HDF5 object转换为str
stock_code = []
for i in range(stock_ticker.shape[0]):
    stock_code.append(''.join([chr(v[0]) for v in data[(stock_ticker[i][0])]]))

all_tradedates = np.transpose(data['sharedInformation']['allDateStr'])
#把日期格式从ascii码转换成整数类型
all_tradedates_int = []
for i in range(all_tradedates.shape[0]):
    onedate = chr(all_tradedates[i][0])
    for j in range(1,all_tradedates.shape[1]):
        onedate = onedate + chr(all_tradedates[i][j])
    all_tradedates_int.append(int(onedate))
trade_dt = [x for x in all_tradedates_int if (x >= start_date) & (x <= curr_date)]

#Generate Pool
stock_return=pd.DataFrame(np.transpose(data['stock']['intradayReturn']))
#剔除当天涨跌停股票
limited = abs(stock_return) < 0.095 
#剔除之后两天如果有涨跌停的股票
limited = limited & limited.shift(-1) & limited.shift(-2)
#剔除ST股票
ST_table = np.transpose(data['stock']['stTable'])
ST_table = (ST_table == 0)
#剔除不交易的股票
tradeday_table = np.transpose(data['stock']['tradeDayTable'])
all_pool = limited & ST_table & tradeday_table
all_pool = all_pool.astype(float).replace(0,np.nan)

#读取数据，被剔除的数据记为nan
price_open = np.transpose(data['stock']['properties']['open'])*all_pool.values
price_open = pd.DataFrame(price_open,index = all_tradedates_int,columns = stock_code).loc[trade_dt].values
price_high = np.transpose(data['stock']['properties']['high'])*all_pool.values
price_high = pd.DataFrame(price_high,index = all_tradedates_int,columns = stock_code).loc[trade_dt].values
price_low = np.transpose(data['stock']['properties']['low'])*all_pool.values
price_low = pd.DataFrame(price_low,index = all_tradedates_int,columns = stock_code).loc[trade_dt].values
price_close = np.transpose(data['stock']['properties']['close'])*all_pool.values
price_close = pd.DataFrame(price_close,index = all_tradedates_int,columns = stock_code).loc[trade_dt].values
volume = np.transpose(data['stock']['properties']['volume'])*all_pool.values
volume = pd.DataFrame(volume,index = all_tradedates_int,columns = stock_code).loc[trade_dt].values
amount = np.transpose(data['stock']['properties']['amount'])*all_pool.values
amount = pd.DataFrame(amount,index = all_tradedates_int,columns = stock_code).loc[trade_dt].values
vwap = amount/volume
stock_return = stock_return.values*all_pool.values
stock_return = pd.DataFrame(stock_return,index = all_tradedates_int,columns = stock_code).loc[trade_dt].values

#储存数据
all_pool.to_pickle(os.path.join(path,"all_pool.pkl"))
pd.DataFrame(price_open).to_pickle(os.path.join(path,"price_open.pkl"))
pd.DataFrame(price_high).to_pickle(os.path.join(path,"price_high.pkl"))
pd.DataFrame(price_low).to_pickle(os.path.join(path,"price_low.pkl"))
pd.DataFrame(price_close).to_pickle(os.path.join(path,"price_close.pkl"))
pd.DataFrame(amount).to_pickle(os.path.join(path,"amount.pkl"))
pd.DataFrame(volume).to_pickle(os.path.join(path,"volume.pkl"))
pd.DataFrame(stock_return).to_pickle(os.path.join(path,"stock_return.pkl"))
