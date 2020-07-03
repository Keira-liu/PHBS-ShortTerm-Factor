# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:16:34 2019

@author: hxzq
"""

import pandas as pd
import numpy as np
import os
import scipy.io

PATH_RAW = "data\\raw"
PATH_INTERM = "data\\interm"
PATH_FINAL = "data\\processed"
barra_path = os.path.join(PATH_RAW,"factor_explore\\barra")
quota_path = os.path.join(PATH_RAW,"factor_explore")

csv_path = quota_path
start_date = 20141201
curr_date = 20190930
#curr_date = np.inf
#cur_date = int((datetime.datetime.now()).strftime('%Y%m%d'))

# read stock_code, trade_dt
stock_code = pd.read_csv(os.path.join(csv_path, 'STOCK_CODE.csv'), header=None).values.T[0].tolist()
all_tradedates = pd.read_csv(os.path.join(csv_path, 'ALLDATES.csv'), header=None, names=['TRADE_DT']).values.T[0].tolist()
trade_dt = [x for x in all_tradedates if (x >= start_date) & (x < curr_date)]



def GeneratePoolData(csv_path):
    dt_list = pd.read_csv(os.path.join(csv_path, 'DT_LIST.csv'), names=['STOCK_CODE', 'DT_LIST', 'DT_DELIST'])
    dt_list['DT_OLD'] = dt_list['DT_LIST'] + 10000
    dt_list['IN'] = True
    dt_list['OUT'] = False
    old_data = pd.pivot_table(dt_list, values='IN', index='DT_OLD', columns='STOCK_CODE').fillna(method='pad').fillna(False).loc[all_tradedates, stock_code].fillna(method='pad').fillna(False).loc[trade_dt, :]
    del_data = pd.pivot_table(dt_list, values='OUT', index='DT_DELIST', columns='STOCK_CODE').fillna(method='pad').fillna(True).loc[all_tradedates, stock_code].fillna(method='pad').fillna(True).loc[trade_dt, :]
    old_stock = old_data & del_data
    isst = pd.read_csv(os.path.join(csv_path, 'ISST.csv'), names=stock_code, index_col=0)
    #isst = ReadRawFactor(factor_name='RAW_ISST')
    istrade = pd.read_csv(os.path.join(csv_path, 'ISTRADEDAY.csv'), names=stock_code, index_col=0)
    #istrade = ReadRawFactor(factor_name='RAW_ISTRADEDAY')
    #price_close = pd.read_csv(os.path.join(csv_path,"CLOSE.csv"),index_col = 0,names = stock_code)
    #price_open = pd.read_csv(os.path.join(csv_path,"OPEN.csv"),index_col = 0,names = stock_code)
    stock_return = (pd.read_csv(os.path.join(csv_path,"RETURN.csv"),index_col = 0,names = stock_code))
    pool_data = old_stock & (isst < 0.5) & (istrade > 0.5)
    pool_data = pool_data.replace(np.nan,0).astype(bool)
    pool_data = pool_data & pool_data.shift(-1) & pool_data.shift(-2)
    limited = abs(stock_return) < 0.095 #当天涨跌停股票也不在pool里
    limited = limited & limited.shift(-1) & limited.shift(-2) #之后两天如果有涨跌停的话，对应的股票也不再当天的股票池里
    pool_data = pool_data & limited
    return pool_data

all_pool = GeneratePoolData(csv_path)
all_pool["601360.SH"] = np.nan #这只股票部分时间没有barra因子，先从pool里删去
all_pool = all_pool.replace(0,np.nan).astype(float)
all_pool.to_pickle(os.path.join(PATH_INTERM,"all_pool.pkl"))

#读取barrra因子文件
def barra_fac(barra_file,stock_code = stock_code , date_list = trade_dt, all_pool = all_pool.loc[trade_dt]):
    temp = pd.read_csv(os.path.join(barra_path,barra_file),index_col = 0,names = stock_code).loc[date_list]
    temp = temp * all_pool
    print ('内存占据内存约: {:.2f} MB'.format(temp.memory_usage().sum()/ (1024**3) * 1024))
    return temp
#读取所有barra因子文件，保存在dict中
files_total = os.listdir(barra_path)
barra_fac_total = {}
for fac in files_total:
    barra_fac_total[fac.replace(".csv","")] = barra_fac(fac)
#将barra因子文件合并，index是日期乘以barra因子数，保存到numpy中，方便切片读取
barra_fac_np = np.zeros((len(trade_dt)*10,len(stock_code)))
ii = 0
for date in trade_dt:
    jj = 0
    for fac in files_total:
        fac = fac.replace(".csv","")
        barra_fac_np[ii*len(files_total)+jj] = barra_fac_total[fac].loc[date].values
        jj = jj + 1
    ii = ii + 1
print (barra_fac_np.shape)
temp = pd.DataFrame(barra_fac_np)
print ('内存占据内存约: {:.2f} MB'.format(temp.memory_usage().sum()/ (1024**3) * 1024))
temp.to_pickle(os.path.join(PATH_INTERM,"barr_fac.pkl"))

#处理原始数据

stock_return = (pd.read_csv(os.path.join(csv_path,"RETURN.csv"),index_col = 0,names = stock_code) * all_pool ).loc[trade_dt].values
price_close = (pd.read_csv(os.path.join(csv_path,"CLOSE.csv"),index_col = 0,names = stock_code,dtype=float) * all_pool).loc[trade_dt].values
price_high = (pd.read_csv(os.path.join(csv_path,"HIGH.csv"),index_col = 0,names = stock_code,dtype=float) * all_pool).loc[trade_dt].values
price_open = (pd.read_csv(os.path.join(csv_path,"OPEN.csv"),index_col = 0,names = stock_code,dtype=float) * all_pool).loc[trade_dt].values
price_low = (pd.read_csv(os.path.join(csv_path,"LOW.csv"),index_col = 0,names = stock_code,dtype=float) * all_pool).loc[trade_dt].values
amount = (pd.read_csv(os.path.join(csv_path,"AMOUNT.csv"),index_col = 0,names = stock_code,dtype=float) * all_pool).loc[trade_dt].values
volume = (pd.read_csv(os.path.join(csv_path,"VOLUME.csv"),index_col = 0,names = stock_code,dtype=float) * all_pool).loc[trade_dt].values
vwap = amount/volume
adjfac_total = pd.read_csv(os.path.join(csv_path,"ADJFACTOR.csv"),index_col = 0,names = stock_code,dtype=float)

#复权
adjfac = (adjfac_total * all_pool).loc[trade_dt].values
def adj_price(price,adjfac = adjfac,method = None):
    if method is None:
        return price
    elif method == "forward":
        return  price * adjfac / adjfac[-1]
    elif method == "backward":
        return price * adjfac 

price_close_adj = adj_price(price_close)
price_high_adj = adj_price(price_high)
price_open_adj = adj_price(price_open)
price_low_adj = adj_price(price_low)
price_vwap_adj = adj_price(vwap)
volume_adj  = adj_price(volume)

#获取twap数据，用来计算真实的股票收益
temp = scipy.io.loadmat(os.path.join(PATH_RAW,"5022.mat"))
stock_list_temp = []
for i in range(len(temp["WIND_CODE"].tolist()[0])):
    stock_list_temp.append(temp["WIND_CODE"].tolist()[0][i].tolist()[0])
half_hour_stock_price = pd.DataFrame(temp["FACTOR_SCORE"],
                                     index = temp["TRADE_DT"][0].tolist(),columns = stock_list_temp)
half_hour_stock_price = half_hour_stock_price.reindex(index = trade_dt,columns = stock_code)
half_hour_stock_price_adj = adj_price(half_hour_stock_price,adjfac_total.loc[trade_dt].values) #将twap数据复权
half_hour_stock_price_adj = half_hour_stock_price_adj.fillna(pd.DataFrame(price_open_adj,index = trade_dt,columns = stock_code)) #由于twap存在一些空值，利用开盘价填充

real_stock_return = (half_hour_stock_price_adj.shift(-2) / half_hour_stock_price_adj.shift(-1) - 1)  #计算下期收益率，并将收益率保存在当期
#如果两天开盘价变动幅度的大于21%则设置成21%
real_stock_return[real_stock_return > 0.21] = 0.21
real_stock_return[real_stock_return < -0.21] = -0.21
real_stock_return  = real_stock_return .fillna(pd.DataFrame(stock_return,index = trade_dt,columns = stock_code)).values #由于仍然有几个数据是空值，利用stock_return数据进行填充
real_stock_return = real_stock_return * all_pool.loc[trade_dt].values #将利用全市场pool过滤收益率数据

pd.DataFrame(price_close_adj).to_pickle(os.path.join(PATH_INTERM,"price_close_adj.pkl"))
pd.DataFrame(price_high_adj).to_pickle(os.path.join(PATH_INTERM,"price_high_adj.pkl"))
pd.DataFrame(price_open_adj).to_pickle(os.path.join(PATH_INTERM,"price_open_adj.pkl"))
pd.DataFrame(price_low_adj).to_pickle(os.path.join(PATH_INTERM,"price_low_adj.pkl"))
pd.DataFrame(price_vwap_adj).to_pickle(os.path.join(PATH_INTERM,"price_vwap_adj.pkl"))
pd.DataFrame(volume_adj).to_pickle(os.path.join(PATH_INTERM,"volume_adj.pkl"))
pd.DataFrame(real_stock_return).to_pickle(os.path.join(PATH_INTERM,"real_stock_return.pkl"))
pd.DataFrame(stock_return).to_pickle(os.path.join(PATH_INTERM,"stock_return.pkl"))
pd.Series(trade_dt).to_csv(os.path.join(PATH_INTERM,"trade_dt.csv"),index = False)

pd.Series(trade_dt[:-2]).to_csv(os.path.join(PATH_FINAL,"fac_index.csv"),index = False)
pd.Series(stock_code).to_csv(os.path.join(PATH_FINAL,"fac_columns.csv"),index = False)