# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:40:45 2019

@author: hxzq
"""
import numpy as np
from scipy.stats.stats import pearsonr
import pandas as pd 
import os

PATH_RAW = "data\\raw"
PATH_INTERM = "data\\interm"
PATH_FINAL = "data\\processed"

class Barra_fac:
    '''该类用来保存barra因子'''
    def __init__(self,start_date,curr_date):      
        quota_path = os.path.join(PATH_RAW,"factor_explore")
        stock_code = pd.read_csv(os.path.join(quota_path, 'STOCK_CODE.csv'), header=None).values.T[0].tolist()
        #all_tradedates = pd.read_csv(os.path.join(quota_path, 'ALLDATES.csv'), header=None, names=['TRADE_DT']).values.T[0].tolist()
        #trade_dt = [x for x in all_tradedates if (x >= start_date) & (x < curr_date)]
        all_tradedates  = pd.read_csv(os.path.join(PATH_INTERM,"trade_dt.csv"),header= None).iloc[:,0].tolist()
        trade_dt = [x for x in all_tradedates if (x >= start_date) & (x < curr_date)]
        start_index = all_tradedates.index(trade_dt[0])
        end_index = all_tradedates.index(trade_dt[-1])
        self.date_list = trade_dt
        self.stock_code = stock_code
        self.barra_fac_list = ['BETA','BTOP', 'EARNYILD','GROWTH', 'LEVERAGE','LIQUIDTY','MOMENTUM','RESVOL','SIZE','SIZENL']
        number = len(self.barra_fac_list)
        temp = pd.read_pickle(os.path.join(PATH_INTERM,"barr_fac.pkl"))
        self.barra_fac_np = temp.values[start_index*number:(end_index + 1)*number]
        
    def barra_fac_date(self,date_index):
        if date_index > len(self.date_list) or date_index < 0:
            print ("The date is not in date list")
            return None
        else:
            number = len(self.barra_fac_list)
            return (self.barra_fac_np[date_index*number:(date_index + 1)*number])

class Data:
    '''该类保存了用来输入到表达式中的原始数据'''
    def __init__(self,start_date,curr_date):
        all_tradedates  = pd.read_csv(os.path.join(PATH_INTERM,"trade_dt.csv"),header= None).iloc[:,0].tolist()
        trade_dt = [x for x in all_tradedates if (x >= start_date) & (x < curr_date)]
        self.date_list = trade_dt
        start_index = all_tradedates.index(trade_dt[0])
        end_index = all_tradedates.index(trade_dt[-1])
        self.price_close_adj = pd.read_pickle(os.path.join(PATH_INTERM,"price_close_adj.pkl")).values[start_index:(end_index+1)]
        self.price_high_adj = pd.read_pickle(os.path.join(PATH_INTERM,"price_high_adj.pkl")).values[start_index:(end_index+1)]
        self.price_open_adj = pd.read_pickle(os.path.join(PATH_INTERM,"price_open_adj.pkl")).values[start_index:(end_index+1)]
        self.price_low_adj = pd.read_pickle(os.path.join(PATH_INTERM,"price_low_adj.pkl")).values[start_index:(end_index+1)]
        self.price_vwap_adj = pd.read_pickle(os.path.join(PATH_INTERM,"price_vwap_adj.pkl")).values[start_index:(end_index+1)]
        self.volume_adj = pd.read_pickle(os.path.join(PATH_INTERM,"volume_adj.pkl")).values[start_index:(end_index+1)]
        self.real_stock_return = pd.read_pickle(os.path.join(PATH_INTERM,"real_stock_return.pkl")).values[start_index:(end_index+1)]
        self.stock_return = pd.read_pickle(os.path.join(PATH_INTERM,"stock_return.pkl")).values[start_index:(end_index+1)]

#剔除掉所有空值，条件是所有barra因子空值位置相同
def dropna_row(x, method = 1):
    '''删除空值'''
    #如果一行有空值的话就把一行都删除，所以需要先转置保证每一行是一个股票
    if method == 1:
        return (x[~np.isnan(x).any(axis=1)])
    else:
        return x[~np.isnan(x)]
    
# 矩阵求解最主要的问题是数据中不能有空值，时间可以很快
def linearRegLsq(x,y):
    '''最小二乘法直接求解回归系数'''
    xtx = np.dot(x.T, x)
    if np.linalg.det(xtx) == 0.0: # 判断xtx行列式是否等于0，奇异矩阵不能求逆
        #print('Can not resolve the problem')
        return None
    theta_lsq = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    return theta_lsq
def add_constant(x):
    return np.c_[x,np.ones(x.shape[0])]


#计算因子中性化然后计算因子的收益率，两次回归
def cal_fac_return(fac_np,start_index,Barra, real_stock_return,selected_fac = None,method = 2):
    '''
    这个函数用来做因子中性化，然后计算因子收益率
    fac_np为因子举证
    start_index为因子数据的开始日期在统一的date_list的位置
    Barra为之前定义的Barra类，用来中性化
    real_stock_return为data_para中的下期收益率，用来计算收益率
    selected_fac为已经选中的因子
    method为计算IR的方式，1表示因子收益率IR,2表示IC_IR
    '''
    new_fac_return_list = np.full(fac_np.shape[0],np.nan)
    new_fac_np = np.copy(fac_np) #保存中性化之后的因子
    if not selected_fac is None:
        selected_fac_number = int(selected_fac.shape[1] / new_fac_np.shape[1])
    for ii in range(fac_np.shape[0]):
        #读取新的因子和已有因子与风格因子横截面数据
        new_fac = fac_np[ii]
        barr_fac = Barra.barra_fac_date(ii + start_index)
        #去空值
        if not selected_fac is None:
            selected_fac_day = selected_fac[ii + start_index]
            barr_fac = np.concatenate((barr_fac,selected_fac_day.reshape(selected_fac_number,barr_fac.shape[1])),axis = 0)
        x = dropna_row(barr_fac.T)
        y = dropna_row(new_fac,method = 2)
        if len(y) == 0 or len(x) == 0: #如果当期因子值全是空值则跳过
            continue
        #中性化
        barra_coef = linearRegLsq(add_constant(x),y) #中性化,有截距项
        if barra_coef  is None:
            continue
        y_neutral = y - np.dot(add_constant(x),barra_coef) 
        new_fac_np[ii][~np.isnan(new_fac_np[ii])] = y_neutral #储存中性化之后的因子
        #计算剔除被已有因子所解释的收益率之后的部分
        stock_return_date = dropna_row(real_stock_return[ii + start_index],method = 2)    
        barra_ret = linearRegLsq(x,stock_return_date) #和下期收益率进行回归
        ret_residual = stock_return_date  - np.dot(x,barra_ret)
        
        if method == 1:    
            #计算新因子收益率
            new_fac_return = linearRegLsq(add_constant(y_neutral),ret_residual) #计算收益的时候添加了截距项
            if new_fac_return  is None:
                continue
            new_fac_return_list[ii] = new_fac_return[0]
        elif method == 2:
            #计算IC
            new_fac_return = pearsonr(y_neutral,ret_residual)
            if new_fac_return  is None:
                continue
            new_fac_return_list[ii] = new_fac_return[0]
    return new_fac_np, new_fac_return_list

def annualized_return(fac_return,cal_type = 1):
    '''年化收益率'''
    if np.isnan(fac_return).sum()/float(len(fac_return)) > 0.2: #缺失值比例超过20%则认为收益为0
        return 0.
    if cal_type == 0:
        fac_return = fac_return + 1 
        return (np.nanprod(fac_return)**(252./float(len(fac_return))) - 1)*100
    elif cal_type == 1:
        return abs(np.nanmean(fac_return))  *252.*100

def annualized_volatility(fac_return):
    '''年化波动率'''
    if np.isnan(fac_return).sum()/float(len(fac_return)) > 0.2: #缺失值比例超过20%则认为收益为0
        return 1.
    return (252.**0.5*np.nanstd(fac_return,ddof = 1))*100

def cal_IR(fac_return):
    '''因子IR'''
    return annualized_return(fac_return)/annualized_volatility(fac_return)

#def cal_IR(return_list):
#    return np.nanmean(return_list) / np.nanstd(return_list)

#去极值
def drop_extreme(x):
    '''利用绝对中位数法去极值，并将超过极值的数均匀分配在一定范围内'''
    mf = np.nanmedian(x,axis = 1)
    MAD = np.nanmedian(abs(x.T -  mf),axis = 0)
    sigma = 1.483
    up_limit = mf + 3 * sigma * MAD
    down_limit = mf - 3 * sigma * MAD
    
    up_limit_index = x.T > up_limit
    down_limit_index = x.T < down_limit

    exceed_up = np.full(x.T.shape,np.nan)
    exceed_up[up_limit_index] = x.T[up_limit_index]
    
    exceed_down = np.full(x.T.shape,np.nan)
    exceed_down[down_limit_index] = x.T[down_limit_index]
  
    exceed_up_df = pd.DataFrame(exceed_up).rank(axis = 0, method = "first")
    exceed_up = (exceed_up_df * 0.5 * sigma * MAD / (exceed_up_df.count(axis = 0) + 1)).values + up_limit
    exceed_down_df = pd.DataFrame(exceed_down).rank()
    exceed_down = (exceed_down_df * -0.5 * sigma * MAD / (exceed_down_df.count(axis = 0) + 1)).values + down_limit


    x.T[up_limit_index] = exceed_up[up_limit_index]
    x.T[down_limit_index] = exceed_down[down_limit_index]
    
    return x



def vdiv(A,B, result = 1):
    '''矩阵除法，除0为1或0'''
    matirix_na = np.ones(A.shape)
    matirix_na[np.isnan(A*B)] = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( A, B )
    c[ np.isinf( c ) ] = result  # -inf inf
    c[ A == 0 ] = 0 # 0/0 = 0 not nan
    #c[ np.isnan( c ) ] = result  # nan
    return c * matirix_na

#标准化
def fac_standardize(x):
    '''标准化'''
    #由于broadcast按照行进行复制，所以需要转置进行运算
    return (vdiv((x.T - np.nanmean(x,axis = 1)),np.nanstd(x,axis = 1)).T)

