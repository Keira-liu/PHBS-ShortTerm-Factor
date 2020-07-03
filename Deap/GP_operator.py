# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:28:10 2019

@author: hxzq
"""
import pandas as pd
import numpy as np
import bottleneck as bk



def Rank(A):
    '''
    横截面排序
    '''
    return bk.nanrankdata(A,axis = 1)


def Stddev_old(A,window):
    '''
    window天的移动标准差
    window >= 2
    '''
    if window < 2:
        #print ("计算stddev，n不能小于2，返回输入")
        return A
    matirix_na = np.ones(A.shape)
    matirix_na[np.isnan(A)] = np.nan
    result = pd.DataFrame(A).rolling(window,axis = 0,min_periods = 2).std() 
    result = ((result.T).fillna(result.mean(axis = 1,skipna = True)) ).T #利用每一天所有股票的均值来填充空值，根据broadcast的原理，需要转置后再填充
    return result.values * matirix_na #保持空值位置与原始的位置相同

def fillna(A,axis = 1):
    '''填充空值'''
    col_mean = np.nanmean(A, axis= axis )
    inds = np.where(np.isnan(A))
    A[inds] = np.take(col_mean, inds[1 - axis])
    return A

def Stddev(A,n):
    '''
    window天的移动标准差
    window >= 2
    '''
    if n < 2:
        #print ("计算stddev，n不能小于2，返回输入")
        return A
    result = bk.move_std(A,n,min_count=2,axis = 0,ddof=1)
    result = fillna(result) #利用每一天所有股票的均值来填充空值，根据broadcast的原理，需要转置后再填充
    result[np.isnan(A)] = np.nan
    return result

def Delay(A,n):
    '''过去n天的数值'''
    if n < 1:
        return A
    temp = np.roll(A,n,axis = 0)
    #利用移动平均值填充空值
    fillna_value = bk.move_mean(A,n+1,axis = 0,min_count = 1)
    temp[np.isnan(temp)] = fillna_value[np.isnan(temp)]
    temp[:n] = np.nan
    temp[np.isnan(A)] = np.nan
    return temp

def Delta(A,n):
    '''当日值减去过去n天的数值'''
    res =  vsub(A , Delay(A,n))
    return res

def vlog(A):
    #如果存在小于等于0的值则返回原数据
    if (A <= 0).sum() > 0:
        #print ("log函数的输入不全大于0，返回输入")
        return A
    return np.log(A)

def vadd(A,B):
    '''矩阵加法,一个值加空值之后不会成为空值，全是空值相加结果为0
    '''
    matirix_na = np.ones(A.shape)
    matirix_na[np.isnan(A*B)] = np.nan
    return np.nansum(np.dstack((A,B)),2) * matirix_na
def vsub(A,B):
    '''矩阵减法'''
    matirix_na = np.ones(A.shape)
    matirix_na[np.isnan(A*B)] = np.nan
    return np.nansum(np.dstack((A,(-1.* B))),2) * matirix_na
def vmul(A,B):
    '''矩阵乘法'''
    return A * B
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

def Abs(A):
    '''绝对值'''
    return np.abs(A)

def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def Prod_shift(A,n):
    '''
    计算n天滚动累乘,在向前看的周期较短（<10)时耗时远小于rolling
    n >= 1
    '''
    if n < 1:
        #print ("计算n天累乘，n不得小于1，返回输入")
        return A
    stacked = np.empty((n,A.shape[0], A.shape[1]))
    for i in range(n):
        temp = shift(A,i)
        stacked[i] = temp
    result = np.nanprod(stacked,axis = 0)
    result[np.isnan(A)] = np.nan
    if np.isinf(result).sum() > 0:
        #print ("累乘计算出现无穷值，返回输入")
        return A
    return result


def Prod(A,n):
    '''
    计算n天滚动累乘
    n >= 1
    '''
    if n < 1:
        #print ("计算n天累乘，n不得小于1，返回输入")
        return A
    matrix_na = vdiv(A,A)
    temp = pd.DataFrame(A).fillna(1)
    result  = temp.rolling(n,axis = 0).apply(np.prod,raw = True).values * matrix_na
    if np.isinf(result).sum() > 0:
        #print ("累乘计算出现无穷值，返回输入")
        return A
    return result

def Rolling_mean_old(A,n):
    '''列方向n天的移动平均值'''
    if n < 1:
        #print ("计算n天均值，n不得小于1，返回输入")
        return A
    result  = []
    matirix_na = np.ones(A.shape)
    for i in range(n):
        temp = np.roll(A,i,axis = 0)
        temp[:i] = np.nan
        result.append(temp)
    result = np.nanmean(np.dstack(result),2)
    return result * matirix_na

def Rolling_mean(A,n):
    '''列方向n天的移动平均值'''
    if n < 1:
        #print ("计算n天均值，n不得小于1，返回输入")
        return A
    result = bk.move_mean(A,n,axis = 0,min_count = 1)
    result[np.isnan(A)] = np.nan
    return result

def Rolling_weighted_mean(A,n,weights):
    '''
    用来计算列方向的加权移动均值
    weights为从T到T-n+1的相对权重
    '''
    if n < 1:
        #print ("计算n天加权均值，n不得小于1，返回输入")
        return A
    stacked = np.empty((n,A.shape[0], A.shape[1]))
    matrix_na_total = np.zeros(A.shape)
    for i in range(n):
        temp = shift(A, i)
        stacked[i] = temp
        matrix_na_total = matrix_na_total + (~np.isnan(temp))*weights[i]
    result = bk.nansum(stacked,0)
    result = vdiv(result,matrix_na_total)
    result[np.isnan(A)] = np.nan
    return  result

def Sma(A,n,m):
    if n < 2:
        #print ("计算Sma时n需大于2，返回输入")
        return A
    if m >= n:
        #print ("计算Sma时m需小于n，返回输入")
        return A
    weights = [float(m)/float(n)]
    weights.extend([(1-float(m)/float(n))/(n-1)]*(n-1))
    return Rolling_weighted_mean(A,n,weights)

def Wma(A,n):
    '''
    加权移动均值
    n为加权的天数
    '''
    if n < 1:
        #print ("计算n天固定加权均值，n不得小于1，返回输入")
        return A
    weights = np.flip(np.array(range(n))+1)
    return Rolling_weighted_mean(A,n,weights)

def vneg(A):
    '''相反数'''
    return A*(-1.)

def Sign(A):
    '''sign函数'''
    with np.errstate(invalid='ignore'):
        return np.sign(A)

def Ts_sum_old(A,n):
    '''
    过去n（包含当天）求和
    n >= 1
    '''
    if n < 1:
        #print ("计算n天的求和，n不得小于1，返回输入")
        return A
    matirix_na = np.ones(A.shape)
    matirix_na[np.isnan(A)] = np.nan
    result = np.zeros(A.shape)
    for i in range(n):
        temp = np.roll(A,i,axis = 0)
        temp[:i] = np.nan
        result = np.nansum(np.dstack((result,temp)),2)
    return result * matirix_na

def Ts_sum(A,n):
    '''
    过去n（包含当天）求和
    n >= 1
    '''
    if n < 1:
        #print ("计算n天的求和，n不得小于1，返回输入")
        return A
    result = bk.move_sum(A, window=n, min_count=1,axis = 0)
    result[np.isnan(A)] = np.nan
    return result

def Ts_min_old(A,n):
    '''
    计算n天(包括当天)的最小值
    n >= 1
    '''
    if n < 1:
        #print ("计算n天的最小值，n不得小于1，返回输入")
        return A
    matirix_na = np.ones(A.shape)
    matirix_na[np.isnan(A)] = np.nan
    for i in range(n):
        temp = np.roll(A,i,axis = 0)
        temp[:i] = np.nan
        if i == 0:
            result = temp
        else:
            result = np.nanmin(np.dstack((result,temp)),2)
    return result * matirix_na

def Ts_min(A,n):
    '''
    计算n天(包括当天)的最小值
    n >= 1
    '''
    if n < 1:
        #print ("计算n天的最小值，n不得小于1，返回输入")
        return A
    result = bk.move_min(A,n,min_count=1,axis = 0)
    result[np.isnan(A)] = np.nan
    return result

def Ts_max(A,n):
    '''
    计算n天(包括当天)的最大值
    n >= 1
    '''
    if n < 1:
        #print ("计算n天的最大值，n不得小于1，返回输入")
        return A
    result = bk.move_max(A,n,min_count=1,axis = 0)
    result[np.isnan(A)] = np.nan
    return result

def Ts_max_old(A,n):
    '''
    计算n天(包括当天)的最大值
    n >= 1
    '''
    if n < 1:
        #print ("计算n天的最大值，n不得小于1，返回输入")
        return A
    matirix_na = np.ones(A.shape)
    matirix_na[np.isnan(A)] = np.nan
    for i in range(n):
        temp = np.roll(A,i,axis = 0)
        temp[:i] = np.nan
        if i == 0:
            result = temp
        else:
            result = np.nanmax(np.dstack((result,temp)),2)
    return result * matirix_na

def Argmin_old(A,n):
    '''
    过去n（包含当天）最小位置
    n >= 1
    '''
    if n < 1:
        #print ("计算n天的最小值位置，n不得小于1，返回输入")
        return A
    matirix_na = np.ones(A.shape)
    matirix_na[np.isnan(A)] = np.nan
    result = []
    for i in range(n):
        temp = np.roll(A,i,axis = 0)
        temp[:i] = np.nan
        #为了防止所有值都是nan不满足nanargmin条件，将当天的值加上0，保证当天的值不为空
        if i == 0:
            temp = np.nansum(np.dstack((temp,np.zeros(temp.shape))),2) 
        result.append(temp)
    temp = np.dstack(result)
    result = np.nanargmin(np.dstack(result),2) + 1
    return result  * matirix_na

def Argmax_old(A,n):
    '''
    过去n（包含当天）最大值位置
    n >= 1
    '''
    if n < 1:
        #print ("计算n天的最大值位置，n不得小于1，返回输入")
        return A
    matirix_na = np.ones(A.shape)
    matirix_na[np.isnan(A)] = np.nan
    result = []
    for i in range(n):
        temp = np.roll(A,i,axis = 0)
        temp[:i] = np.nan
        if i == 0:
            temp = np.nansum(np.dstack((temp,np.zeros(temp.shape))),2)
        result.append(temp)
    result = np.nanargmax(np.dstack(result),2) + 1
    return result * matirix_na

def Argmax(A,n):
    '''
    过去n（包含当天）最大值位置
    n >= 1
    '''
    if n < 1:
        #print ("计算n天的最大值位置，n不得小于1，返回输入")
        return A
    result = bk.move_argmax(A,n,min_count= 1, axis = 0) + 1
    result[np.isnan(A)] = np.nan
    return result

def Argmin(A,n):
    '''
    过去n（包含当天）最大值位置
    n >= 1
    '''
    if n < 1:
        #print ("计算n天的最大值位置，n不得小于1，返回输入")
        return A
    result = bk.move_argmin(A,n,min_count= 1, axis = 0) + 1
    result[np.isnan(A)] = np.nan
    return result

def Cov(A,B,n):
    '''
    计算两个因子向前n天的协方差
    n >= 2
    '''
    if n < 2:
        #print ("计算A和B n天的协方差，n不得小于2，返回A")
        return A
    result_A = []
    result_B = []
    for i in range(n):
        temp_A = np.roll(A,i,axis = 0)
        temp_A[:i] = np.nan
        temp_B = np.roll(B,i,axis = 0)
        temp_B[:i] = np.nan
        result_A.append(temp_A)
        result_B.append(temp_B)
    result_A = np.stack(result_A)
    mean = bk.nanmean(result_A,axis = 0)
    result_A = result_A - mean
    result_B = np.stack(result_B)
    mean = bk.nanmean(result_B,axis = 0)
    result_B = result_B - mean
    result = bk.nanmean(result_A * result_B,axis = 0)
    result[np.isnan(A*B)] = np.nan
    return result

def Corr(A,B,n):
    '''
    计算两个因子向前n天的相关系数
    n >= 2
    '''
    if n < 2:
        #print ("计算A和B n天的相关系数，n不得小于2，返回输入")
        return A
    stacked_A = np.empty((n,A.shape[0], A.shape[1]))
    stacked_B = np.empty((n,B.shape[0], B.shape[1]))
    for i in range(n):
        temp_A = shift(A,i)
        stacked_A[i] = temp_A
        temp_B = shift(B,i)
        stacked_B[i] = temp_B
    mean = bk.nanmean(stacked_A,axis = 0)
    A_submean = stacked_A - mean
    mean = bk.nanmean(stacked_B,axis = 0)
    B_submean = stacked_B - mean
    deno = bk.nanstd(stacked_A,axis = 0) * bk.nanstd(stacked_B,axis = 0 )
    cov = bk.nanmean(A_submean * B_submean,axis = 0)
    result = vdiv(cov,deno,0)
    result[np.isnan(A*B)] = np.nan
    return result

def Max(A,b):
    #比较矩阵和单个值，取其大
    result = A.copy()
    result[result < b] = b
    return result

def Min(A,b):
    #比较矩阵和单个值，取其小
    result = A.copy()
    result[result > b] = b
    return result



def Mean2(A,B):
    return np.mean(np.dstack((A,B)),axis = 2)

def Mean3(A,B,C):
    return np.mean(np.dstack((A,B,C)),axis = 2)

def Itself_value(A):
    return A

def Itself_int(a):
    return a

def Clear_by_bound(A,B,C):
    result = C.copy()
    result[A < B] = 0
    result[np.isnan(C)] = np.nan
    return result

def If_then_else(A,B,C,D):
    #def if_then_else(a,b,c,d):
    #    return c if a < b else d
    #If_then_else_vec = np.vectorize(if_then_else)
    result = D.copy()
    result[A<B] = C[A<B]
    result[np.isnan(C*D)] = np.nan
    return result

def Ts_rank(A,n):
    if n < 2:
        #print ("计算周期的排序，n不得小于2，返回输入")
        return A
    result = bk.move_rank(A,n,axis = 0,min_count = 1)
    result[np.isnan(A)] = np.nan
    result = result + 2 # 不希望值中出现0
    return result

def Sqrt(A):
    '''求平方根'''
    if (A < 0).sum() > 0:
        #print ("求平方根函数不能有负值，返回输入")
        return A
    return  np.sqrt(A)

def Inv(A):
    '''求倒数'''
    return vdiv(np.ones(A.shape),A,0)

def Sin(A):
    '''三角函数'''
    return np.sin(A)

def Cos(A):
    '''三角函数'''
    return np.cos(A)