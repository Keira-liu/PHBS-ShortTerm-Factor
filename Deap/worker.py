# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:17:28 2019

@author: hxzq
"""

import numpy as np
import GP_operator


from deap import base
from deap import creator
from deap import tools
from deap import gp

import os
import pandas as pd
import multiprocess_func
import time
import pickle
import random

PATH_RAW = "data\\raw"
PATH_INTERM = "data\\interm"
PATH_FINAL = "data\\processed"



#用来储存遗传算法参数和结果
class GP_result:
    def __init__(self):
        self.tournsize = 10
        self.deep_min = 1 #最小树深度
        self.deep_max = 10 #最大树深度
        self.CXPB = 0.4 #交叉概率
        self.MUPB = 0.1 #变异概率
        self.threshold_2year = 5.
        self.threshold_4year = 3.
        self.selected_fac = None 
        self.useful_formula_orgin = []          
        self.useful_formula = pd.DataFrame(columns=["formula","IR_4year","IR_2year","time_used"])
        self.useful_fac_n = 0
        self.total_fac_n = 0
        self.each_thousand_time = 0
        self.each_useful_fac_time = 0
        self.ret_df = None
        self.IC_df = None
    def useful_fac_n_add(self):
        self.useful_fac_n = self.useful_fac_n + 1
    def total_fac_n_add(self, n = None):
        if n is None:
            self.total_fac_n = self.total_fac_n + 1
        else:
            self.total_fac_n = self.total_fac_n + n

    def fac_result_add(self,result_list):
        self.useful_formula.loc[self.useful_fac_n] = result_list
    def fac_orgin_add(self,individual):
        self.useful_formula_orgin.append(individual)
    def selected_fac_add(self,new_fac):
        if self.selected_fac is None:
           self.selected_fac  = new_fac             
        else:
           self.selected_fac = np.concatenate((self.selected_fac,new_fac),axis = 1)
    def continous(self):
        '''从之前寻找的因子的基础上进行挖掘'''
        files_total = os.listdir(PATH_FINAL)
        if ("selected_fac.pkl" in files_total and "formula_result.csv" in 
            files_total and "IC_df.pkl" in files_total and "ret_df.pkl" in files_total):
            self.selected_fac = pd.read_pickle(os.path.join(PATH_FINAL,"selected_fac.pkl")).values
            self.useful_formula = pd.read_csv(os.path.join(PATH_FINAL,"formula_result.csv"),index_col = 0)
            self.ret_df = pd.read_pickle(os.path.join(PATH_FINAL,"ret_df.pkl"))
            self.IC_df = pd.read_pickle(os.path.join(PATH_FINAL,"IC_df.pkl"))
            self.useful_fac_n = self.useful_formula.shape[0]
    def crate_ret_df(self,Barra):
        self.ret_df = pd.DataFrame(columns = Barra.date_list[:-2])
        self.IC_df = pd.DataFrame(columns = Barra.date_list[:-2])
        return
    def save_ret(self,ret_list,IC_list):
        if self.ret_df is None or self.IC_df is None:
            print ("需要先创建存储ret_list的dataframe")
            return 
        if len(ret_list) != self.ret_df.shape[1]:
            print ("rest_list时间范围不正确")
            return
        self.ret_df.loc[self.useful_fac_n] = ret_list
        self.IC_df.loc[self.useful_fac_n] = IC_list
        return 

    
gp_result = GP_result()  


pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.vadd, [np.ndarray,np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.vsub, [np.ndarray,np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.vmul, [np.ndarray,np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.vdiv, [np.ndarray,np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.vneg,[np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.vlog,[np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.Sign,[np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.Rank, [np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.Delay, [np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Delta, [np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Stddev, [np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Abs,[np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.Prod_shift, [np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Sma,[np.ndarray,int,int],np.ndarray)
pset.addPrimitive(GP_operator.Rolling_mean,[np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Cov,[np.ndarray,np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Corr,[np.ndarray,np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Ts_max,[np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Ts_min,[np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Ts_sum,[np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Ts_rank,[np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Argmax,[np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Argmin,[np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Cos,[np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.Sin,[np.ndarray],np.ndarray)
#pset.addPrimitive(GP_operator.Max,[np.ndarray,int],np.ndarray)
#pset.addPrimitive(GP_operator.Min,[np.ndarray,int],np.ndarray)
pset.addPrimitive(GP_operator.Clear_by_bound,[np.ndarray,np.ndarray,np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.If_then_else,[np.ndarray,np.ndarray,np.ndarray,np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.Mean2,[np.ndarray,np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.Mean3,[np.ndarray,np.ndarray,np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.Itself_value,[np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.Itself_int, [int],int)
pset.addPrimitive(GP_operator.Inv, [np.ndarray],np.ndarray)
pset.addPrimitive(GP_operator.Sqrt, [np.ndarray],np.ndarray)

#重命名变量名
pset.renameArguments(ARG0='High')
pset.renameArguments(ARG1='Open')
pset.renameArguments(ARG2='Low')
pset.renameArguments(ARG3='Close')
pset.renameArguments(ARG4='Vwap')
pset.renameArguments(ARG5='Volume')
pset.renameArguments(ARG6='Ret')

#加入int类型的叶子节点，作为函数的参数
i = 0
j = 1
#命名不能和曾经的命名重复
while i == 0 : 
    name = "rand" + str(j)
    try:
        pset.addEphemeralConstant(name,lambda: np.random.randint(0,10),int)
        i = 1
    except:
        i = 0
        j = j + 1



#创建遗传算法类，目标是是最大值
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#创建遗传算法里每一个个体类
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
#创建遗传算法类的内置函数
toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=gp_result.deep_min, max_=gp_result.deep_max) # 设置树的生成方式，和最大最小深度
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) #加入个体生成函数
toolbox.register("population", tools.initRepeat, list, toolbox.individual) #加入群体生成函数
toolbox.register("compile", gp.compile, pset=pset) #将树结构打转换成可计算的公式
    
def multi_mutate(individual,expr,pset):
    '''定义变异函数'''
    rand = np.random.uniform(0)
    if rand <= 0.33:
        return gp.mutUniform(individual,expr,pset)
    elif rand <= 0.66:
        return gp.mutShrink(individual)
    else:
        return gp.mutNodeReplacement(individual,pset)

toolbox.register("select", tools.selTournament, tournsize=gp_result.tournsize) #交叉或者进化时，每次从中随机选取tournsize个个体，随机抽取群体个数次
toolbox.register("mate", gp.cxOnePoint) #定义交叉点方式
toolbox.register("expr_mut", gp.genGrow, min_=gp_result.deep_min - 1 , max_= gp_result.deep_max) #定义变异后子树生成的方式
toolbox.register("mutate", multi_mutate, expr=toolbox.expr_mut, pset=pset)  #定义变异的方式

def evalSymbReg(individual,start_index,price_high, price_open, price_low, price_close, vwap, volume,ret,selected_fac,Barra,real_stock_return,toolbox,pset,method = 2):
    '''
    用来计算单个表达式的IR和处理之后的因子
    start_index表示所使用数据在所有数据中的位置
    selected_fac为已经挖到的处理之后的因子数据
    Barra为barra因子数据
    real_stock_return为下期收益率数据
    toolbox和pset为之前定义的遗传算法类
    '''
    #print ("formula:",str(individual))
    #print (individual.height)
    # 将树结构转换成可以计算的表达式
    func = toolbox.compile(expr=individual, pset = pset)
    # 生成新因子
    new_fac = func(price_high, price_open, price_low, price_close, vwap, volume,ret)
    if type(new_fac) == float or type(new_fac) == int : #如果得到的新因子不是array格式则抛弃
        return -1,
    if (np.isnan(new_fac).sum() == new_fac.shape[0] * new_fac.shape[1]) or np.nanstd(new_fac) == 0 : #如果新因子全是空值或者都相同则抛弃
        return -1,
    new_fac = multiprocess_func.drop_extreme(new_fac)
    new_fac = multiprocess_func.fac_standardize(new_fac)
    if (np.isnan(new_fac).sum() == new_fac.shape[0] * new_fac.shape[1]) or np.nanstd(new_fac) == 0 : #如果标准化之后全是空值或者都相同则抛弃
        return -1,
    #计算因子收益率
    new_fac, fac_return_list = multiprocess_func.cal_fac_return(new_fac,start_index,Barra,real_stock_return,selected_fac = selected_fac,method = method)
    IR = multiprocess_func.cal_IR(fac_return_list)
    #print (np.isnan(fac_return_list).sum())
    #print ("IR_2year:",IR_2year)
    if IR == np.nan:
        print ("IR is nan：",str(individual))
        return -1,
    return IR,new_fac,fac_return_list


def eval_orgin(individual,High,Open,Low,Close,Vwap,Volume,Return,real_stock_return,Barra,selected_fac):
    '''该函数时在非并行计算的程序中使用的计算表达式IR的函数'''
    #将树结构转换成可以计算的表达式
    #print ("formula:",str(individual))
    #print (individual.height)
    func = toolbox.compile(expr=individual, pset = pset)
    # 测试开始的时间的index
    # 生成新因子
    start_index = int(len(Barra.date_list)/2)    
    new_fac = func(High[start_index:-2],Open[start_index:-2],Low[start_index:-2],Close[start_index:-2],Vwap[start_index:-2],Volume[start_index:-2],Return[start_index:-2])
    
    if type(new_fac) == float or type(new_fac) == int : #如果得到的新因子不是array格式则抛弃
        return -1,
    if (np.isnan(new_fac).sum() == new_fac.shape[0] * new_fac.shape[1]) or np.nanstd(new_fac) == 0 : #如果新因子全是空值或者都相同则抛弃
        return -1,
    new_fac = multiprocess_func.drop_extreme(new_fac)
    new_fac = multiprocess_func.fac_standardize(new_fac)
    if (np.isnan(new_fac).sum() == new_fac.shape[0] * new_fac.shape[1]) or np.nanstd(new_fac) == 0 : #如果标准化之后全是空值或者都相同则抛弃
        return -1,
    new_fac, fac_return_list = multiprocess_func.cal_fac_return(new_fac,start_index,Barra,real_stock_return,selected_fac = selected_fac)
    IR_2year = multiprocess_func.cal_IR(fac_return_list)
    #print (np.isnan(fac_return_list).sum())
    #print ("IR_2year:",IR_2year)
    if abs(IR_2year) > gp_result.threshold_2year:
        start_index = 0
        new_fac = func(High[start_index:-2],Open[start_index:-2],Low[start_index:-2],Close[start_index:-2],Vwap[start_index:-2],Volume[start_index:-2],Return[start_index:-2])
        new_fac = multiprocess_func.drop_extreme(new_fac)
        new_fac = multiprocess_func.fac_standardize(new_fac)
        new_fac, fac_return_list = multiprocess_func.cal_fac_return(new_fac,start_index ,Barra,real_stock_return,selected_fac = selected_fac)
        IR_4year = multiprocess_func.cal_IR(fac_return_list)
        if abs(IR_4year) > gp_result.threshold_4year:
            print ("发现了一个有用的表达式：",str(individual))  
            return (individual,IR_2year)
            #保存表达式原始格式
            #gp_result.fac_orgin_add(individual)
            #output = open(os.path.join(PATH_FINAL,"useful_formula_orgin.pkl"), 'wb')
            #pickle.dump(gp_result.useful_formula_orgin,output)
            #output.close()
            #保存表达式string格式以及IR
            time_used = time.perf_counter() - gp_result.each_useful_fac_time
            gp_result.each_useful_fac_time = time.perf_counter()
            gp_result.fac_result_add([str(individual),IR_4year,IR_2year,time_used])
            gp_result.useful_fac_n_add()
            gp_result.useful_formula.to_csv(os.path.join(PATH_FINAL,"formula_result.csv"))
            gp_result.selected_fac_add(new_fac)
            pd.DataFrame(gp_result.selected_fac).to_pickle(os.path.join(PATH_FINAL,"selected_fac.pkl"))
    '''  
    #输出耗时
    gp_result.total_fac_n_add()
    if gp_result.total_fac_n % 1000 == 0:
        average = (time.perf_counter() - gp_result.each_thousand_time) / 1000
        gp_result.each_thousand_time = time.perf_counter()
        print ("最近1000个因子的平均耗时为:", average)
    '''
    if IR_2year == np.nan:
        print ("IR为空值的表达式：",str(individual))
        IR_2year = 0.
    return abs(IR_2year),

def easimple_multi_pros(q,data_para,Barra,toolbox,pset,order,gp_result,
             POP_SIZE = 100, GEN = 30):
    '''
    该函数用来并行多个遗传算法，这里表示其中的一个进程
    q是进程间通信的queue变量，用来储存相应的结果
    data_para是储存原始数据的类
    Barra是储存原始因子的类
    rand_seed是随机种子
    go_result用来储存遗传算法参数和结果的类
    POP_SIZE为每一代群体的数量
    GEN为多少代
    假设每一代交叉和变异的概率分别为0.1，那么每一代会大约生成pop*0.2个个体
    '''
    CXPB = gp_result.CXPB
    MUPB = gp_result.MUPB
    #random.seed(random_seed)
    #initializing population
    pop = toolbox.population(n = POP_SIZE)
    '''开始进化'''
    print ("单次进化开始，次序：",order)
    
    #evaluating the fitness
    start_index = int(len(Barra.date_list)/2)
    '''
    args_2year = (
            start_index, 
            data_para.price_high_adj[start_index:-2],
            data_para.price_open_adj[start_index:-2],
            data_para.price_low_adj[start_index:-2],
            data_para.price_close_adj[start_index:-2],
            data_para.price_vwap_adj[start_index:-2],
            data_para.volume_adj[start_index:-2],
            data_para.stock_return[start_index:-2],
            gp_result.selected_fac,
            Barra,
            data_para.real_stock_return,
            toolbox,
            pset)
    #start_index = 0
    args_4year = (
            0, 
            data_para.price_high_adj[:-2],
            data_para.price_open_adj[:-2],
            data_para.price_low_adj[:-2],
            data_para.price_close_adj[:-2],
            data_para.price_vwap_adj[:-2],
            data_para.volume_adj[:-2],
            data_para.stock_return[:-2],
            gp_result.selected_fac,
            Barra,
            data_para.real_stock_return,
            toolbox,
            pset)
    '''
    start_time = time.perf_counter()
    fitnesses = []
    for ind in pop:
        result = evalSymbReg(ind,start_index, 
                            data_para.price_high_adj[start_index:-2],
                            data_para.price_open_adj[start_index:-2],
                            data_para.price_low_adj[start_index:-2],
                            data_para.price_close_adj[start_index:-2],
                            data_para.price_vwap_adj[start_index:-2],
                            data_para.volume_adj[start_index:-2],
                            data_para.stock_return[start_index:-2],
                            gp_result.selected_fac,
                            Barra,
                            data_para.real_stock_return,
                            toolbox,
                            pset)
        
        #print (random_seed,":,numebr:",number )
        #如果有进程找到有用的因子则停止
        try:
            result_q = q.get(timeout = 5.)
            result_q["number"] = result_q["number"] + 1
            q.put(result_q,timeout = 5.)
            if type(result_q["find"]) == tuple:
                print ("单次进化结束1，次序：",order)
                return 
        except:
            print ("无法获取queue")
            print ("单次进化结束5，次序：",order)
            return 
        
        if result[0] > gp_result.threshold_2year:
            result_4year = evalSymbReg(ind, 0, 
                                        data_para.price_high_adj[:-2],
                                        data_para.price_open_adj[:-2],
                                        data_para.price_low_adj[:-2],
                                        data_para.price_close_adj[:-2],
                                        data_para.price_vwap_adj[:-2],
                                        data_para.volume_adj[:-2],
                                        data_para.stock_return[:-2],
                                        gp_result.selected_fac,
                                        Barra,
                                        data_para.real_stock_return,
                                        toolbox,
                                        pset)
            ret_4year = evalSymbReg(ind, 0, 
                                    data_para.price_high_adj[:-2],
                                    data_para.price_open_adj[:-2],
                                    data_para.price_low_adj[:-2],
                                    data_para.price_close_adj[:-2],
                                    data_para.price_vwap_adj[:-2],
                                    data_para.volume_adj[:-2],
                                    data_para.stock_return[:-2],
                                    gp_result.selected_fac,
                                    Barra,
                                    data_para.real_stock_return,
                                    toolbox,
                                    pset,
                                    method = 1)[2]
            try:
                result_q = q.get(timeout = 5.)
                if result_4year[0] > gp_result.threshold_4year:
                    print ("发现了一个有用的表达式：",str(ind))
                    #保存了表达式原值，IR_2year,IR_4year,new_fac_4year,ret_list_4year,IC_list_4year
                    result_q["find"] = (ind,result[0],result_4year[0],result_4year[1],ret_4year,result_4year[2])
                    q.put(result_q,timeout = 5.)
                    print ("单次进化结束2，次序：",order)
                    return 
                q.put(result_q,timeout = 5.)
                print ("单次进化结束3，次序：",order)
                return 
            except:
                print ("无法获取queue")
                print ("单次进化结束5，次序：",order)
                return 
        
        fitnesses.append((result[0],))
        
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit       

    #输出每一代的统计结果
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.nanmean)
    mstats.register("std", np.nanstd)
    mstats.register("min", np.nanmin)
    mstats.register("max", np.nanmax)
    record = mstats.compile(pop) if mstats else {}
    logbook = tools.Logbook()
    logbook.header = (mstats.fields if mstats else [])
    logbook.record(**record)
    print ("单次进化进程次序：",order)
    print (logbook)
    end_time = time.perf_counter()
    print (order,":计算父代IR耗时为",end_time - start_time)
    print (order,":计算父代IR单个因子平均耗时", (end_time - start_time)/len(pop))
    for g in range(GEN):
        #start_time = time.perf_counter()
        #选择
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        #交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                
                del child1.fitness.values
                del child2.fitness.values
        
        #变异
        for mutant in offspring:
            if random.random() < MUPB:
                toolbox.mutate(mutant)
                # 防止变异之后树深度大于最大深度
                while mutant.height > gp_result.deep_max:
                    toolbox.mutate(mutant)
                del mutant.fitness.values
        cx_mu_end = time.perf_counter()
        #print (order,":交叉和变异的耗时为：", cx_mu_end - start_time)
        #评估新子代的适应度
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print ("子代数量：", len(invalid_ind))
        fitnesses = []
        for ind in invalid_ind :
            result = evalSymbReg(ind,start_index, 
                                data_para.price_high_adj[start_index:-2],
                                data_para.price_open_adj[start_index:-2],
                                data_para.price_low_adj[start_index:-2],
                                data_para.price_close_adj[start_index:-2],
                                data_para.price_vwap_adj[start_index:-2],
                                data_para.volume_adj[start_index:-2],
                                data_para.stock_return[start_index:-2],
                                gp_result.selected_fac,
                                Barra,
                                data_para.real_stock_return,
                                toolbox,
                                pset)            
            #如果有进程找到有用的因子则停止
            try:
                result_q = q.get(timeout = 5.)
                result_q["number"] = result_q["number"] + 1
                q.put(result_q,timeout = 5.)
                if type(result_q["find"]) == tuple:
                    print ("单次进化结束1，次序：",order)
                    return 
            except:
                print ("无法获取queue")
                print ("单次进化结束5，次序：",order)
                return 
            
            if result[0] > gp_result.threshold_2year:
                result_4year = evalSymbReg(ind, 0, 
                                            data_para.price_high_adj[:-2],
                                            data_para.price_open_adj[:-2],
                                            data_para.price_low_adj[:-2],
                                            data_para.price_close_adj[:-2],
                                            data_para.price_vwap_adj[:-2],
                                            data_para.volume_adj[:-2],
                                            data_para.stock_return[:-2],
                                            gp_result.selected_fac,
                                            Barra,
                                            data_para.real_stock_return,
                                            toolbox,
                                            pset)
                ret_4year = evalSymbReg(ind, 0, 
                                        data_para.price_high_adj[:-2],
                                        data_para.price_open_adj[:-2],
                                        data_para.price_low_adj[:-2],
                                        data_para.price_close_adj[:-2],
                                        data_para.price_vwap_adj[:-2],
                                        data_para.volume_adj[:-2],
                                        data_para.stock_return[:-2],
                                        gp_result.selected_fac,
                                        Barra,
                                        data_para.real_stock_return,
                                        toolbox,
                                        pset,
                                        method = 1)[2]
                try:
                    result_q = q.get(timeout = 5.)
                    if result_4year[0] > gp_result.threshold_4year:
                        print ("发现了一个有用的表达式：",str(ind))
                        #保存了表达式原值，IR_2year,IR_4year,new_fac_4year,ret_list_4year,IC_list_4year
                        result_q["find"] = (ind,result[0],result_4year[0],result_4year[1],ret_4year,result_4year[2])
                        q.put(result_q,timeout = 5.)
                        print ("单次进化结束2，次序：",order)
                        return 
                    q.put(result_q,timeout = 5.)
                    print ("单次进化结束3，次序：",order)
                    return 
                except:
                    print ("无法获取queue")
                    print ("单次进化结束5，次序：",order)
                    return 
                
            fitnesses.append((result[0],))
        
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit       
        
        #更新种群
        pop[:] = offspring
        
        #输出每一代的统计结果
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.nanmean)
        mstats.register("std", np.nanstd)
        mstats.register("min", np.nanmin)
        mstats.register("max", np.nanmax)
        record = mstats.compile(pop) if mstats else {}
        logbook.record(**record)
        print ("单次进化进程次序：",order)
        print (logbook)
        cal_IR_end_time =  time.perf_counter()
        print (order,":计算子代IR耗时为",cal_IR_end_time - cx_mu_end)
        print (order,":计算子代IR单个因子平均耗时", (cal_IR_end_time - cx_mu_end)/len(invalid_ind))
    return 


'''
def func(A,B,C,D,E,F,G,q,order,result_q,tool_box,pset,useless):
    total = 20
    result = np.full(total,np.nan)
    for i in range(total):
        result[i] = GP_operator.Corr(A,B,10).sum()
        a = A*B*C*D*E*F*G
        print ("i",i)
        try:
            n = q.get(timeout = 10)
            #print ("n",n)
        except:
            print ("[WATCHDOG]: Maybe WORKER is slacking")
            #result_dict = result_q.get()
            #result_dict[str(order)] = result
            #result_q.put(result_dict)
            return
        if i <= 1000:
            #print ("f2",n)
            q.put(n + 1)
        else:
            print ("f2_not_put",n)
            #result_dict = result_q.get()
            #result_dict[str(order)] = result
            #result_q.put(result_dict)
            return
    #send_end.send(result)
    result_dict = result_q.get()
    result_dict[str(order)] = result
    result_q.put(result_dict)
    return 

def func_test(A,B):
    result = np.full(30,np.nan)
    for i in range(30):
        result[i] = GP_operator.Corr(A,B,10).sum()
        print ("i",i)
    return result
'''
def f(inds,q,order,start_index,price_high, price_open, price_low, price_close, vwap, volume,ret,selected_fac,Barra,real_stock_return,toolbox,pset,threshold_2year):
    '''
    该函数用来在并行计算许多个表达式的IR
    inds为表达式个体
    q为在进程之间通信的queue变量，用来储存进程结果
    order为当前进程在所有进程中的位置
    '''
    
    IR_2y_result = np.full(len(inds),np.nan)
    for index, ind in enumerate(inds):
        #print ("index:",index)
        try:
            IR_2y_result[index] = evalSymbReg(ind,start_index,price_high, 
                                              price_open, price_low,
                                              price_close, vwap, volume,ret,
                                              selected_fac,Barra,real_stock_return,
                                              toolbox,pset)[0]
        except:
            print ("计算IR_2year出现问题")
            IR_2y_result[index] = 0
        '''
        try:
            n = q.get()
            #print (n)
        except :
            #如果当前在5s内无法获取q值，说明其他进程找到满足阈值因子，结束当前进程
            return 
        if IR_2y_result[index] >= threshold_2year:
            #如果当前进程找到了满足阈值的因子，不重新提取q值之后不放回，将这个因子表达式保存到result中
            print ("找到IR_2year大于阈值的因子")
            result_dict = result.get()
            result_tuple = (ind,IR_2y_result[index])
            result.put(result_tuple)
            return 
        else:
            q.put(n + 1)
        '''
        try:
            res = q.get(timeout = 5.)
            res["number"] = res["number"] + 1
        except:
            print ("读取queue出现问题")
            continue
        if type(res["find"]) == tuple:
            #说明有进程找到了有用的因子，结束所有进程
            q.put(res)
            return
        elif IR_2y_result[index] >= threshold_2year:
            #如果当前进程找到了满足阈值的因子，不重新提取q值之后不放回，将这个因子表达式保存到result中
            print ("找到IR_2year大于阈值的因子")
            res["find"] = (ind,IR_2y_result[index])
            q.put(res)
            return
        else:
            q.put(res)
    #如果所有因子计算完毕，则将IR结果按照进程顺序保存到result的dict中
    try:
        res = q.get(timeout = 5.)    
    except: 
        print ("储存结果到queue出现问题")
        return 
    res[str(order)] = IR_2y_result
    q.put(res)
    return 