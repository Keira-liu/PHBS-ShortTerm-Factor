from multiprocessing import Process, Queue


import numpy as np
import time 
import worker
import multiprocess_func
import os
import pickle
from deap import tools
import random
import pandas as pd


import warnings
warnings.filterwarnings("ignore")

PATH_RAW = "data\\raw"
PATH_INTERM = "data\\interm"
PATH_FINAL = "data\\processed"

'''
shape = (468,3800)
a = np.random.randn(*shape)
b = np.random.randn(*shape)
c = np.random.randn(*shape)
d = np.random.randn(*shape)
e = np.random.randn(*shape)
f = np.random.randn(*shape)
g = np.random.randn(*shape)
n = 10
loop = 90

def f1(A,ii):
    for i in range(30):
        result = GP_operator.Corr(A,10)
        #print (i)
    return result


#Applying the function sequentially
tic = time.perf_counter()
for ii in range(loop):
    result = GP_operator.Corr(a,b,10)
    #print (ii)
tac = time.perf_counter()
print("time for sequential sorting: ", tac-tic)
'''
            
gp_result = worker.gp_result

def pop_IR_multipro(pop,data_para,Barra,njobs = 5):
    '''
    该函数用来并行计算许多表达式的IR
    njobs表示并行的进程数量
    '''
    #生成用来存储结果的Queue变量（进程交互）
    q = Queue()
    result_dict = {"number": 0,
                   "find": 0}
    q.put(result_dict)
    
    start_index = int(len(Barra.date_list)/2)     #最后两期没有下期收益率数据
    High = data_para.price_high_adj[start_index:-2]
    Open = data_para.price_open_adj[start_index:-2] 
    Low = data_para.price_low_adj[start_index:-2]
    Close = data_para.price_close_adj[start_index:-2] 
    Vwap = data_para.price_vwap_adj[start_index:-2]
    Volume = data_para.volume_adj[start_index:-2]
    Return = data_para.stock_return[start_index:-2]
    selected_fac  = gp_result.selected_fac
    real_stock_return = data_para.real_stock_return
    args = (start_index,
            High, #最后两期没有下期收益率数据
            Open, 
            Low, 
            Close, 
            Vwap, 
            Volume,
            Return,
            selected_fac,
            Barra,
            real_stock_return,
            worker.toolbox,worker.pset,
            gp_result.threshold_2year)
    
    #计算每一个进程需要计算的因子表达式数量
    if len(pop) % njobs  == 0:
        single_length = len(pop) // njobs 
    elif len(pop) % (njobs -  1)  == 0:
        single_length = len(pop) // (njobs - 1 )
        njobs = njobs - 1
    else:
        single_length = len(pop) // (njobs - 1 )
        
    #生成进程
    pros = []
    for ii in range(njobs):
        try:
            p = Process(target=worker.f, args=(pop[ii*single_length:(ii+1)*single_length],q,ii,*args))
            p.start()
            pros.append(p)
        except:
            print ("线程紊乱")
            for p in pros:
                p.terminate()
                #p.join()
            q.close() 
            return None

    for p in pros:
        p.join()
        
    output = q.get()
    q.close()  
    
    if type(output["find"]) == int:
        temp = []
        for ii in range(njobs):
            if str(ii) in output.keys():
                temp.append(output[str(ii)])
            else:
                temp.append(np.zeros(single_length))
        fitnesses = np.concatenate(temp).tolist()
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit,
        gp_result.total_fac_n_add(output["number"])
        return False
    else:
        start_index = 0
        args = (start_index,
                data_para.price_high_adj[start_index:-2],
                data_para.price_open_adj[start_index:-2], 
                data_para.price_low_adj[start_index:-2], 
                data_para.price_close_adj[start_index:-2], 
                data_para.price_vwap_adj[start_index:-2], 
                data_para.volume_adj[start_index:-2],
                data_para.stock_return[start_index:-2],
                selected_fac,
                Barra,
                real_stock_return,
                worker.toolbox,worker.pset)
        
        useful_ind = output["find"][0]
        IR_2year = output["find"][1]
        IR_4year,new_fac = worker.evalSymbReg(useful_ind,*args)
        gp_result.total_fac_n_add(output["number"])
        if abs(IR_4year) > gp_result.threshold_4year:
            print ("发现了一个有用的表达式：",str(useful_ind))  
            #保存表达式原始格式
            #gp_result.fac_orgin_add(useful_ind)
            #output = open(os.path.join(PATH_FINAL,"useful_formula_orgin.pkl"), 'wb')
            #pickle.dump(gp_result.useful_formula_orgin,output)
            #output.close()
            #保存表达式string格式以及IR
            time_used = time.perf_counter() - gp_result.each_useful_fac_time
            gp_result.each_useful_fac_time = time.perf_counter()
            gp_result.fac_result_add([str(useful_ind),IR_4year,IR_2year,time_used])
            gp_result.useful_fac_n_add()
            gp_result.useful_formula.to_csv(os.path.join(PATH_FINAL,"formula_result.csv"))
            gp_result.selected_fac_add(new_fac)
            pd.DataFrame(gp_result.selected_fac).to_pickle(os.path.join(PATH_FINAL,"selected_fac.pkl"))
        return True
    
def easimple(data_para,Barra,CXPB = gp_result.CXPB,MUPB = gp_result.MUPB,
             POP_SIZE = 100, GEN = 10,random_seed = 1,toolbox = worker.toolbox):
    '''
    单次遗传过程函数
    CXPB为交叉的概率
    MUPB为变异的概率
    遗传算法主函数
    POP_SIZE为每一代群体的数量
    GEN为多少代
    假设每一代交叉和变异的概率分别为0.1，那么每一代会大约生成pop*0.2个个体
    '''
    random.seed(random_seed)
    #initializing population
    pop = worker.toolbox.population(n = POP_SIZE)
    '''开始进化'''
    print ("单次进化开始，随机种子：",random_seed)
    #evaluating the fitness
    
    result = pop_IR_multipro(pop,data_para,Barra)
    if result is None:
        print ("单次进化结束")
        return False
    elif result == True:
        print ("单次进化结束")
        return True
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
    print (logbook.stream)
    for g in range(GEN):
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
        
        #评估新子代的适应度
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        result = pop_IR_multipro(invalid_ind,data_para,Barra,njobs = 4)
        if result is None:
            print ("单次进化结束")
            return False
        elif result == True:
            print ("单次进化结束")
            return True
        '''
        fitnesses = []
        for ind in invalid_ind :
            result = worker.toolbox.eval_orgin(ind)
            if result[0] > gp_result.threshold_2year:
                print ("单次进化结束")
                return True
            fitnesses.append(result)
            
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit       
        '''
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
        print (logbook.stream)
    print ("单次进化结束")
    return False

def main(data_para,Barra,POP_SIZE = 300, GEN = 10, Times = 2,method = 1,continous = 0):
    '''
    该函数是在并行计算多个表达式情况下的主函数
    data_para,Barra都是计算需要的数据
    POP_SIZE,GEN分别是单次遗传的群体和遗传代数
    Times为总共进行的单次遗传算法的次数
    method表示停止的条件，1表示找到Times个有用的因子后再停止，2表示循环Times次后即停止
    continous 表示是否继续之前的结果进行挖掘，0表示不继续,1表示继续
    '''
    if continous == 1:    
        gp_result.continous()
    start_time = time.perf_counter()
    gp_result.each_thousand_time= time.perf_counter()
    gp_result.each_useful_fac_time = time.perf_counter() 
    if method == 1:
        ii = int(time.perf_counter()) % 1000                     
        #ii = 61
        while gp_result.useful_fac_n < Times:
            easimple(data_para,Barra,POP_SIZE=POP_SIZE,GEN = GEN, random_seed= ii)
            ii = ii + 1
            print ("目前总耗时：", time.perf_counter() - start_time )
    elif method == 2:
        for ii in range(Times):
            easimple(data_para,Barra,POP_SIZE=POP_SIZE,GEN = GEN, random_seed= ii)
    end_time = time.perf_counter()
    print ("总耗时：", end_time - start_time)
    print ("生成因子数：", gp_result.total_fac_n)
    print ("平均耗时(s)：",(end_time - start_time) / gp_result.total_fac_n)

def easimple_main_test(data_para,Barra,toolbox,pset,gp_result,order = 0,
                       POP_SIZE = 100, GEN = 30,njobs = 3):
    '''该函数用来并行计算许多遗传算法过程'''
    q = Queue()
    result_dict = {"number": 0,
                   "find": 0}

    q.put(result_dict)
     #生成进程
    pros = []
    for ii in range(njobs):
        try:
            p = Process(target=worker.easimple_multi_pros, args=(q,data_para,Barra,toolbox,pset,ii+order,gp_result,POP_SIZE,GEN))
            p.start()
            pros.append(p)
        except:
            print ("线程紊乱")
            for p in pros:
                p.terminate()
                #p.join()
            q.close() 
            return None 
        
    #由于再所有进程结束前queue中不能有值，所以需要在njob-1个进程结束之后就提取出queue值
    while True:
        allExited  =  0
        for t in pros:
            if not t.exitcode is None:
                allExited = allExited + 1 
        if allExited == njobs - 1:
            time.sleep(3)
            try:
                output = q.get(timeout = 5.)
            except:
                for p in pros:
                    if p.is_alive():
                        p.terminate()
                print ("无法获取output")
                return None
            break
        time.sleep(3)
    
    for p in pros:
        p.join()
        
    for p in pros:
        if p.is_alive():
            p.terminate()
        
    
    if type(output["find"]) != int:
        useful_ind = output["find"][0]
        IR_2year = output["find"][1]
        IR_4year = output["find"][2]
        new_fac = output["find"][3]
        ret_list = output["find"][4]
        IC_list = output["find"][5]
        time_used = time.perf_counter() - gp_result.each_useful_fac_time
        gp_result.each_useful_fac_time = time.perf_counter()
        gp_result.fac_result_add([str(useful_ind),IR_4year,IR_2year,time_used])
        gp_result.useful_fac_n_add()
        gp_result.useful_formula.to_csv(os.path.join(PATH_FINAL,"formula_result.csv"))
        gp_result.selected_fac_add(new_fac)
        pd.DataFrame(gp_result.selected_fac).to_pickle(os.path.join(PATH_FINAL,"selected_fac.pkl"))
        gp_result.total_fac_n_add(output["number"])
        #保存表达式因子收益率列表和IC列表
        gp_result.save_ret(ret_list,IC_list)
        gp_result.ret_df.to_pickle(os.path.join(PATH_FINAL,"ret_df.pkl"))
        gp_result.IC_df.to_pickle(os.path.join(PATH_FINAL,"IC_df.pkl"))
        return True
    else:
        gp_result.total_fac_n_add(output["number"])
    return False

def main_test(data,Barra,POP_SIZE = 300, GEN = 10, needed_fac = 2,continous = 0,njobs = 3):
    '''
    该函数为在并行计算多个遗传算法过程情况下的主函数
    data_para,Barra都是计算需要的数据
    POP_SIZE,GEN分别是单次遗传的群体和遗传代数
    needed_fac为需要找到的因子数量
    continous 表示是否继续之前的结果进行挖掘，0表示不继续,1表示继续
    '''
    if continous == 1:
        print ("继续之前找到的因子进行挖掘")
        gp_result.continous()
    start_time = time.perf_counter()
    gp_result.each_thousand_time= time.perf_counter()
    gp_result.each_useful_fac_time = time.perf_counter() 
    
    #random_seed = int(time.perf_counter()) % 10000                     
    order  = 0
    #ii = 61
    while gp_result.useful_fac_n < needed_fac:
        easimple_main_test(data_para,Barra,worker.toolbox,worker.pset,gp_result,order,POP_SIZE=POP_SIZE,GEN = GEN,njobs = njobs)
        order = order + njobs
        end_time = time.perf_counter()
        print ("目前总耗时：", time.perf_counter() - start_time )
        print ("目前生成因子数：", gp_result.total_fac_n)
        print ("平均耗时(s)：",(end_time - start_time) / gp_result.total_fac_n)

    
#Using multiprocessing
if __name__ == "__main__":
    Barra = multiprocess_func.Barra_fac(start_date = 20150101,curr_date = 20190101)
    data_para = multiprocess_func.Data(start_date = 20150101,curr_date = 20190101)
    '''
    worker.toolbox.register("eval_orgin",worker.eval_orgin,
                            High = data_para.price_high_adj,
                            Open = data_para.price_open_adj,
                            Low = data_para.price_low_adj,
                            Close = data_para.price_close_adj,
                            Vwap = data_para.price_vwap_adj,
                            Volume = data_para.volume_adj,
                            Return = data_para.stock_return,
                            real_stock_return = data_para.real_stock_return,
                            Barra = Barra,
                            selected_fac = gp_result.selected_fac)
    '''
    gp_result.crate_ret_df(Barra)
    main_test(data_para,Barra,POP_SIZE = 100,GEN = 30,needed_fac = 30,njobs = 2,continous = 1)
    

    '''
    #p = Pool(processes=3)
    q = Queue()
    #q.put(0)
    #生成用来存放结果的queue变量
    result = Queue()
    #result_dict = {"number": 0,
    #               "find": 0}
    #for ii in range(njobs):
    #    result_dict[str(ii)] = None
    #result.put(result_dict)
    q.put(0)

    result_dict = {}
    for ii in range(3):
        result_dict[str(ii)] = None
    result.put(result_dict)
    
    pros = []
    args = (a,b,c,d,e,f,g)
    pop = worker.toolbox.population(n = 100)
    for ii in range(3):
        p = Process(target=worker.func, args=(*args,q,ii,result,worker.toolbox,worker.pset,pop[ii*10:(ii+1)*10]))
        #p = Process(target=worker.func_test, args=(a,b))
        pros.append(p)
    for p in pros:
        p.start()
    for p in pros:
        p.join()
    
    #pool = multiprocessing.Pool(processes = 3)
    #print(pool.map(functools.partial(f2, A = a,B = b,q = q), range(3)))
    
    tac = time.perf_counter()
    #print("time for parallel sorting: ",tac-tic)
    #print (result.get())
    '''
