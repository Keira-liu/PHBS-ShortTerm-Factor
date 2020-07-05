# PHBS-ShortTerm-Factor
用GEP算法来实现一个自动挖掘因子的模型，然后再用多因子模型回测。重点关注因子挖掘过程中的三大问题：（1）过拟合；（2）搜索范围太大，效率较低（分层结构+人工过滤条件）（3）提前收敛到局部最优的情况（warm start + replacement method）。


### phase 1:利用geap搭建GEP框架

核心模块:
1.base: 基本结构。包含Toolbox(存储自定的EA运行所需的对象和操作)， Fitness(个体的适应度的基类)等。
2.creator: 允许通过动态添加属性或函数来创建符合问题需求的类，常用来创建个体。
3.tools: 包含多种选择(selection)操作函数，遗传算子操作函数(多种crossover, mutation)等。

还会用到的模块：
algorithms (包含常用进化算法),
gp (需要通过阅读论文构造自己的GEP)
(E:\project\github)

[========]

###ddl(指定日期：July.05/2020):
+ ddl: July.12
    + Anyi & Jizhong: 编写GEP_operatory及相应的documentation（47+）
    + Bolan:处理原始数据存入pkl文件（close,high,open,low,vwap,volume,return,trade_dt）
+ ddl: July.19
    * All:设计class diagram以及sequence diagram
    * Anyi: Sequence diagram
    * Jizhong: class GEP_result(用来储存GEP参数和结果)
    * Bolan: class Data (保存用来输入到表达式中的原始数据)
+ ddl :July. 26
    * All:记录GEP算法与GP算法上区别的细节
    * Anyi：Initialization of Population (设定 primitive set以及族群生成方法)
    * Jizhong: Fitness Evaluation（设计适应度函数）
	* Bolan: Selection（选择方法：锦标赛选择法、轮盘赌选择法等）
+ ddl : Aug.02
    * Bolan:设计mutate方法
	* Anyi & Jizhong:编写Algorithms 计算主程序
