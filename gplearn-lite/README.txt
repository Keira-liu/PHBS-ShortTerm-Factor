1.测试程序为test.py
2.X.npy为输入因子，y_5.npy为个股5天后收益率，y_10.npy为个股10天后收益率，y_20.npy为个股20天后收益率。它们在test.py中被读入程序
3.neutral_fac20.mat为中性化因子,在fitness.py中被读入程序
4.修改了pandas包中的frame.py文件（路径：Anaconda3\Lib\site-packages\pandas\core\frame.py），增加了covwith函数（计算协方差）。修改时请先备份原来的frame.py文件
5.修改了numpy包中的nanfunctions.py文件（路径：Anaconda3\Lib\site-packages\numpy\lib\nanfunctions.py），修改了nanargmax和nanargmin函数。修改时请先备份原来的nanfunctions.py文件
6.pandas 0.24.0 计算相关系数的函数corrwith非常慢,需要卸载后安装更低版本的pandas（如0.20.1）