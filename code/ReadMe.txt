文件夹内容：
1. 数据集：coil-20, 101_ObjectCategories
2. 代码文件
#dataset : 实现了读取数据集中的数据、存取数据、打乱数据集
#models: 实现了GNMF、PCA、K-means、NMF
#experiment: 实现了模型选择、模型比较、模型比较结果的可视化以及相关结果的保存
3. 运行
实验中采用了5折交叉，比较耗时，可以在第2个和第4个cell中减少ks/regs的长度和times的数值（一对#%%符号之间是一个cell）
或者直接运行第一个和最后一个cell
