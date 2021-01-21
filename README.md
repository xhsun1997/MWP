# MWP


#### math23k_data中的数据是根据split_train_test.py划分的
将原来的math23k_data随机打乱，90%作为训练集，10%作为测试集合。**划分后的数据集将作为日后所有实验的数据集**

##### 利用GTS的源码在测试数据集上的准确率分别是: 0.6158和0.7251
##### 利用GTS的源码在测试数据集上(如果generate_tree_input中+generate_nums)的准确率分别是: 0.6167和0.7246
**也就是说在train_and_evaluate.py中的generate_tree_input函数中，num_start+generate_nums+num居然和num_start+num的结果差不多**

##### +generate_nums+group_attention之后的准确率是0.6145和0.7302(几乎没有提升)

