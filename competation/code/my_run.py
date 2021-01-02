from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import os,json,random
from tqdm import tqdm


batch_size = 128
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 2e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

#data = load_raw_data("data/Math_23K.json")

############################################定义math23k的训练数据和测试数据###############################################
from my_need_process import load_json_data,prepare_train_data,prepare_test_data,get_math23_test_pairs
train_data=load_json_data("./data/math23_train.json")
test_data=load_json_data("./data/math23_test.json")

#把问题中的数字替换为NUM，并且将表达式的数字替换为Ni，同时得到num和num_pos

train_data_after_NUM, generate_nums, copy_nums = transfer_num(train_data)
print("在表达式中出现，但是没有在问题中出现的数字定义为常数，它们是 : ",generate_nums)
print("所有问题中出现数字次数最多时，一共出现了%d个数字"%copy_nums)

print('-'*100)

random_idx=random.randint(a=0,b=len(train_data)-1)
print("没有进行NUM转前的原始样本的输入形式 : ",train_data[random_idx])
#把问题中的数字替换为NUM，并且将表达式的数字替换为Ni，同时得到num和num_pos
print("经过NUM转换以后，同时将表达式中的数字替换为Ni，此时样本形式 : ",train_data_after_NUM[random_idx])

temp_pairs = []
for p in train_data_after_NUM:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_trained = temp_pairs
print("经过前缀转换以后的样本形式 : ",pairs_trained[random_idx])

pairs_tested=get_math23_test_pairs(test_data)

input_lang, output_lang, train_pairs= prepare_train_data(pairs_trained, trim_min_count=5, generate_nums=generate_nums,
                                                                copy_nums=copy_nums, tree=True)
print("prepare_train_data的目的就是sentence to id ，此时样本形式 : ",train_pairs[random_idx])

test_pairs=prepare_test_data(pairs_tested,input_lang,output_lang,tree=True)
test_random_idx=random.randint(a=0,b=len(test_pairs)-1)
print('-'*100)
print("输入数据的样本形式为 : ",test_data[test_random_idx])
print("经过transfer_num以及前缀转换后的结果 : ",pairs_tested[test_random_idx])
print("sentence to id后的结果 : ",test_pairs[test_random_idx])
print('-'*100)



def prepare_com_train_data(pairs_tested,input_lang,output_lang,tree=True):
    test_pairs=[]
    for pair in pairs_tested:
        #pair的第二个值是N0+N1
        assert len(pair)==4
        num_stack = []
        for word in pair[1]:
            #pair[0]是问题文本
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3], num_stack))
        if num_stack!=[]:
            print("比赛的数据集中也出现了重复的数字 : ",pair)
    print('比赛训练集合的样本的个数 %d' % (len(test_pairs)))
    return test_pairs

##########################################获取比赛的训练数据集######################################################################
print("##########################################获取比赛的训练数据集#################################################################")
from for_com_data import get_data,transfer_num_for_com_data

com_train_data=get_data(correct_train_csv="./data/correct_train.csv")

com_pairs,com_copy_nums=transfer_num_for_com_data(com_train_data)
assert com_copy_nums<=copy_nums
#必须要保证copy_nums要小于在math23k上的copy_nums，否则output_lang.word2index就会报错
com_train_pairs=prepare_com_train_data(com_pairs,input_lang,output_lang,tree=True)
length_of_com_train_data=len(com_pairs)
print("比赛的训练集中有%d个问题"%length_of_com_train_data)
print("随机的打印两个输入数据的样例")
for _ in range(2):
    n1=random.randint(a=0,b=length_of_com_train_data-1)
    print("经过NUM取代前的原始输入 : ",com_train_data[n1])
    print("经过NUM取代输入问题中的数字，并且记录下数字及其位置后的输入数据形式 : ",com_pairs[n1])
    print("经过sentence2id之后的比赛训练集合的样本如下: ",com_train_pairs[n1])
    print('-'*100)

#############################################定义模型的组成成分############################################################
encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

#encoder.load_state_dict(torch.load("./saved_models/encoder"))
#predict.load_state_dict(torch.load("./saved_models/predict"))
#generate.load_state_dict(torch.load("./saved_models/generate"))
#merge.load_state_dict(torch.load("./saved_models/merge"))
###############################################################################
encoder.load_state_dict(torch.load("./saved_models/encoder",map_location='cpu'))
predict.load_state_dict(torch.load("./saved_models/predict",map_location='cpu'))
generate.load_state_dict(torch.load("./saved_models/generate",map_location='cpu'))
merge.load_state_dict(torch.load("./saved_models/merge",map_location='cpu'))
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
########################################################################################################################
#if USE_CUDA:
#    encoder.cuda()
#    predict.cuda()
#    generate.cuda()
#    merge.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

print("这些常数对应的在decoder端的output_lang.word2index中的id : ",generate_num_ids)
print("output_lang.word2index : ",output_lang.word2index)
#############################################在math23k上测试一下模型，看看准确率################################
value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()
for test_batch in tqdm(test_pairs[:500]):
    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                             merge, output_lang, test_batch[5], beam_size=beam_size)
    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if val_ac:
        value_ac += 1
    if equ_ac:
        equation_ac += 1
    eval_total += 1
print(equation_ac, value_ac, eval_total)
print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
print("testing time", time_since(time.time() - start))
print("------------------------------------------------------")
#####################################################预测出来比赛的训练数据中的equation#########################################
import pickle

com_train_data_expression=[]
bad_question_nums=[]

def get_com_train_expression(com_train_pairs):
    for i,com_train_example in tqdm(enumerate(com_train_pairs)):
        #com_train_example的长度是7，第一个值是input_seq_id，第二个值是序列长度
        #第三个值是虚拟的表达式id，第四个值是表达式的长度
        #第五个值是nums，第六个值是num_pos，第七个值是num_stack
        if i%1000==0:
            print("当前的样本 : ",com_train_data[i])
            print("将问题中的数字替换为NUM : ",com_pairs[i])
            print("这个问题对应的输入数据的样式 : ",com_train_example)
        try:
            res=evaluate_tree(com_train_example[0],com_train_example[1],generate_num_ids,encoder,predict,generate,
                                merge,output_lang,com_train_example[5],beam_size=beam_size)
        except:
            print("出现了错误")
            print("当前错误的样本 : ",com_train_data[i])
            print("错误样本问题中的数字替换为NUM : ",com_pairs[i])
            print("错误问题对应的输入数据的样式 : ",com_train_example)
            bad_question_nums.append(i)
            res=[]
        assert type(res)==list
        #res就是预测的表达式中每一个token对应的id
        prefix_expression_list=out_expression_list(res,output_lang,num_list=com_train_example[4],num_stack=com_train_example[6])
        com_train_data_expression.append(prefix_expression_list)

get_com_train_expression(com_train_pairs)

print("出现错误的问题有%d个"%len(bad_question_nums))
print("这些错误问题对应的id是 : ",bad_question_nums)

with open("./com_train_data_expression.pkl","wb") as f:
    pickle.dump(com_train_data_expression,f)






