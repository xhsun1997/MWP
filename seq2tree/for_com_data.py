# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import os,json,random
from tqdm import tqdm
import jieba,re


batch_size = 128
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 2e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

def removeSurplusPeriod(question):  # 去除多余的英文句号
    # 需要移除题号

    for i in range(len(question) - 2):
        if question[i] >= '0' and question[i] <= '9':
            if question[i + 1] == '.':
                if question[i + 2] < '0' or question[i + 2] > '9':
                    question = question.replace(question[i + 1], "", 1)
                    return question
    return question


################################################################################################################
def get_correct_question(wrong_question):
    wrong_example=re.findall("(的\d+，)",wrong_question)#找到问题中的缺少分号的数字
    wrong_example=wrong_example[0]
    assert len(wrong_example)==4
    true_example=wrong_example[0]+wrong_example[1]+"/"+wrong_example[2]+wrong_example[3]
    true_question=wrong_question.replace(wrong_example,true_example)
    return true_question

def replace_chinese_digit(question):
    chinese_digits={'０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9'}
    temp_question=list(question)
    flag=False
    for i in range(1,len(temp_question)-1):
        if temp_question[i] in chinese_digits:
            if temp_question[i-1] in chinese_digits and temp_question[i+1] in chinese_digits:
                a=chinese_digits[temp_question[i-1]]
                b=chinese_digits[temp_question[i]]
                c=chinese_digits[temp_question[i+1]]
                temp_question[i-1:i+1+1]=a+b+c
            elif temp_question[i-1] in chinese_digits:
                a=chinese_digits[temp_question[i-1]]
                b=chinese_digits[temp_question[i]]
                temp_question[i-1:i+1]=a+b
            elif temp_question[i+1] in chinese_digits:
                a=chinese_digits[temp_question[i+1]]
                b=chinese_digits[temp_question[i]]
                temp_question[i:i+1+1]=b+a
            else:
                temp_question[i]=chinese_digits[temp_question[i]]
            flag=True
    processed_chinese_digit_question=''.join(temp_question)
    if flag:
        print("这个问题中出现了中文字符编码的数字 : ",question)
        print("将中文字符编码转化为英文编码后的数字　: ",processed_chinese_digit_question)
    return processed_chinese_digit_question
    

def processQuestion(question):  # question预处理
    question = re.sub('(\d+)_(\d+/\d+)', '(\\1+\\2)', question)  # /d+ 数字
    question = question.replace("l", "1")  # 替换掉l
    question = question.replace("m2", "平方米")  # 替换m2
    question = question.replace("做多分", "最多分")  # 替换错误词语
    question = question.replace("百分之？", "百分之几？")
    question = question.replace("铅笔08元", "铅笔0.8元")
    question = replace_chinese_digit(question)

    if len(re.findall("(的\d+，)",question)):
        print("这些问题中的数字实际上是缺少分号的: ",question)
        question=get_correct_question(question)
        print("纠正后的问题文本如下: ",question)

    if question.split('、')[0].isdigit():
        # print("这些问题的开头位置中出现了无用的数字编号: ", question)
        question = question.split('、')[1]
        # print("去掉无用的数字编号后的问题: ", question)
    if "解答)" in question[-5:]:
        # 一列火车从甲地开往乙地，已经行了全程的2/5，行了245千米，甲乙两地之间的距离是多少千米?(用方程解答)
        # 去掉问题中的(用xx解答)字样，这些字符对理解问题无用
        re_question = re.findall("(\(.*?\))", question)
        assert "解答)" in re_question[-1]
        question = question[:-len(re_question[-1])]  # 去掉(xxx解答)字样
    if len(re.findall("(\(\d\))", question)) > 0:
        # 匹配问题中出现(数字)
        alabo_digit_dict = {"0": '零', '1': 'x', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': "八",
                            '9': '九'}
        digit_buckets = re.findall("(\(\d\))", question)
        for digit_bucket in digit_buckets:
            # 对于问题中出现的每一个(数字)，将(数字)替换为对应的汉字，如(2):二,(3):三
            # 这里我将(1)替换为x，因为问题中经常会出现一共、一起等字符，所以没有将(1)替换为一
            assert len(digit_bucket) == 3
            digit = digit_bucket[1]
            chinese_char = alabo_digit_dict[digit]
            question = question.replace(digit_bucket, chinese_char)

    question = removeSurplusPeriod(question)
    return question



def get_correct_question(wrong_question):
    wrong_example=re.findall("(的\d+，)",wrong_question)#找到问题中的缺少分号的数字
    wrong_example=wrong_example[0]
    #assert len(wrong_example)==4
    true_example=wrong_example[0]+wrong_example[1]+"/"+wrong_example[2]+wrong_example[3]
    true_question=wrong_question.replace(wrong_example,true_example)
    return true_question

def correct_train_csv(train_csv,train2_csv,train3_csv,written_train_csv):
    def get_data_dict(train_csv):
        train_data_dict={}
        with open(train_csv) as f:
            lines=f.readlines() 
            for line in lines:
                line=line.strip()
                line=line.replace('\n','').replace('"','')
                line_split=line.split(',')
                if len(line_split)!=3:
                    assert len(line_split)>3
                    #我已经把数据集中一句话变成两行的这种问题手工调整了，所以line_split一定是大于等于3的
                    #大于3是因为问题中出现了英文的,
                    new_line_split=[]
                    temp_question="，".join(line_split[1:-1])
                    new_line_split.append(line_split[0])
                    new_line_split.append(temp_question)
                    new_line_split.append(line_split[-1])
                    line_split=new_line_split
                assert len(line_split)==3
                train_data_dict[line_split[0]]=line_split
        return train_data_dict

    def correct_data(train_data_dict,correct_data_dict,correct_data_dict2):
        train_data=[]
        for id_,example in train_data_dict.items():
            assert id_==example[0]
            if id_ in correct_data_dict:
                example=correct_data_dict[id_]
            if id_ in correct_data_dict2:
                example=correct_data_dict2[id_]
            train_data.append(example)
        return train_data

    train_data_dict=get_data_dict(train_csv)
    correct_data_dict=get_data_dict(train2_csv)
    correct_data_dict2=get_data_dict(train3_csv)
    #train2_csv和train3_csv是用来纠正错误的
    train_data=correct_data(train_data_dict,correct_data_dict,correct_data_dict2)
    
    with open(written_train_csv,'w') as f:
        for example in train_data:
            assert len(example)==3
            question=processQuestion(question=example[1])
            segmented_list=list(jieba.cut(question))
            for i,token in enumerate(segmented_list):
                #temp_list.append(token)
                if i>1 and i<len(segmented_list)-1 and (segmented_list[i-1].isdigit() and segmented_list[i]=='/' and segmented_list[i+1].isdigit()):
                    segmented_list[i-1:i+2]=['('+''.join(segmented_list[i-1:i+2])+')']
                    continue
            #jieba分词会导致1/8变成1 / 8
            segmented_text=' '.join(segmented_list)
            example_dict={'id':example[0],'original_text':question,'segmented_text':segmented_text,
                            'equation':'x=(11-1)*2','ans':example[2]}
            f.write(json.dumps(example_dict,ensure_ascii=False)+"\n")
    print("问题已经被纠正了，并且将问题文本进行了jieba分词，同时所有的问题都生成一个伪equation，train.csv在%s"%written_train_csv)

def get_data(correct_train_csv):
    com_train_data=[]
    with open(correct_train_csv) as f:
        lines=f.readlines()
        for line in lines:
            com_train_data.append(json.loads(line.strip()))
    return com_train_data


def transfer_num_for_com_data(data):
    print("Transfer numbers for com_data...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        temp_seg = d["segmented_text"].strip().split(" ")
        seg=[]
        for token in temp_seg:
            if token!='':
                seg.append(token)
        assert '' not in seg
        #equations = d["equation"][2:]#d["equation"]-->x=...

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        #out_seq = seg_and_tag(equations)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        out_seq=['N0','+','N1']#所有问题的expresson统一用这个替代
        pairs.append((input_seq,out_seq,nums, num_pos))

    return pairs, copy_nums


if __name__=="__main__":
    train_csv="./data/train.csv"
    train2_csv="./data/train2.csv"
    train3_csv="./data/train3.csv"
    written_train_csv="./data/correct_train.csv"
    correct_train_csv(train_csv,train2_csv,train3_csv,written_train_csv)
