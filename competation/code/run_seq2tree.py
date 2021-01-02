# coding: utf-8
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
n_epochs = 60
learning_rate = 5e-4
weight_decay = 1e-5
beam_size = 5
n_layers = 2

#data = load_raw_data("data/Math_23K.json")


def prepare_train_data(pairs_trained,trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
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
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack))
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    return input_lang, output_lang, train_pairs


def prepare_test_data(pairs_tested,input_lang,output_lang,tree=True):
    test_pairs=[]
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
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
    print('Number of testind data %d' % (len(test_pairs)))
    return test_pairs

def special_for_com_data():
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
            seg = d["segmented_text"].strip().split(" ")
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
            pairs.append((input_seq,nums, num_pos))

        return pairs, copy_nums

def get_math23_test_pairs(test_data):
    test_data_after_NUM, generate_nums, copy_nums = transfer_num(test_data)
    #把问题中的数字替换为NUM，并且将表达式的数字替换为Ni，同时得到num和num_pos
    #这里面对generate_nums和copy_nums我们选择不用
    temp_pairs = []
    for p in test_data_after_NUM:
        temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
    pairs_tested=temp_pairs
    print("在math23k的测试集合中，常数数字有 : ",generate_nums)
    print("在math23k的测试集合中，出现数字次数最多时的次数　: ",copy_nums)
    return pairs_tested


def load_json_data(file_):
    data=[]
    with open(file_) as f:
        lines=f.readlines()
        for line in lines:
            example=json.loads(line.strip())
            data.append(example)
    return data

############################################定义math23k的训练数据和测试数据###############################################
train_data=load_json_data("./data/math23_train.json")
test_data=load_json_data("./data/math23_test.json")
random_idx=random.randint(a=0,b=len(train_data)-1)

train_data_after_NUM, generate_nums, copy_nums = transfer_num(train_data)
print("在表达式中出现，但是没有在问题中出现的数字定义为常数，它们是 : ",generate_nums)
print("所有问题中出现数字次数最多时，一共出现了%d个数字"%copy_nums)

print('-'*100)

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
#########################################################################################################################


#############################################定义模型的组成成分############################################################
encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
encoder.load_state_dict(torch.load("./saved_models/encoder"))
predict.load_state_dict(torch.load("./saved_models/predict"))
generate.load_state_dict(torch.load("./saved_models/generate"))
merge.load_state_dict(torch.load("./saved_models/merge"))
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

if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

print("这些常数对应的在decoder端的output_lang.word2index中的id : ",generate_num_ids)
print("output_lang.word2index : ",output_lang.word2index)


for epoch in range(n_epochs):
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()
    for idx in range(len(input_lengths)):
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx])
        loss_total += loss

    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()#在torch 1.0以后，学习率的更新要放在参数更新之后
    if epoch % 10 == 0 or epoch > n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
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
        torch.save(encoder.state_dict(), "saved_models/encoder")
        torch.save(predict.state_dict(), "saved_models/predict")
        torch.save(generate.state_dict(), "saved_models/generate")
        torch.save(merge.state_dict(), "saved_models/merge")

