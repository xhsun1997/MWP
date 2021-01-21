from src.expressions_transfer import *
import os,json,random,re
from src.train_and_evaluate import *
from src.models import *


def transfer_num(data):  # transfer num into "NUM"
    '''
    data是一个list，每一个element是一个dict，keys=="id","original_text","equation","ans","segmented_text"
    '''
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]#d["equation"]-->x=...

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

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    #pairs是一个list，每一个值是一个tuple，tuple的长度是4
    #first element is segmented input sequence(输入序列中的数字已经被NUM取代)
    #second element is output expression sequence(例如: ['(', 'N3', '+', 'N2', ')', '/', '(', '1', '-', 'N1', ')'])
    #third element is 记录的segmented input sequence中所有的数字num
    #forth element is 记录的是所有数字在input sequen中出现的位置num_pos
    return pairs, temp_g, copy_nums

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

