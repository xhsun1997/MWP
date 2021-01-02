import pickle,re
from copy import deepcopy

def compute_prefix_expression_(pre_fix):
    #pre_fic is a prefix list
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        #print(p,st)
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if type(a)==int or type(a)==float:
                a=str(a)
            if type(b)==int or type(b)==float:
                b=str(b)
            #print(a,b)
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None
#compute_prefix_expression(['*', '*', '3.14', '^', '6', '6', '1'])

with open("../competation/com_train_data_expression.pkl","rb") as f:
    res_expression=pickle.load(f)

from pprint import pprint
res_expression=res_expression
pprint(res_expression)


print(len(res_expression))
print([] in res_expression)

import json
def get_data(correct_train_csv):
    com_train_data=[]
    with open(correct_train_csv) as f:
        lines=f.readlines()
        for line in lines:
            com_train_data.append(json.loads(line.strip()))
    return com_train_data
com_train_data=get_data(correct_train_csv="../competation/correct_train.csv")


target_ans=[]
for train_example in com_train_data:
    target_ans.append(train_example['ans'])


eval_nums=0
correct_nums=0
i=0
for each_pression,each_ans in zip(res_expression,target_ans):
    #print(each_pression)
    try:
        if each_ans[-1]=='%':
            ans=float(each_ans[:-1])/100
        else:
            ans=eval(each_ans)
    except:
        print("bad answer",ans)
        continue
    
    assert each_pression!=[]
    for i,temp_token in enumerate(each_pression):
        if temp_token=='08':
            each_pression[i]='0.8'
            print("已经将08替换为0.8")
    predict=compute_prefix_expression_(each_pression)

    if predict==None:
        eval_nums+=1
        continue
        
    eval_nums+=1
    #assert type(ans)==int or type()
    if abs(predict-ans)<1e-2:
        correct_nums+=1

print(correct_nums/eval_nums)