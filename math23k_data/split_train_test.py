import json,os,random
from pprint import pprint
def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data
prefix_path="/home/xhsun/Desktop/MWP/seq2tree/data"
file_name="Math_23K.json"
math23k_file=os.path.join(prefix_path,file_name)


math23k_data=load_raw_data(filename=math23k_file)
total_example_nums=len(math23k_data)
print(total_example_nums)
idx=random.randint(a=0,b=total_example_nums-1)
pprint(math23k_data[idx])


random.shuffle(math23k_data)
train_example_nums=int(total_example_nums*0.9)
test_example_nums=total_example_nums-train_example_nums
train_example=math23k_data[:train_example_nums]
test_example=math23k_data[-test_example_nums:]
print("train example nums : ",len(train_example))
print("test example nums : ",len(test_example))



def write_data(written_file,data):
    with open(written_file,'w') as f:
        for example in data:
            f.write(json.dumps(example,ensure_ascii=False)+"\n")

train_file_name="math23_train.json"
test_file_name="math23_test.json"
write_data(written_file=os.path.join(prefix_path,train_file_name),data=train_example)
write_data(written_file=os.path.join(prefix_path,test_file_name),data=test_example)
