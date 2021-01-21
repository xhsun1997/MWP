import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from copy import deepcopy


#using ．and ，to split sentence

def src_to_mask(src,input_lang):
    src = src.cpu().numpy()
    batch_data_mask_tok = []
    for encode_sen_idx in src:

        token = 1
        mask = [0] * len(encode_sen_idx)
        for num in range(len(encode_sen_idx)):
            mask[num] = token
            if (encode_sen_idx[num] == input_lang.word2index["．"] or encode_sen_idx[num] == input_lang.word2index["，"]) \
                    and num != len(encode_sen_idx) - 1:
                token += 1#表示经过了逗号或者句号之后，变成了另一个数字的上下文了
            if encode_sen_idx[num]==0:mask[num] = 0#pad位置
        for num in range(len(encode_sen_idx)):
            if mask[num] == token and token != 1:
                mask[num] = 1000
        batch_data_mask_tok.append(mask)
    return np.array(batch_data_mask_tok)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    add & norm
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
def group_mask(batch,type="self",pad=0):
    #穿进来的batch就是mask，星行如:
# array([[   1,    1,    1,    1,    1,    1,    1,    1,    2,    2,    2,
#            2,    2,    2, 1000, 1000, 1000, 1000, 1000],
#        [   1,    1,    1,    1,    1,    1,    1,    2,    2,    2,    2,
#            2, 1000, 1000, 1000, 1000, 1000, 1000,    0],
#        [   1,    1,    1,    1,    1,    1,    1,    1,    2,    2,    2,
#            2,    2,    2, 1000, 1000, 1000, 1000,    0],
#        [   1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
#            1, 1000, 1000, 1000,    0,    0,    0,    0],
#        [   1,    1,    1,    1,    1,    1,    2,    2,    2,    2,    2,
#            0,    0,    0,    0,    0,    0,    0,    0]])
#type in {self,between,question}
    length = batch.shape[1]
    lis = []
    if type=="self":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    if ele != 1000:copy[copy == 1000] = 0
                    copy[copy != ele] = 0
                    copy[copy == ele] = 1
                    #print("self copy",copy)
                '''
                if ele == 1000:
                    copy[copy != ele] = 1
                    copy[copy == ele] = 0
                '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif type=="between":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy==1000] = 0
                    copy[copy ==ele] = 0
                    copy[copy!= 0] = 1
                    '''
                    copy[copy != ele and copy != 1000] = 1
                    copy[copy == ele or copy == 1000] = 0
                    '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif type == "question":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy != 1000] = 0
                    copy[copy == 1000] = 1
                if ele==1000:
                	copy[copy==0] = -1
                	copy[copy==1] = 0
                	copy[copy==-1] = 1
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    else:return "error"
    return res

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    '''
    mask.size()==(batch_size,num_heads,seq_len,seq_len)
    query.size()==key.size()==value.size()==(batch_size,num_heads,seq_len,per_size)
    '''
    d_k = query.size(-1)#64
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             /math.sqrt(d_k)
    #scores.size==mask.size
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    #也就是说，有8个头，我们希望的是这8个头可以学习不同的语义
    #以前四个头为例子,第一个头对应的mask是quantity-related mask,那么第一个头学习的attention就是quantity-related attention
    #第二个头对应的mask是global mask,那么此时它和原始的自注意力是一模一样的
    #第三个头对应的question-related mask,第四个头就应该对应的是quantity-between mask
    #后四个头和前四个头是一样的
    #更形象一点就是，每一个头的size()==(seq_len,seq_len)对应的mask同样是(seq_len,seq_len)
    #然后利用masked_fill盖住相应的位置，自然每一个头做的注意力是不一样的，有的是自注意力，有的是互注意力
    #对于原始点transformer来讲，假设有4个头,那么(4,seq_len,seq_len)的mask是一样的，都是全1的tensor
    #然后做自注意力，此时是都可以看到的
    p_attn = F.softmax(scores, dim=-1)#(batch_size,8,seq_len,seq_len)#由于盖住的位置是-1e9，所以softmax后就是0
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn#(batch_size,8,seq_len,per_size)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class GroupAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(GroupAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def get_mask(self,src,input_lang,pad=0):
        #src行如：
# tensor([[ 8360, 10513,  3230,  2180,  1454, 10636,    91, 14766,  8886,  4800,
#           4427,    92,  1411, 14766, 12683,  2295,  4461,  1411, 14770],
#         [ 5222,   455,  7864, 10507,    91,  2025, 14766,  8013, 10507,    92,
#           2025, 14767,  8013,  8889,  7864, 12033, 10610, 14770,     0],
#         [ 2093, 10478,  3099, 10290, 10477,    91, 11317, 14766,  8886,  1182,
#           4359,    92, 11317, 14766,  4359,  1274, 10610, 14770,     0],
#         [ 1504,    91, 10636,  1925, 13672,  2561,    92,    13,    93, 10636,
#           3940, 14766,  5794,    14, 14770,     0,     0,     0,     0],
#         [  714,  4066, 13627,  7760,    91, 14766,  4959, 10636, 11000,    14,
#          14767,     0,     0,     0,     0,     0,     0,     0,     0]])
        mask = src_to_mask(src,input_lang)#这里面data_loader.vocab_dict["．"]=14767
        #data_loader.vocab_dict["，"]=14766
        #所以src_to_mask就是将
        #mask行如:
# array([[   1,    1,    1,    1,    1,    1,    1,    1,    2,    2,    2,
#            2,    2,    2, 1000, 1000, 1000, 1000, 1000],
#        [   1,    1,    1,    1,    1,    1,    1,    2,    2,    2,    2,
#            2, 1000, 1000, 1000, 1000, 1000, 1000,    0],
#        [   1,    1,    1,    1,    1,    1,    1,    1,    2,    2,    2,
#            2,    2,    2, 1000, 1000, 1000, 1000,    0],
#        [   1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
#            1, 1000, 1000, 1000,    0,    0,    0,    0],
#        [   1,    1,    1,    1,    1,    1,    2,    2,    2,    2,    2,
#            0,    0,    0,    0,    0,    0,    0,    0]])
        #1的位置是第一个数字的语义，2的位置是第二个位置的语义，1000表示的是问题的语义，0表示pad
        self.src_mask_self = torch.from_numpy(group_mask(mask,"self",pad).astype('uint8')).unsqueeze(1)
        #self只能看到自己的那块上下文
        self.src_mask_between = torch.from_numpy(group_mask(mask,"between",pad).astype('uint8')).unsqueeze(1)
        self.src_mask_question = torch.from_numpy(group_mask(mask, "question", pad).astype('uint8')).unsqueeze(1)
        self.src_mask_global = (src != pad).unsqueeze(-2).unsqueeze(1)#tensor([[[[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #     True,  True,  True,  True,  True,  True,  True,  True,  True]]],
        # [[[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #     True,  True,  True,  True,  True,  True,  True,  True, False]]],
        # [[[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #     True,  True,  True,  True,  True,  True,  True,  True, False]]],
        # [[[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #     True,  True,  True,  True,  True, False, False, False, False]]],
        # [[[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #     True, False, False, False, False, False, False, False, False]]]])pad的位置是False
        #src_mask_self,between and question.size()==(batch_size,1,max_seq_length_this_batch,max_seq_length_this_batch)
        self.src_mask_global = self.src_mask_global.expand(self.src_mask_self.shape)
        #print(self.src_mask_between.long().cuda(),self.src_mask_self.long().cuda(),self.src_mask_global.long().cuda(),self.src_mask_question.long().cuda())
        self.final = torch.cat((self.src_mask_between.long().cuda(),self.src_mask_self.long().cuda(),self.src_mask_global.long().cuda(),self.src_mask_question.long().cuda()),1)
        return self.final.cuda()
        #inputs_lengths=array([19, 18, 18, 15, 11])
        #final.size()==(batch_size,4,max_seq_length_this_batch,max_seq_length_this_batch)
    def forward(self, query, key, value, mask=None):
        #print("query",query,"\nkey",key,"\nvalue",value)self_attn(x, x, x, mask)
        "Implements Figure 2"
        #query==key==value==output og BiLSTM
        #mask.size()==(batch_size,4,...,...)
        if mask is not None and len(mask.shape)<4:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        else:
            mask = torch.cat((mask, mask), 1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # which is linears(query, key, value)


        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)


        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)#N=1
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        '''Pass the input (and mask) through each layer in turn.
        input_lengths==array([19, 18, 18, 15, 11])的case下
        这里面的x是经过了BiLSTM的output
        mask=src_mask size()==(batch_size,4,max_seq_length_this_batch,max_seq_length_this_batch)
        4指的是:
        (1) global_attention_mask,size()==(batch_size,1,max_seq_length_this_batch,max_seq_length_this_batch)
        global_attention_mask[0]是全1的tensor，因为每一个位置都可以互相看到
        (2) quantity_related attention mask,quantity_related attention means words around quantity usually
        provide beneficial clues，根据逗号分割之后，只能自己看自己的上下文，显然假如有三段，第一段是1，其它两段是0
        都是在对角线上
        (3) quantity_pair attention the relationship between two quantities is of great importance in determinining their associated operator
        这个mask由两部分组成,(1)是自己作为Q，可以看到对方作为K,V，注意此时不能看自己，也不能看问题(attention between quantities)
        		(2)是把问题当做Q，此时Q可以看到所有的数字的上下文K,V，但是不能看自己(attention between quantities and question)
        
        (4) question-related attention，question can derive distinguishing information such as whether the answer is positive
         These are alsi two parts when modeling this type of relation:
         	(1)第i个数字的span作为Q，问题看做K,V
         	(2)问题作为Q，可以看到所有数字的span
        '''
        for layer in self.layers:
            x = layer(x, mask)
            #layer=EncoderLayer(self.d_model,deepcopy(self.group_attention),deepcopy(ff),dropout)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
    	#x.size()==(batch_size,max_seq_length_this_batch,hidden_dim*2)它是BiLSTM的输出
    	#mask.size()==(batch_size,4,max_seq_length_this_batch,...)
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        #sublayer[0]就是一个LayerNorm+Dropout,sublayer[1]也是
        #return x + self.dropout(sublayer(self.norm(x)))，其中sublayer就是group_attention的返回结果
        #x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))这条语句对应的是图一的下半部分
        #也就是将输入+Group_attention(输入)
      	
        return self.sublayer[1](x, self.feed_forward)

class EncoderGroupATT(nn.Module):
    def __init__(self, input_lang,input_size,embedding_size,hidden_size,d_ff=2048,dropout=0.5,n_layers=2,N=1):
        super(EncoderGroupATT, self).__init__()
        self.input_lang=input_lang
        self.d_model = hidden_size##########hidden_size=512,使得BiLSTM的输出变成了1024维度
        self.hidden_size=hidden_size
        #(batch_size,seq_length,1024),然后设置头的数目是8-->(batch_size,8,seq_length,128)
        ff = PositionwiseFeedForward(self.d_model, d_ff, dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.group_attention = GroupAttention(8,self.d_model)
        self.onelayer = Encoder(EncoderLayer(self.d_model,deepcopy(self.group_attention),deepcopy(ff),dropout),N)

    def forward(self, input_var, input_lengths,hidden=None):
        embedded = self.embedding(input_var)
        embedded=self.em_dropout(embedded)
        #print("embedd.size() : ",embedded.size())
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        #embedded.size()=(batch_size,max_seq_length_this_batch,embed_dim)
        pade_outputs, hidden = self.gru_pade(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)
        #print("gru output size(): ",outputs.size())

        problem_outputs =outputs[-1, :, :self.hidden_size] +outputs[0, :, self.hidden_size:]
        outputs=outputs[:,:,:self.hidden_size]+outputs[:,:,self.hidden_size:]

        #print("problem_outputs .size() : ",problem_outputs.size())

        #output.size()==(batch_size,max_seq_length_this_batch,hidden_dim*2)
        #hidden[0].size()==hidden[1].size()==(num_layers*num_derictions,batch_size,hidden_dim)
        #print("input_var.size() : ",input_var.size())
        src_mask = self.group_attention.get_mask(input_var,input_lang=self.input_lang)
        #print("src mask .size() : ",src_mask.size())
        #src_mask.size()==(batch_size,4,max_seq_length_this_batch,max_seq_length_this_batch)
        group_att_outputs = self.onelayer(outputs,src_mask)
        #print("after groupAttention size() : ",outputs.size())
        return group_att_outputs, problem_outputs#output.size()==(batch_size,seq_length,dim)
