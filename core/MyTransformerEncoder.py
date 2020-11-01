'''
  Reference : https://github.com/graykode/nlp-tutorial
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

def get_sinusoid_encoding_table(n_position, d_model):
    ##PE(pos,2i) = sin(pos/10000^(2i/dmodel))
    ##PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    ##计算每个位置的token的pos_embedding
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        ##scores就是Q*K(T)/sqrt(d_k)这里Q[1,8,5,64],K(T)[1,8,64,5]所以输出[1,8,5,5]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        # Fills elements of self tensor with value where mask is one.
        ##补0的位置填一个小数防止反向传播梯度为0
        scores.masked_fill_(attn_mask, -1e9) 
        ##scores经过softmax后和V相乘
        ##Softmax(dim=-1)对某一维度的行进行softmax运算
        attn = nn.Softmax(dim=-1)(scores)
        ##attn维度[1,8,5,5],V维度[1,8,5,64],所以context维度[1,8,5,64]
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        ##Q，K，V矩阵d_k = d_v = 64,n_heads=8所以输出维度512
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        ##layerNorm就是将同一层神经元的输入变成均值0方差1
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        ##Q,K,V都是同一个enc_inputs，维度为[batch_size x seq_len x embed_dim],d_model是embed_dim
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        ##对输入数据用W_Q，W_K, W_V矩阵进行变换
        ##假设输入是[1,5,512],W_Q输出还是512所以维度没变依然[1,5,512]
        ##要转化成多头形式，n_heads=8所以view()以后变成[1,5,8,64]，然后转置成[1,8,5,64]用于后面的计算
        # q_s: [batch_size x n_heads x len_q x d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        # k_s: [batch_size x n_heads x len_k x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        # v_s: [batch_size x n_heads x len_k x d_v]  
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        ##unsqueeze()给数据增加一个维度，unsqueeze(1)就是增加第二个维度比如(2,3)->(2,1,3)
        ##unsqueeze(1)将attn_mask维度变化为[1,5,5]->[1,1,5,5]，然后repeat()将第二个维度重复n_heads次
        # attn_mask : [batch_size x n_heads x len_q x len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) 

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        ##计算ScaledDotProductAttention的值，attn用于结果的可视化
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        ##计算完后再转置回原来的维度[1,5,8,64]，然后view()将多头的结果合并变成[1,5,512]
        ##view()操作是对整块内存进行的，所以在view()之前执行contiguous()把tensor变成在内存中连续分布的形式
        # context: [batch_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) 
        output = self.linear(context)
        return self.layer_norm(output + residual)# output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        ##conv1d在kernel_size=1，步长为1，no pading时可以当成MLP使用
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        ##layerNorm就是将同一层神经元的输入变成均值0方差1
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        ##多头注意力加前馈神经网络
        # enc_inputs to same Q,K,V
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        ##nn.Embedding嵌入向量查找表，参数为(词典大小，嵌入向量维度)
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(vocab_size+1, d_model),freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        ##加上位置信息pos_emb
        ##如果一次输入5个token[1,5]，embed成512维的向量，所以batch_size=1，seq_len=5，embed_dim=512
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(enc_inputs)
        ##记录数据中补0的位置，替换成一个小数，以免反向传播时梯度为0
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, target_size, src_len):
        super(TransformerEncoder, self).__init__()
        self.encoder = Encoder(vocab_size)
        self.projection = nn.Linear(src_len*d_model, target_size, bias=False)
    def forward(self, enc_inputs):
        ##输入为[BatchSize,SeqLen]如[1,5]
        # print(f'enc_inputs.shape:{enc_inputs.shape}')
        enc_outputs = self.encoder(enc_inputs)
        ##encoder输出维度[BatchSize,SeqLen,EmbedDim]如[1,5,512]
        ##经过nn.Linear()线性变换成想要输出的维度
        # print("enc_outputs.shape:",enc_outputs.shape)
        enc_logits = self.projection(enc_outputs.contiguous().view(enc_outputs.size(0),-1))
        return enc_logits


EmbeddingSize = 512
d_model = EmbeddingSize  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


if __name__ == '__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    labels = ['ge','en','en']


    ##文本特征化为数字，label直接转化为数字
    ##nn.CrossEntropyLoss()的输入为一个二维张量和一个一维张量
    def make_batch(sentences,labels):
        assert(len(sentences)==len(labels))
        input_batch = [[word2id[t] for t in n.split()] for n in sentences]
        output_batch = [label2id[i] for i in labels]
        return torch.LongTensor(input_batch),torch.LongTensor(output_batch)

    ##Global things
    word2id = None
    label2id = None

    ##parameters
    LR=0.001
    EPOCHS=20
    MAX_LEN = 5
    src_len = MAX_LEN # length of source
    tgt_len = 5 # length of target

    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    model = None

    def train(sentences, labels):
        global word2id
        vocab_size = None
        global label2id
        target_size = None
        global model
        vocab = set((' '.join(sentences)).split())
        vocab = sorted(vocab)
        word2id = {w:i for i,w in enumerate(vocab)}
        vocab_size = len(word2id)

        label2id = {v:i for i,v in enumerate(sorted(set(labels)))}
        target_size = len(label2id)

        model = TransformerEncoder(vocab_size, target_size, src_len)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        enc_inputs,output_batch = make_batch(sentences,labels)
        enc_inputs=enc_inputs.to(device)
        output_batch=output_batch.to(device)
        print(f'inputs:{enc_inputs.shape}',f'outputs:{output_batch.shape}')
        print('device:',enc_inputs.device)

        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(enc_inputs)
            loss = criterion(outputs, output_batch)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            loss.backward()
            optimizer.step()
            if loss<0.1:
                print('loss<0.1,break')
                break

    def predict(data_inputs):
        ##test时一定要model.eval()
        ##model.train()启用 BatchNormalization 和 Dropout
        ##model.eval()不启用 BatchNormalization 和 Dropout
        input_batch = [[word2id[t] for t in n.split()] for n in data_inputs]
        id2label = {v:k for k,v in label2id.items()}

        if not os.path.exists('./model/torch_model.pkl'):
            print('model not exists')

        ## Test
        model.eval()
        predict = model(torch.LongTensor(input_batch))
        confidence = predict.softmax(1)
        # print(confidence)
        predict = predict.data.max(1, keepdim=True)[1]

        if len(data_inputs) == 1:
            # print(data_inputs,'->',id2label[predict.squeeze().item()])
            return {'label':id2label[predict.squeeze().item()],'confidence':confidence.max().item()}
        else:
            # print(data_inputs, '->', [id2label[i.item()] for i in predict.squeeze()])
            return [{'label':id2label[v.item()],'confidence':confidence[i].max().item()} for i,v in enumerate(predict.squeeze())]


    train(sentences,labels)
    r=predict(['ich mochte ein bier P'])
    # r=predict(sentences)
    print(r)
