"""
实现cbt
"""

# 导入包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pypinyin import lazy_pinyin

import spacy
import numpy as np

import random
import math
from tqdm import tqdm
import time
import csv

# 参数设置地方
BATCH_SIZE = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化随机参数
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 拿到训练集验证集合测试集，并且得到词典统计
# 1. 定义分词方式
def tokenize_src(text):
    return [tok for tok in text.split()]
def tokenize_trg(text):
    return [tok for tok in text.split()]

# 定义源语言和目标语言的词汇Field
SRC = Field(tokenize = tokenize_src,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            include_lengths = True)

TRG = Field(tokenize = tokenize_trg,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)
p2c_datafields = [
    ('post', SRC),
    ('resp', TRG)
]
train_data, valid_data, test_data = TabularDataset.splits(
    path='data',
    train='train.csv', validation='dev.csv', test='test.csv',
    format='csv',
    skip_header=True,
    fields=p2c_datafields
)
SRC.build_vocab(train_data, max_size=40000)
TRG.build_vocab(train_data, max_size=40000)

# PY_VOCAB = torch.load('checkpoints/src.vocab.data')

# 去除包含UNK的句子

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
     batch_size=BATCH_SIZE,
     sort_within_batch=True,
     sort_key=lambda x:len(x.post),
     device=device)

def load_py():
    csv_file = csv.reader(open('data/train.csv', 'r', encoding='utf-8'))
    pys = set()
    pys.add('pad')
    pys.add('unk')
    for stu in csv_file:
        post = stu[0]
        resp = stu[1]
        for word in post.split():
            pys.add(''.join(lazy_pinyin(word)))
        for word in resp.split():
            pys.add(''.join(lazy_pinyin(word)))

    py_id = {x:y for y,x in enumerate(pys)}
    id_py = {x:y for x,y in enumerate(pys)}

    return py_id, id_py

py_id, id_py = load_py()
PY_UNK_ID = py_id['unk']

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers=3, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src len, batch size]
        # src_len = [src len]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_outputs, hidden = self.rnn(packed_embedded)
        # hidden_size: [num_layers * bidirection, batch_size, hidden]

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer

        # hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        hidden = torch.tanh(self.fc(hidden.view(3, -1, self.enc_hid_dim * 2)))
        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden

# 注意力机制
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        # hidden: [batch_size, hidden_dim]->[batch_size, src_len, hidden_dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)

# 解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, py_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim * 2, dec_hid_dim, num_layers=3)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim + 100, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.py_embedding = nn.Embedding(py_dim, emb_dim)
        self.read_gate = nn.Linear(emb_dim, 100)

    def forward(self, input, hidden, attn_input, encoder_outputs, mask, py):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        py_embedded = self.dropout(self.py_embedding(py))
        py_embedded_gate = F.relu(self.read_gate(py_embedded))

        # embedded = [1, batch size, emb dim]

        a = self.attention(attn_input, encoder_outputs, mask)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted, py_embedded), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden)

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        # assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        py_embedded_gate = py_embedded_gate.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded, py_embedded_gate), dim=1))

        # prediction = [batch size, output dim]

        return prediction, output, hidden, a.squeeze(1)
# 定义Seq2Seq结构
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, py_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.py_pad_idx = py_pad_idx

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        mask = self.create_mask(src)

        # mask = [batch size, src len]
        # 双向lstm的outputs的维度是[seq_len, batch_size, hidden_dim * 2]
        encoder_hidden_dim = encoder_outputs.shape[2]//2
        attn_input = (encoder_outputs[-1:, :, encoder_hidden_dim:] + encoder_outputs[-1:, :, : encoder_hidden_dim]).squeeze(0)

        # 随机取1-2两个位置放入拼音编码
        py_counts = random.randint(1, 2)

        for t in range(1, trg_len):
            # 拿到pinyin
            if t > py_counts:
                py = torch.tensor([[self.py_pad_idx] * batch_size], dtype=torch.long, device=self.device)
            else:

                py = torch.tensor([[py_id.get(''.join(lazy_pinyin(TRG.vocab.itos[x.item()])), PY_UNK_ID)
                                    for x in trg[t]]], dtype=torch.long, device=self.device)

            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, rnn_output, hidden, _ = self.decoder(input, hidden, attn_input, encoder_outputs, mask, py)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            attn_input = rnn_output

        return outputs

# train seq2seq model
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 1024
DEC_HID_DIM = 1024
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
PY_PAD_IDX = py_id['pad']
PY_DIM = len(py_id)

# 实例化注意力机制
attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
# 实例化编码器
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
# 实例化解码器
dec = Decoder(OUTPUT_DIM, PY_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, PY_PAD_IDX, device).to(device)

# 初始化参数
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)

# 控制台输出模型结构
print(model)

# 控制台输出模型所包含的参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip, epoch):
    model.train()

    epoch_loss = 0
    t = 0
    for i, batch in tqdm(enumerate(iterator)):

        src, src_len = batch.post
        trg = batch.resp

        optimizer.zero_grad()

        output = model(src, src_len, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        t += loss.item()
        if (i + 1) % 1000 == 0:
          print('epoch:%d %d%% %.4f' % (epoch, i / len(iterator) * 100, t / 1000))
          t = 0
          if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len = batch.post
            trg = batch.resp

            output = model(src, src_len, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

print('start training')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP, epoch)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'lr: {lr:.6f}')
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    if valid_loss < best_valid_loss:
        # 保存模型文件
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'checkpoints/pcm_embedding.pt')
    else:
      # 改变学习率
      lr /= 2
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr