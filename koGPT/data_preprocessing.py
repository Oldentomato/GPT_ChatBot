import math
import numpy as np
import pandas as pd
import random
import re
import torch
import urllib.request
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
from get_data import Chatbot_Data

#Tokenizer들은 3가지 기능을 제공한다
#1. Tokenizing: 입력 문자열을 token id로 변환(encoding), token id를 다시 문자열로 변환(decoding)의 기능
#2. 기존의 구조(BPE, Sentencepiece 등)에 독립적으로 추가적인 token들을 추가하는 기능
#3. Special Token들을 (mask, BOS, EOS 등) 관리하는 기능


Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

#허깅페이스 transformers 에 등록된 사전 학습된 koGPT2 토크나이저를 가져온다.
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK)
#파라미터 의미
# bos_token : 문장의 시작을 나타내는 token
# eos_token : 문장의 끝을 나타내는 token
# unk_token : 모르는 단어를 나타내는 token
# pad_token : 동일한 batch 내에서 입력의 크기를 동일하게 하기 위해서 사용하는 token

# PreTrainedTokenizerFast 에서 제공되는 함수
# tokenize(): tokenizer를 이용해서 string을 token id의 리스트로 변환한다.
# get_added_vocab(): token to index에 해당하는 dict를 리턴한다.
# batch_decode(): token id로 구성된 입력을 하나의 연결된 string으로 출력한다.
# convert_ids_to_tokens(): token id의 리스트를 token으로 변환한다. skip_special_tokens=True로 하면 decoding할 때 special token들을 제거한다.
# convert_tokens_to_ids(): token string의 리스트를 token id 또는 Token id의 리스트로 변환한다.
# decode(): tokenizer와 vocabulary를 이용해서 token id를 string으로 변환한다. skip_special_token=True로 지정하면 special token들을 제외한다. 
# encode(): token string을 token id의 리스트로 변환한다. add_special_tokens=False로 지정하면 token id로 변환할 때 special token들을 제외한다. 
# padding token을 어떻게 추가할지도 지정할 수 있다.

#챗봇 데이터를 처리하는 클래스를 만든다
class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40): #데이터셋의 전처리를 해주는 부분
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self): #chatbotdata 의 길이를 리턴한다.
        return len(self._data)

    def __getitem__(self, idx): #로드한 챗봇 데이터를 차례차례 DataLoader로 넘겨주는 메서드
        turn = self._data.iloc[idx]
        q = turn["Q"] #질문 행 가져오기
        q = re.sub(r"([?.!,])", r" ", q) #구둣점들을 제거한다. 

        a = turn["A"] #답변을 가져온다.
        a = re.sub(r"([?.!,])", r" ", a) #구둣점들을 제거한다. 

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        #질문의 길이가 최대길이보다 크면
        if q_len > self.max_len:
            a_len = self.max_len - q_len #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0: #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과한다면
                q_toked = q_toked[-(int(self.max_len / 2)) : ] #질문길이를 최대길이의 반으로
                q_len = len(q_toked)
                a_len = self.max_len - q_len # 답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        # 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]
        labels = [self.mask,] * q_len + a_toked[1:]

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # 답변 labels을 index 로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # 질문 + 답변을 index 로 만든다.    
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        #질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)


def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)


train_set = ChatbotDataset(Chatbot_Data, max_len=40)

#윈도우 환경에서 num_workers 는 무조건 0으로 지정, 리눅스에서는 2
train_dataloader = DataLoader(train_set, batch_size=32, num_workers=0, shuffle=True, collate_fn=collate_batch)

#데이터 생성
print("start")
for batch_idx, samples in enumerate(train_dataloader):
    token_ids, mask, label = samples
    print("token_ids ====> ", token_ids)
    print("mask =====> ", mask)
    print("label =====> ", label)
print("end")
