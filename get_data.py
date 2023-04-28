import math
import numpy as np
import pandas as pd
import random
import re
import torch
import urllib.request
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

#챗봇 데이터 다운로드
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
    filename="ChatBotData.csv"
)
Chatbot_Data = pd.read_csv("ChatBotData.csv")

#일단 테스트용으로 300개만 사용
Chatbot_Data = Chatbot_Data[:300]
Chatbot_Data.head()



