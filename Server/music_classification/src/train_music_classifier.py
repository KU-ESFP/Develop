import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from konlpy.tag import Kkma
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

from Server.music_classification.src.models.model import GRU

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 20


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

kkma = Kkma()

'''
    [데이터 전처리 02]
    불용어 처리 (custom_tokenizer)
     - 영어 및 특수기호 등 제거
     - 즉, 한글만 tokenize
'''
def custom_tokenizer(text):
    # print('text: ', text)             # Shit 말을 마
    tag_word = kkma.pos(text)
    # print('tag_word: ', tag_word)     # [('Shit', 'OL'), ('말', 'NNG'), ('을', 'JX'), ('마', 'NNG')]

    # 불용어 처리
    stopwords = []
    for word, tag in tag_word:
        if tag in ['SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'UN', 'OL', 'OH', 'ON', 'NR', 'NNA']:
            stopwords.append(word)

    # 태깅한 데이터들을 담은 변수
    tmp_word = [word for word, tag in tag_word if not word in stopwords]
    # print('tagging data', tmp_word)     # ['말', '을', '마']
    return tmp_word


''' 
    [Field 정의]
    데이터셋이 순차적인 데이터셋임을 알 수 있도록 sequential 인자값으로 True를 명시
    레이블은 단순한 클래스를 나타내는 숫자로 순차적인 데이터가 아니므로 False를 명시
    batch_first는 신경망에 입력되는 텐서의 첫번째 차원값이 BATCH_SIZE 가 되도록 함 

    sequential : 시퀀스 데이터 여부. (True가 기본값)
    use_vocab : 단어 집합을 만들 것인지 여부. (True가 기본값)
    tokenize : 어떤 토큰화 함수를 사용할 것인지 지정. (string.split이 기본값)
    lower : 영어 데이터를 전부 소문자화한다. (False가 기본값)
    batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값) 
    is_target : 레이블 데이터 여부. (False가 기본값)
    fix_length : 최대 허용 길이. 이 길이에 맞춰서 패딩 작업(Padding)이 진행.
'''
TEXT = Field(sequential=True, use_vocab=True, tokenize=custom_tokenizer, batch_first=True)
LABEL = Field(sequential=False, use_vocab=False, is_target=True)

# Dataset 만들기
train_data, test_data = TabularDataset.splits(path='../csv_datasets/', train='train_data.csv', test='test_data.csv',
                                              format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)


print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))
print('필드 구성 확인: {}'.format(train_data.fields.items()))
print(vars(train_data[0]))

# for item in train_data:
#     print('item', item.text)

'''
    [단어 집합(vocabulary) 만들기]
    min_freq는 학습 데이터에서 최소 N번 이상 등장한 단어만을 단어 집합에 추가하겠다는 의미
    이때, 학습 데이터에서 N번 미만으로 등장한 단어는 Unknown이라는 의미에서 '<unk>'라는 토큰으로 대체됨
    min_freq : 단어 집합에 추가 시 단어의 최소 등장 빈도 조건을 추가
    max_size : 단어 집합의 최대 크기를 지정
'''
TEXT.build_vocab(train_data, min_freq=1)        # 단어 집합 생성

vocab_size = len(TEXT.vocab)                    # vocab_size 저장해두기: 17407
n_classes = 4                                   # 감정 개수 4개 (angry, happy, neutral, sad)
print('단어 집합 크기 : {}'.format(vocab_size))
print('클래스 개수 : {}'.format(n_classes))
print(TEXT.vocab.stoi)


'''
    [토치 텍스트의 데이터로더 만들기]
    훈련 데이터와 테스트 데이터는 분리하였지만, 이제 검증 데이터를 분리할 차례
    훈련 데이터를 다시 8:2로 분리하여 검증 데이터를 만들기
    검증 데이터는 val_data 라는 변수에 저장
    
    즉, 훈련 데이터: train_data, 테스트 데이터: test_data, 검증 데이터: val_data
'''
train_data, val_data = train_data.split(split_ratio=0.8)


'''
    토치텍스트는 모든 텍스트를 배치 처리하는 것을 지원하고, 단어를 인덱스 번호로 대체하는 BucketIterator를 제공
    BucketIterator는 batch_size, device, shuffle 등의 인자를 받음
    BATCH_SIZE는 앞서 64로 설정
'''
train_loader, val_loader, test_loader = BucketIterator.splits((train_data, val_data, test_data), batch_size=BATCH_SIZE,
                                                              sort_key=lambda x: len(x.text), sort_within_batch=False)


print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_loader)))
print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_loader)))
print('검증 데이터의 미니 배치의 개수 : {}'.format(len(val_loader)))

# batch = next(iter(train_loader))    # 첫번째 미니배치
# print(batch.text.shape)             # torch.Size([3, 109])
# print(batch.text)


# 모델 설계
model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# # 모델 형태
# for i in range(len(list(model.parameters()))):
#     print(list(model.parameters())[i].size())


# 모델 훈련 함수
def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)

        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()


# 모델 평가 함수
def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)

        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        ''' 
            logit.max(1)
                1. the max value in each row of model (각 행중 최대값을 선택에 차원 1로 만듦)
                2. the column index at which the max value is found (최대값의 위치(index) 반환)
            
            Ex)
                 0.6857  0.1098  0.4687  0.7822
                 0.4170  0.2476  0.1339  0.5563
                 0.9425  0.8433  0.1335  0.3169
                 y.max(1) returns two tensors…
                
                # 1.
                     0.7822
                     0.5563
                     0.9425
                # 2.
                     3
                     3
                     0
        '''
        # print('max data', logit.max(1)[1].view(y.size()).data)
        # print('y data', y.data)
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


# 모델 훈련
best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_loader)
    val_loss, val_accuracy = evaluate(model, val_loader)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("../trained_models"):
            os.makedirs("../trained_models")
        torch.save(model.state_dict(), '../trained_models/txt_classification_02.pth')
        best_val_loss = val_loss

model.load_state_dict(torch.load('../trained_models/txt_classification_02.pth'))
test_loss, test_acc = evaluate(model, test_loader)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))
