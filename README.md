# DLthon 프로젝트 5팀 [조아요]

# 프로젝트 설명
2021 인공지능 그랜드 챌린지 대회를 위한 TUNiB에서 자체적으로 제작한 데이터 셋을  활용하여 "대화의 셩격을 위협 세부 클래스 4개 또는 일반 대화 중 하나로 예측하는 과제"를 진행

# 데이터 설명
* 학습 데이터는 '협박', '갈취', '직장 내 괴롭힘', '기타 괴롭힘' 등 4개 클래스 각 약 1천 개로 구성
* 테스트 데이터는 '협박', '갈취', '직장 내 괴롭힘', '기타 괴롭힘', '일반 대화' 등 5개 클래스 각 1백여 개로 구성. Index와 Conversation만 제공

# 평가 지표
모델이 분류한 결과와 정답 간의 f1 score로 측정

- $F1 Score = 2 * {recall * precision \over recall + precision}$

# 파일 설명
* EDA_DAta.ipynb [EDA 설명](#eda) / [전처리 설명](#데이터-전처리)
* Model_ckeck.ipynb [모델 선택 설명](#모델-선택)
* Train.ipynb [모델 학습 설명](#모델-학습-튜닝)

# EDA
![t](img/train.png)
![struc](img/structure.png)

![Length](img/Length.png)

### 데이터별 워드클라우드
![wor](img/wgal.png)
![wor](img/wgue.png)
![wor](img/whyub.png)
![wor](img/wjig.png)
![norm](img/wnorm.png)

### 클래스 분포
![class](img/classdis.png)

![box](img/box.png)

![l1](img/l1.png)
![l2](img/l2.png)
![l3](img/l3.png)
![l4](img/l4.png)
![l5](img/l5.png)

# 데이터 전처리
## 1. 전처리를 위한 stopword파일을 사용한 불용어 제거와 맞춤법 검사를 진행.
- 띄어쓰기, 맞춤법 [부산대 맞춤법 검사기 활용]
![5BF8E726-3F07-4D87-89D9-C41571321BD3.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/b7b5d266-8faa-4927-a4b1-88386592c2c6/5BF8E726-3F07-4D87-89D9-C41571321BD3.png)
![F3EBC887-3330-4B5C-A98B-0792FB589A64.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/a7156680-7d10-4fe9-9baf-6bcabcb6514c/F3EBC887-3330-4B5C-A98B-0792FB589A64.png)
- 불용어처리
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/60bab0c7-26cf-4011-b217-3f1e2a3001e5/Untitled.png)
- 코드
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/7eba00c5-27e4-4750-949f-37e4cb358eb7)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/27c34d58-46fc-45cc-9654-0b361164b611)
## 2. 변현된 데이터 시각화
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/84ce5b57-78de-4aa3-b68b-b14fd21a682c/Untitled.png)![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/3c705033-30ef-4237-87b7-ac8084f86fd9/Untitled.png)![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/d605aaca-c114-4f31-809e-48d2261623d7/Untitled.png)
## 3. 전처리된 데이터 학습 진행결과
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/45eac45b-bb85-4b09-ab49-786ff5f65562/Untitled.png)
## 4. 전처리 단순화
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/e146b011-c495-4648-be55-c4f2ba46afaa)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/1e7f0701-8b87-412f-acce-2af320b4c492)

## 번역을 이용한 Back Translation
- 코드
papago
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/25d7be5d-ee32-443f-aba2-22aa55bb3bce)
google
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/3c7ab34a-765d-495f-a2f7-a422f5159d4f)

-결과 (적용 X)
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/22e08906-af59-4c36-ad0e-c6d9c6cf6d1f/Untitled.png)
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/851458ce-8c3c-45a0-abdc-fcdfb3c0a56b/Untitled.png)
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/666ae610-a367-41d5-b8b8-5395f61560ff/Untitled.png)

# 모델 선택
## 1. 여러 모델 실험 1
- 시계열 계열의 모델들을 테스트.
- 성능이 그렇게 좋은 폭으로 나오지 않음.
- 오버피팅이 약 2~3에폭에서 발생하는 문제.
### 1. LSTM - 순환 신경망(RNN)의 일종으로, 긴 시퀀스 데이터에서의 의존성을 잘 학습할 수 있는 모델
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/efffa74b-2492-47de-a5b0-6347b7efce49)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/56d684a1-739c-4f28-b4b0-dbca779e2e29)


## 2. 여러 모델 실험 2
- KoELECTRA, Roberta base, Roberta Large 등 다양한 Transformer 계열의 사전 학습 모델을 사용
- 앙상블의 경우 모델의 확률값 epoch 20으로 진행.
- 단, 모델의 무게에 따라 batch_size와 epoch을 조금씩 조절하는 방식.

## 3. 여러 모델 실험 3
### 2. CNN-LSTM -  CNN과 LSTM을 결합한 모델로, 시퀀스 데이터에서 공간적, 시간적 패턴을 동시에 학습 가능 한 모델
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/90a4f77a-ed90-4b73-a020-03e56cf476d5/Untitled.png)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/2167faaa-54d7-446f-ba59-ce631b5cfd3d)

### 3. bidrectional LSTM - 양방향으로 정보를 처리하는 LSTM 모델로, 시퀀스 데이터를 순방향과 역방향으로 동시에 처리하여 더 많은 컨텍스트 정보를 학습 가능한 모델
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/727cd13b-12d9-4efc-a05f-bee7efe5c2f9/Untitled.png)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/2f248b56-a6e9-4c92-83fb-cbf31437062b)

### 4. BERT - 구글이 제안한 모델로, 양방향성과 트렌스포머 아키텍쳐를 기반으로 하며 사전학습과 미세조정이 가능해 텍스트 분류, 질의 응답, 감정 분석 등 다양한 NLP 작업에 사용되는 모델
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/9e400e5b-1e98-4fd5-b99d-bfa5dab5bb7d/Untitled.png)

### 5. BERT Pre-training - BERT를 사전학습시킨 모델
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/fbef97d3-1471-4231-8f74-c0c647e8fba9/Untitled.png)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/891debed-fa88-4e52-bde0-02660b727304)

# 모델 학습 [튜닝]
## 모델 앙상블
- Hard Voting: 각각의 모델들이 결과를 예측하면 단순하게 가장 많은 표를 얻은 결과를 선택
- Soft Voting: 각 class 별로 모델들이 예측한 probability를 합산하여 가장 높은 class를 선택
- Seed Ensemble: train-test-split 과정에서 seed number를 다르게 지정하여 훈련을 통해 나온 모델 기하 평균

# 결과 분석

# 리더 보드

# 실험 Log
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/51a1dad4-818b-4266-bf3c-793e428aaa58/Untitled.png)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/1cdcd862-8db1-4251-9f35-db0d30fed595)
