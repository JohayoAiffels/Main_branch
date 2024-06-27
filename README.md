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
* transformer_encoder.ipynb [여러 모델 실험](#transformer_encoder)
* dialog-koelectra.ipynb [여러 모델 실험](#KoELECTRA)
* roberta_model.ipynb [여러 모델 실험](#Roberta Large)
* kobert+koelectra.ipynb [모델 학습 [튜닝]](#모델 앙상블)
  


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
  ![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/530155ec-99bc-4306-a4e2-9bef9fe3688d)
  ![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/baea8a6a-da3f-4353-87e0-2b49f38bee0a)

- 불용어처리
  ![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/e18eb1fe-fadf-4267-aaad-932599beb294)

- 코드
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/7eba00c5-27e4-4750-949f-37e4cb358eb7)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/27c34d58-46fc-45cc-9654-0b361164b611)
## 2. 변형된 데이터 시각화
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/aac868c7-21f3-45e7-ab0e-5615794e8682)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/b83b0fdc-8a75-4766-b958-d2388eeb900e)

## 3. 전처리된 데이터 학습 진행결과
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/33d25cca-b47d-4e39-a49c-190d6b19a3de)

## 4. 전처리 단순화
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/e146b011-c495-4648-be55-c4f2ba46afaa)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/1e7f0701-8b87-412f-acce-2af320b4c492)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/a7c5a6b8-d3fe-4c34-a377-3909400ebdcd)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/1d3b76ac-74f8-4496-a12b-b84ca75d399e)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/65043ed1-3448-4a66-a8ea-5ebf1ee63747)

## 번역을 이용한 Back Translation
- 코드
papago
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/25d7be5d-ee32-443f-aba2-22aa55bb3bce)
google
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/3c7ab34a-765d-495f-a2f7-a422f5159d4f)

-결과 (적용 X)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/e0a551cb-97dd-4145-bac7-47c325d19968)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/da943ad8-eed1-4f11-92b5-4f12ab4d6ba5)




# 모델 선택
## 1. 여러 모델 실험 1
- 시계열 계열의 모델들을 테스트.
- 성능이 그렇게 좋은 폭으로 나오지 않음.
- 오버피팅이 약 2~3에폭에서 발생하는 문제.
### 1. LSTM
- 순환 신경망(RNN)의 일종으로, 긴 시퀀스 데이터에서의 의존성을 잘 학습할 수 있는 모델
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/efffa74b-2492-47de-a5b0-6347b7efce49)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/56d684a1-739c-4f28-b4b0-dbca779e2e29)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/ea3ecbb1-87cd-4d3c-97c7-11a82a23838a)

### 2. CNN-LSTM
- CNN과 LSTM을 결합한 모델로, 시퀀스 데이터에서 공간적, 시간적 패턴을 동시에 학습 가능 한 모델
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/5410b7c8-a503-434e-bac0-9205ca556073)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/2167faaa-54d7-446f-ba59-ce631b5cfd3d)

### 3. bidrectional LSTM
- 양방향으로 정보를 처리하는 LSTM 모델로, 시퀀스 데이터를 순방향과 역방향으로 동시에 처리하여 더 많은 컨텍스트 정보를 학습 가능한 모델
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/9ba6182b-7e25-4362-9c41-be1469a229a2)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/2f248b56-a6e9-4c92-83fb-cbf31437062b)

### 4. BERT
- 구글이 제안한 모델로, 양방향성과 트렌스포머 아키텍쳐를 기반으로 하며 사전학습과 미세조정이 가능해 텍스트 분류, 질의 응답, 감정 분석 등 다양한 NLP 작업에 사용되는 모델
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/9982f6ce-c180-4c5f-a5dc-875d7c068502)

### 5. BERT Pre-training
- BERT를 사전학습시킨 모델
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/d2dd908e-5d02-468c-af1d-590e55fb70fb)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/891debed-fa88-4e52-bde0-02660b727304)

## 2. 여러 모델 실험 2
- KoELECTRA, Roberta Large 등 다양한 Transformer 계열의 사전 학습 모델을 사용

### transformer_encoder
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/df5a51c1-bc9b-4476-8ab7-8b30c58e8aaf)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/27d272f4-f220-4213-96cb-cfed5a253f60)
  
### KoELECTRA
- ELECTRA의 Generator-Discriminator 구조를 채택한 모델로 한국어 텍스트 코퍼스를 사용해 학습된 모델
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/8d76dff0-b9fa-479f-8fcf-a94f9a52c772)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/45775d0a-6969-47a4-9403-e6a7c239e707)

### Roberta Large
- BERT의 개선된 버전으로, 더 나은 성능을 위해 여러 가지 최적화와 실험을 통해 개발된 모델
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/34e9612a-0674-46ff-b41c-fdaf9b2ab0b6)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/47ca8190-4369-4032-98e8-af384dec040d)

# 모델 학습 [튜닝]
## 모델 앙상블(koBERT+KoELECTRA)
- Soft Voting: 각 class 별로 모델들이 예측한 probability를 합산하여 가장 높은 class를 선택
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/7e3d6aa2-b16b-4dc5-8dd3-a7649e5ed1d2)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/58045c37-c4b2-4f22-bff6-71ecdefdee79)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/3ab5aa09-2113-48c7-a592-dc9b2e960f87)

# 결과 분석

# 리더 보드
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/2d461534-ae99-4bb4-ace1-50bfeb9cb194)

# 실험 Log
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/51a1dad4-818b-4266-bf3c-793e428aaa58/Untitled.png)
![image](https://github.com/JohayoAiffels/Main_branch/assets/132184507/1cdcd862-8db1-4251-9f35-db0d30fed595)
