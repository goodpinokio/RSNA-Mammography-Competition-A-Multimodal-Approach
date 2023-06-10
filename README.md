# RSNA 유방암 데이터셋 전처리, 이미지 크롭 및 멀티 모달 학습
이 문서는 유방암 데이터셋의 전처리 과정과 이미지의 크롭(crop) 과정, 그리고 멀티 모달 학습을 설명합니다.

# 데이터 전처리
데이터 전처리 단계는 딥러닝 프로젝트에서 중요한 단계로, 입력 데이터의 품질과 모델의 성능에 큰 영향을 미칩니다. 
여기서 우리는 biopsy 값이 1인 데이터만을 선택하여 새로운 DataFrame을 생성합니다. biopsy 값이 1인 것을 기준으로 하는 이유는, 의료 데이터에서 biopsy는 조직 검사를 의미하며, 이 값이 1일 때는 보통 병세(여기서는 유방암)가 발견된 경우를 의미합니다.

''' 
import pandas as pd

data_input = pd.read_csv('train_data.csv')

DF_train = data_input[data_input['biopsy']==1].reset_index(drop=True)
DF_train.info()
'''

다음으로, 우리는 클래스 불균형 문제를 해결하기 위해 재샘플링을 진행합니다. 
'cancer' 열을 기준으로 그룹을 만든 후, 각 그룹에서 임의로 1158개의 샘플을 선택하여 새로운 DataFrame을 만듭니다.

''' 
DF_train = DF_train.groupby(['cancer']).apply(lambda x: x.sample(1158,replace = True)).reset_index(drop=True)

print('New Data Size:', DF_train.shape[0])
print(DF_train.value_counts())
'''

# 이미지 크롭


