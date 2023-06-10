# RSNA 유방암 데이터셋 전처리, 이미지 크롭 및 멀티 모달 학습
이 문서는 유방암 데이터셋의 전처리 과정과 이미지의 크롭(crop) 과정, 그리고 멀티 모달 학습을 설명합니다.

# 데이터 전처리
데이터 전처리 단계는 딥러닝 프로젝트에서 중요한 단계로, 입력 데이터의 품질과 모델의 성능에 큰 영향을 미칩니다. 
여기서 우리는 biopsy 값이 1인 데이터만을 선택하여 새로운 DataFrame을 생성합니다. 
biopsy 값이 1인 것을 기준으로 하는 이유는, 의료 데이터에서 biopsy는 조직 검사를 의미하며, 
이 값이 1일 때는 보통 병세(여기서는 유방암)가 발견된 경우를 의미합니다.

code
```python
import pandas as pd

data_input = pd.read_csv('train_data.csv')

DF_train = data_input[data_input['biopsy']==1].reset_index(drop=True)
DF_train.info()
```

다음으로, 우리는 클래스 불균형 문제를 해결하기 위해 재샘플링을 진행합니다. 
'cancer' 열을 기준으로 그룹을 만든 후, 각 그룹에서 임의로 1158개의 샘플을 선택하여 새로운 DataFrame을 만듭니다.

```python
DF_train = DF_train.groupby(['cancer']).apply(lambda x: x.sample(1158,replace = True)).reset_index(drop=True)

print('New Data Size:', DF_train.shape[0])
print(DF_train.value_counts())
```

# 이미지 크롭
원본 이미지는 512*512 사이즈이다.
유방과 관련이 없는 데이터도 같이 들어있다.
그래서 글쓴이는 Kmeans와 roi 추출을 사용해서 유방부분만 크롭했다.
원본 이미지(512*512)는 각 픽셀의 값을 기준으로 K-평균 클러스터링을 통해 여러 그룹으로 분류됩니다.
가장 어두운 클러스터는 배경 노이즈로 가정되어 값이 0으로 설정되어 이미지의 노이즈를 줄입니다.

code
```python
def kmeans_set_zero(img, dsize=(320,512), num_clusters=4):
    # 이미지를 1차원으로 만듭니다.
    img_1d = img.reshape(-1,1)
     
    # kmeans를 적용합니다.
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(img_1d)
    cluster_ids_x = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    smallest_sum = np.inf
    smallest_id = 0
    for cluster_id in range(num_clusters):
        cluster_sum = img_1d[cluster_ids_x == cluster_id].sum() / (100000)
        if cluster_sum < smallest_sum:
            smallest_sum = cluster_sum
            smallest_id = cluster_id

    img_1d[cluster_ids_x == smallest_id] = 0    
    kmeans_img = img_1d.reshape(dsize)
    return kmeans_img

```

# ROI 추출
이미지에서 각 행과 열의 합을 계산하고, 이 값이 일정 임계값보다 큰 부분만 선택합니다. 
이 방법을 통해 이미지에서 의미 있는 부분을 잘라낼 수 있습니다. 
사용되는 임계값은 이미지 전체 픽셀 값의 합의 평균입니다.

code
```python
def extraxtor_from_roi_box(d, frame, dsize=(512, 320), num_clusters=4):

    h, w = frame.shape
    org_dsize = (h, w)

    frame = kmeans_set_zero(frame, dsize=org_dsize, num_clusters=num_clusters)
    frame_org = copy.copy(frame)

    thres1 = np.min(frame) + 68
    np.place(frame, frame < thres1, 0)

    thres2 = frame_org.sum() / (h * w)
    vertical_not_zero = [True if frame[:, idx].sum() > thres2 else False for idx in range(w)]
    horizontal_not_zero = [True if frame[idx, :].sum() > thres2 else False for idx in range(h)]

    crop = frame_org[horizontal_not_zero, :]
    crop = crop[:, vertical_not_zero]

    crop = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)

    return crop

```

# 이미지 크기 조절 
마지막으로, 모든 이미지는 동일한 크기 (예: 320x512)로 조정됩니다. 
이 단계는 모델에 입력될 이미지의 크기를 통일하는 데 필요합니다.

code
```python
crop = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
```

# 멀티 모달 학습을 위한 원핫 인코딩
사람의 두 눈은 같은 장면을 약간 다른 각도에서 보기 때문에 공간 인식에 도움을 줍니다. 
마찬가지로 유방암 진단에서도 여러 각도에서 촬영한 이미지를 동시에 사용하여 학습하면 정확도를 높일 수 있습니다.

우리는 'laterality'와 'view' 피처를 원핫 인코딩하여 각 이미지가 왼쪽 유방 이미지인지 오른쪽 유방 이미지인지, 
그리고 어떤 각도에서 촬영한 이미지인지를 나타내는 정보를 학습에 포함시킵니다.

code
```python
laterality_one_hot = pd.get_dummies(DF_train['laterality'], prefix='laterality')
view_one_hot = pd.get_dummies(DF_train['view'], prefix='view')

DF_train_encoded = pd.concat([DF_train, laterality_one_hot, view_one_hot], axis=1)
DF_train_encoded = DF_train_encoded.drop(['laterality', 'view'], axis=1)
```

이렇게 처리한 후의 DataFrame에서, 
우리는 'laterality_L', 'laterality_R', 'view_1', 'view_5' 피처와 타겟 변수 'cancer'를 추출하여 학습 데이터로 사용합니다.
또한 각 이미지의 경로도 추출하여 이후에 이미지를 불러올 때 사용합니다.

code
```python
structured_data = DF_train_encoded[['laterality_L', 'laterality_R', 'view_1', 'view_5']].values
target = DF_train_encoded['cancer'].values
image_paths = DF_train_encoded['img_path'].values
```

학습 데이터를 나누기 전에, 우리는 모든 이미지를 동일한 크기로 리사이징하고, 
이미지의 픽셀 값을 0~1 범위로 정규화합니다. 
이렇게 하면 모델의 학습 속도를 높이고, 학습이 안정적으로 이루어지도록 돕습니다.

code
```python
from keras.preprocessing.image import load_img, img_to_array

def load_and_process_image(image_path):
    # Load image file
    image = load_img(image_path, target_size=(512, 320))  # resize the image to target size while loading
    # Convert the image to numpy array and normalize to [0,1]
    image = img_to_array(image) / 255.0
    return image

train_images = np.array([load_and_process_image(img_path) for img_path in train_image_paths])
test_images = np.array([load_and_process_image(img_path) for img_path in test_image_paths])
```

# 멀티 모달 학습을 위한 모델 설계
우리는 두 개의 다른 종류의 입력을 받을 수 있는 모델을 만듭니다. 하나는 이미지 데이터를 처리하는 컨볼루션 신경망(CNN) 부분이고, 다른 하나는 원핫 인코딩된 'laterality'와 'view' 정보를 처리하는 완전 연결 계층(Dense layer)입니다.

이 두 부분은 각각 처리된 후에 결합(concatenate)되어 최종 출력을 생성합니다.

code
```python
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.optimizers import Adam

image_input = Input(shape=(512, 320, 3))
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

structured_input = Input(shape=(4,))
y = Dense(16, activation='relu')(structured_input)
y = Dropout(0.5)(y)

combined = concatenate([x, y])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.5)(z)
z = Dense(32, activation='relu')(z)
z = Dropout(0.5)(z)
output = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[image_input, structured_input], outputs=output)
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
```

# 모델 학습
이제 우리는 학습 데이터와 타겟 값을 사용하여 모델을 학습시킬 수 있습니다. 
학습 과정에서는 검증 데이터도 사용하여 학습이 잘 이루어지고 있는지를 확인합니다.

code
```python
history = model.fit([train_images, train_structured_data], train_target, validation_split=0.2, epochs=10, batch_size=32)
```

# 학습 결과 평가
학습이 끝난 후에는 테스트 데이터를 사용하여 모델의 성능을 평가합니다. 
테스트 데이터에 대한 예측 값을 실제 값과 비교하여 정확도를 계산하고, 이를 출력합니다.

code
```python
test_loss, test_acc = model.evaluate([test_images, test_structured_data], test_target)
print("Test accuracy:", test_acc)
```
img

<img src="https://github.com/goodpinokio/RSNA-Mammography-Competition-A-Multimodal-Approach/assets/73101224/27ad1976-a002-4d89-a42f-2342924e82da">
 
이렇게 학습한 모델을 사용하니, 테스트 정확도가 0.8039로 상당히 높게 나왔습니다. 
이는 학습 데이터에 'laterality'와 'view' 정보를 추가로 포함함으로써 모델의 성능이 크게 향상되었음을 보여줍니다.

# 결과 시각화
또한, 학습 과정에서의 정확도와 손실 값의 변화를 시각화하여 학습이 어떻게 진행되었는지 확인할 수 있습니다.

code
```python
import matplotlib.pyplot as plt

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize = (15,10))

plt.subplot(2, 2, 1)
plt.plot(accuracy, label = "Training Accuracy")
plt.plot(val_accuracy, label = "Validation Accuracy")
plt.ylim(0.4, 1)
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.title("Training vs Validation Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.subplot(2, 2, 2)
plt.plot(loss, label = "Training Loss")
plt.plot(val_loss, label = "Validation Loss")
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.title("Training vs Validation Loss")
plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()
```

이 외에도, 모델의 성능을 평가하는 여러 가지 방법들이 있습니다. 
ROC 곡선을 그려 모델의 성능을 시각적으로 확인하거나, 혼동 행렬(confusion matrix)를 사용하여 실제 값과 예측 값의 관계를 확인하는 것이 일반적입니다.
이들을 통해 모델의 성능을 좀 더 자세히 이해하고, 필요한 경우 모델을 개선하는 데 도움이 될 수 있습니다.
