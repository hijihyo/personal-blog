---
date:   2022-02-25 14:56:00 +0900
category: lecture-summary
tags: michigan dl4cv
---

*Justin Johnson님이 미시간대학교에서 진행했던 ‘컴퓨터 비전을 위한 딥러닝 (Deep learning for computer vision)’ 강의를 듣고 정리하겠다. (2019년도 가을 학기 기준)*

### 1. 이미지 분류의 어려움과 중요성

**이미지 분류 (Image classification)** 는 이미지를 입력받아 정해진 카테고리 집합 중 하나에 배정하는 작업이다. 이미지 분류는 쉽지 않은 작업인데, 이는 우리 인간이 보는 이미지와 컴퓨터가 보는 이미지는 의미적으로 다르기 때문이다. 예를 들어 우리에게는 고양이 이미지로 보이더라도 컴퓨터에게는 0과 255 사이의 숫자로 이루어진 행렬 그 이상 그 이하도 아닐 것이다.

이러한 의미론적 차이를 인지하더라도 이미지 분류는 거쳐야 할 난관이 많다. 촬영 위치 변화 (Viewpoint variation), 클래스 내 다양성 (Interclass variation), 클래스 세분화 (Fine-grained categories), 배경 교란 (Background clutter), 조명 변화 (Illumination changes), 모양 변화 (Deformation), 폐색 (Occlusion) (?) 등이 그 예이다. 우리는 이러한 변화와 다양성을 염두에 두고 이에 대처할 수 있는 견고한 알고리즘을 설계해야 한다.

이미지 분류 작업은 쉽지 않은 작업이지만, 의학, 천문학, 생물학과 같이 다양한 분야에서 활용할 수 있기 때문에 중요성이 크다. 나아가 이미지 분류는 객체 탐지 (Object detection), 이미지 캡셔닝 (Image captioning) 등의 다른 작업들의 기초 토대로써 꼭 필요하다.

### 2. 이미지 분류 데이터셋

다음은 이미지 분류 알고리즘을 학습시키기 위한 데이터셋이다.

- MNIST
    - 0부터 9까지의 손글씨 사진
    - 컴퓨터 비전 분야의 “초파리"
    - 너무 흔하며 대부분의 알고리즘이 잘 학습하기 때문에 신뢰도가 높지 않다.
- CIFAR10
    - 10개의 클래스로 구성된 다양한 유색 사진
- CIFAR100
    - 100개의 클래스로 구성된 다양한 유색 사진
- ImageNet
    - 1000개의 클래스로 구성된 다양한 유색 사진
    - 노이즈를 고려하여 성능 함수로 “상위 5 정확도 (Top 5 accuracy)” 를 사용한다.
    - 대부분의 논문에서 이를 사용할만큼 벤치마크로써 중요하다.
- MIT Places (Places365)
    - 365개의 클래스로 구성된 다양한 유색 장소 사진

<aside>
💡 ImageNet과 MIT Places의 경우 사진마다 크기가 다른데, 학습할 때는 $256\times256$으로 조정하여 사용한다.

</aside>

- Omniglot
    - 1623개의 클래스로 구성된 다양한 문자 사진
    - 각 카테고리마다 20개의 이미지만 제공하여 퓨 샷 러닝 (few shot learning) 에 주로 사용한다.

### 3. 이미지 분류기와 최근접 이웃 분류기

일반적인 함수와 다르게 **이미지 분류기 (Image classifier)** 는 그 내용을 작성할 만한 명백한 알고리즘이 존재하지 않는다. 가장자리를 찾거나, 동물의 특성을 입력하는 등 인간의 지식을 하드코딩할 수는 있으나 이러한 방식은 비효율적이고 확장성이 없다.

우리는 이미지 분류기를 데이터 중심 접근법 (Data-driven approach) 으로 디자인할 것이다. 데이터 중심 접근법은 다음과 같다.

1. 이미지와 레이블 데이터를 수집하고,
2. 머신 러닝을 도입해 분류기를 학습시킨 후,
3. 해당 분류기로 새로운 이미지를 평가한다.

새로운 이미지 분류기를 만들어야 할 때 데이터만 새로 구하면 될 뿐, 코드를 다시 짤 필요가 없어진다.

우리가 가장 먼저 살펴볼 분류기는 **최근접 이웃 (Nearest neighbor)** 분류기이다. 최근접 이웃 분류기는 학습할 때 모든 데이터와 레이블을 저장한 후, 예측할 때는 주어진 이미지와 가장 유사한 학습 이미지의 레이블로 예측하는 모델이다. 기본적으로 최근접 이웃 분류기의 학습 복잡도는 $O(1)$이고, 테스트 복잡도는 $O(N)$이다.

이때 이미지 간의 유사한 정도를 어떻게 측정할 것인지 정해야 하는데, **L1 거리 (L1 distance, Manhattan distance)** 와 **L2 거리 (L2 distance, Euclidean distance)** 가 가장 일반적으로 사용되는 거리 함수이다. 이미지 분류에서 L1 거리는 두 이미지의 각 픽셀값의 차이를 모두 더하는 것이고, L2 거리는 두 이미지의 각 픽셀값의 차이를 제곱하여 모두 더하는 것이다.

최근접 이웃 분류기의 예시 코드는 다음과 같다.

```python
import numpy as np

class NearestNeighbor:
	def __init__(self):
		pass

	def train(self, X, y):
		""" X is N x D where each row is an example. Y is 1-dimensional of size N """
		# the nearest neighbor classifier simply remembers all the training data
		self.Xtr = X
		self.ytr = y

	def predict(self, X):
		""" X is N x D where each row is an example we wish to predict label for """
		num_test = X.shape[0]
		# lets make sure that the output type matches the input type
		Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

		# loop over all test rows
		for i in range(num_test):
			# find the nearest training image to the i'th test image
			# using the L1 distance (sum of absolute value differences)
			distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
			min_index = np.argmin(distances) # get the index with smallest distances
			Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

		return Ypred
```

### 4. 최근접 이웃 분류기와 결정 경계

**결정 경계 (Decision boundary)** 는 두 개의 분류 범위가 만나는 경계선을 의미한다. 아래 그림에서 점들은 학습 데이터를, 점의 색깔은 해당 데이터의 레이블을 나타낸다고 하자. 그리고 배경 색깔은 테스트 데이터가 해당 위치에 있을 때 최근접 이웃 분류기가 예측할 레이블을 나타낸다고 하자. 이때 파란색과 빨간색 범위가 만나는 경계선, 노란색과 보라색 범위가 만나는 경계선 등이 결정 경계이다.

![최근접 이웃 분류기의 결정 경계 예시](/assets/images/2022-02-25-lecture-2-image-classification/resource-1.png)

그러나 위의 그림에서 경계선은 꽤 울퉁불퉁하고, 초록색 지역에 노란색 점이 덩그라니 있는 것처럼 **이상치 (outlier)** 에 영향을 받고 있다. 이상치에 대처하고 경계선을 조금 더 합리적으로 만드는 방법은 없을까?

바로 여러 개의 이웃을 이용하는 것이다. **k-최근접 이웃 분류기 (k-Nearest neighbor)** 는 k개의 최근접 이웃을 찾은 후 다수결 투표를 진행하여 가장 많은 레이블로 예측을 하는 방식이다. 해당 분류기를 사용하면 여러 이웃을 참고하기 때문에 결정 경계가 매끄러워지고, 이상치의 영향도 덜 받게 된다. 다만 몇몇 부분에서 가장 많은 레이블이 두 개 이상이 될 수 있어, 이러한 충돌이 발생할 때 대처할 방안도 고안해야 한다.

![k-최근접 이웃 분류기의 결정 경계 예시](/assets/images/2022-02-25-lecture-2-image-classification/resource-2.png)

### 5. 하이퍼파라미터

**하이퍼파라미터 (hyperparameter)** 는 학습 과정 중에 얻는 인자가 아닌, 우리가 학습 과정을 시작하기 전에 설정하는 인자이다. k-최근접 이웃 분류기의 경우 k의 값과 거리 함수가 하이퍼파라미터이다. 문제마다 가장 좋은 하이퍼파라미터가 달라지는 편이다. 일반적으로 우리는 모든 하이퍼파라미터를 시도해보고 가장 잘 동작하는 것 같은 인자를 선택한다.

하이퍼파라미터를 설정하는 방식은 다음과 같다.

1. 데이터에 가장 잘 동작하는 하이퍼파라미터로 선택하기
    - 좋은 방식이 아니다. k-최근접 이웃 분류기의 경우 학습 데이터에 대해서는 $k=1$이 항상 잘 동작하기 때문이다.
    
    ![하이퍼파라미터 설정 방식 예시](/assets/images/2022-02-25-lecture-2-image-classification/resource-3.png)
    
2. 데이터를 학습 데이터와 테스트 데이터로 나누고 테스트 데이터에 가장 잘 동작하는 하이퍼파라미터로 선택하기
    - 테스트 데이터에 가장 잘 동작하는 하이퍼파라미터로 선택하고 테스트 데이터에 대한 성능으로 최종 성능을 가늠할 경우, 이미 테스트 데이터를 봤기 때문에 정확한 성능을 알 수 없다.
    
    ![하이퍼파라미터 설정 방식 예시](/assets/images/2022-02-25-lecture-2-image-classification/resource-4.png)
    
3. 데이터를 학습 데이터와 평가 데이터 (validation data), 테스트 데이터로 나누고 평가 데이터에 가장 잘 동작하는 하이퍼파라미터로 선택한다. 성능 측정은 테스트 데이터로 수행한다.
    - 위의 두 방식보다 나은 방식이다. 테스트 데이터에 대한 성능 측정은 여러 번 수행해서는 안되고 가장 마지막에 수행해야 한다.
    
    ![하이퍼파라미터 설정 방식 예시](/assets/images/2022-02-25-lecture-2-image-classification/resource-5.png)
    
4. **교차 검증 (Cross validation)**. 데이터를 여러 개의 폴드 (fold) 로 나누고, 각 폴드를 돌아가면서 평가 데이터나 테스트 데이터로 사용한다.
    - 가장 견고한 방식이지만 오래 걸린다. 그래서 데이터셋의 크기가 작을 경우에는 유용하게 사용하지만 딥러닝에서는 잘 사용하지 않는다.
    
    ![하이퍼파라미터 설정 방식 예시](/assets/images/2022-02-25-lecture-2-image-classification/resource-6.png)
    

### 6. 최근접 이웃 분류기의 한계와 활용

최근접 이웃 분류기 혹은 k-최근접 이웃 분류기 (이하 최근접 이웃 분류기) 는 테스트를 수행하는 데에 오래 걸리고 학습한 데이터에 대해 깊이 있게 이해하지 못한다는 단점이 있다. 그러나 학습 데이터가 많아질수록 최근접 이웃 분류기는 더 많은 함수를 표현해낼 수 있다. 학습 데이터가 충분하기만 하다면 최근접 이웃 분류기는 이론적으로 모든 함수를 표현해낼 수 있다. 이를 **만능 근사 (Universal approximation)** 라 부른다.

![하이퍼파라미터 설정 방식 예시](/assets/images/2022-02-25-lecture-2-image-classification/resource-7.png)

다만 만능 근사를 충분히 하기 위해서는 해당 데이터 공간을 골고루 커버할 수 있는 만큼의 데이터가 필요한데, 차원이 높아질 수록 필요한 데이터 수가 기하 급수적으로 증가한다. 이러한 **차원의 저주 (Curse of dimensionality)** 도 최근접 이웃 분류기에 한계를 부여한다.

그럼에도 불구하고 최근접 이웃 분류기가 활용성이 없는 것은 아니다. TF-IDF와 같이 거리 함수를 세심하게 고안하면 최근접 이웃 분류기로도 유사 문서 찾기와 같은 유용한 작업을 수행할 수 있다. 그리고 최근접 이웃 분류기에 원래 이미지를 입력하지 않고 뉴럴 네트워크에서 생성된 특성 벡터를 입력하면 더욱 고도화된 작업을 수행할 수 있다.