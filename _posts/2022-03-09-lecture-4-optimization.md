---
date:   2022-03-09 16:05:00 +0900
category: lecture-summary
tags: michigan dl4cv
---

*Justin Johnson님이 미시간대학교에서 진행했던 ‘컴퓨터 비전을 위한 딥러닝 (Deep learning for computer vision)’ 강의를 듣고 정리하겠다. (2019년도 가을 학기 기준)*

### 1. 최적화와 임의 검색법 (Optimization and Random Search)

**최적화**

지난 시간에는 선형 분류기 $f(x; W)=Wx$ 에 대해 손실 함수를 정의하여 가중치 $W$를 평가하는 방법을 다루었다. 이제는 손실 함수의 값이 가장 작게 나오는 가중치를 선택하여 선형 분류기의 가중치로 사용하면 된다. 그러나 “가장 적합한 가중치” 내지 “손실 함수의 값이 가장 작은 가중치”를 어떻게 찾을 것인가? 

가장 좋은 가중치를 찾는 문제는 손실 함수의 값을 최소화하는 **최적화 문제 (Optimization problem)** 와 동일하다. 가중치 $w$에 대한 손실 함수를 $L(w)$라고 할때, 일반적으로 최적화 문제는 다음과 같이 표현한다:

$$
w^*=\arg\min_w{L(w)}
$$

가중치를 2차원 파라미터라고 가정하면 최적화 문제를 보다 더 직관적으로 이해할 수 있다. 어떤 2차원 평면에서 임의의 점 $(x,y)$를 하나의 가중치 값이라고 생각하자. 그리고 해당 평면을 바닥으로 하는 3차원 공간을 상상한 후 각 점에 대한 손실 함수의 값이 바닥으로부터의 높이에 해당한다고 가정하자. 마치 아래 그림과 같을 것이다.

![어떤 3차원 공간에서의 지형](/assets/images/2022-03-09-lecture-4-optimization/resource-1.png)

어떤 3차원 공간에서의 지형

이와 같을 때 손실 함수의 값을 최소화해나가는 과정은, 우리가 위 지형의 한 점에서 출발하여 가장 낮은 곳으로 하강해나가는 과정과 동일할 것이다. 

**임의 검색법 (Random search)**

최적화를 수행하기 위한 가장 쉬운 방법은 **임의 검색법 (Random search)** 이다. 이 기법에서는 가중치 $W$의 값을 임의로 선택하여 여러 번 시도해본 후 그 중에서 손실 함수의 값이 가장 작았던 가중치를 선택한다. 가중치를 어떻게 선택하느냐에 따라 결과가 매번 달라질 수 있으며 예측 성능도 그다지 좋지는 않다.

단순한 호기심으로 임의 검색법을 구현하여 CIFAR10 데이터셋을 예측해보았다. 강의 자료와 다르게 테스트셋에 대한 정확도가 50% 정도였는데 이유는 잘 모르겠다. 그리고 실습을 통해 임의 검색법 과정 동안 손실 함수의 값은 굉장히 많이 진동한다는 것을 직접 확인할 수 있었다.

![임의 검색법 과정에서 손실 함수값의 변화](/assets/images/2022-03-09-lecture-4-optimization/resource-2.png)

임의 검색법 과정에서 손실 함수값의 변화

[Google Colaboratory](https://colab.research.google.com/drive/1hOiq0MOwDHX2uayaSaZpPJDowSbMZyDj?usp=sharing)

04 Optimization - Random Search

### 2. 그래디언트 계산 (Computing Gradients)

**수치적 그래디언트 (Numerical gradient)**

경사 하강법을 다루기 전에 그래디언트와 이를 계산하는 방법을 알아보자.

1차원에서는 함수를 미분값 (derivative) 을 구하여 기울기를 알 수 있다. 나아가 다차원에서는 각 차원에 대한 편미분값을 구할 수 있으며, 이들로 이루어진 벡터를 **그래디언트 (gradient)** 라고 부른다. 어떤 점에서 그래디언트의 방향은 해당 점에서 가장 가파르게 증가하는 방향이다.

$$
\cfrac{df(x)}{dx}=\lim_{h\rightarrow0}{\cfrac{f(x+h)-f(x)}{h}}
$$

$$
\nabla_x{f(x)}=
\big(
\cfrac{\partial f(x)}{\partial x_1},
\cfrac{\partial f(x)}{\partial x_2},
\cdots,
\cfrac{\partial f(x)}{\partial x_m}
\big)
\quad
\text{for }x\in\mathbb{R}^m\text{, and }f(x)\in\mathbb{R}
$$

**수치적 그래디언트 (Numerical gradient)** 는 **수치 미분 (Numerical differentiation)** 을 이용하여 구한 것이다. 수치 미분에서는 입력 벡터 $x$의 각 차원마다 값을 조금씩 변경하여 새로운 손실 함수 값을 구한 뒤, 위의 1차 미분식을 이용하여 해당 요소의 미분값을 근사적으로 알아내는 것이다.

수치적 그래디언트는 구하기는 쉽지만, 각 차원마다 손실 함수값을 구해서 계산해야 하기 때문에 시간 복잡도가 $O(\sharp\text{dimensions})$ 로 느리다. 그리고 수치 미분값은 근사값일 뿐 실제 그래디언트를 계산한 것은 아니다. 이러한 이유로 수치적 그래디언트는 직접 사용하지 않으며, 다른 방법으로 구한 그래디언트가 맞는지 확인하는 용도 (gradient check) 로 사용한다.

**해석적 그래디언트 (Analytic gradient)**

해석적 그래디언트 (Analytic gradient) 는 해석 미분 (Analytic differentiation) 을 이용하여 구한 것으로, 손실 함수와 가중치의 관계에 수학적 배경 지식을 적용하여 구한 것이다. 해석적 그래디언트 식을 이끌어내는 과정에서 실수하기는 쉬우나, 빠르고 정확하다는 점에서 수치적 그래디언트를 압도한다. 어떻게 구하는지는 추후에 다루기로 하자.

### 3. 경사 하강법 (Gradient Descent)

**경사 하강법 (Gradient descent)**

강의 초반부에 따르면 최적화 기법은 임의의 지형에서 하강해나가는 것과 유사하다고 했다. 만약 우리가 처음 보는 산의 정상에 있다면 어떻게 내려갈 것인가? 아마도 대부분의 사람들은 내려가는 방향으로 걸음을 옮기면서 하산할 것이다.

**경사 하강법 (Gradient descent)** 은 함수의 기울기를 구하여 내려가는 방향으로 진전하도록 하는 알고리즘이다. 만약 우리가 서있는 지점에서 내려가는 방향으로 향한다면, 몇 번의 반복 후에는 꽤 많이 내려갈 수 있을 것이다. 임의의 점에서 함수의 그래디언트는 해당 점에서 가장 가파르게 올라가는 방향이므로, 그래디언트의 정반대 방향 즉 -그래디언트가 해당 점에서 내려가는 방향이라고 생각할 수 있다. 경사 하강 과정을 자세하게 나타내자면 다음과 같다:

```python
# Vanilla gradient descent
w = initialize_weights()
for t in range(num_steps):
	dw = compute_gradient(loss_fn, data, w)
	w -= learning_rate * dw
```

경사하강법의 하이퍼파라미터는 3가지로, 가중치 초기화 방식 (weight initialization method), 스텝 수 (number of steps), 학습률 (learning rate) 이 이에 해당한다.

데이터셋 $\{(x_i, y_i)\}_{i=1}^N$ 에 대해 경사 하강법을 적용할 경우 이를 **배치 경사 하강법 (Batch gradient descent)** 라고 부른다. 이때 손실 함수와 이의 그래디언트는 다음과 같다:

$$
L(W) = \cfrac{1}{N}\displaystyle\sum_{i=1}^N{L_i(x_i, y_i, W)} + \lambda R(W)
$$

$$
\nabla_WL(W) = \cfrac{1}{N}\displaystyle\sum_{i=1}^N{\nabla_WL_i(x_i, y_i, W)} + \lambda\nabla_WR(W)
$$

**확률적 경사 하강법 (Stochastic gradient descent; SGD)**

배치 경사 하강법을 도입할 경우 한 스텝 나아갈 때마다 $N$개의 데이터에 대해 손실 함수와 이의 그래디언트를 구해야 한다. $N$이 커질수록 더 많은 양을 계산해야 하기 때문에 이러한 방식은 비효율적이다. 이러한 단점을 개선한 것이 바로 **확률적 경사 하강법 (Stochastic gradient descent; SGD)** 이다. 확률적 경사 하강법에서는 매 스텝마다 특정한 개수의 데이터를 임의로 선택하여 이들로만 가중치를 업데이트한다.

$$
L(W) =
\mathbb{E}_{(x,y)\sim{p_{data}}}[L(x,y,W)]+\lambda R(W)
\\\quad~~\approx\cfrac{1}{N}\displaystyle\sum_{i=1}^N{L(x_i, y_i, W)}+\lambda R(W)
$$

$$
\nabla_WL(W) =
\nabla_W\mathbb{E}_{(x,y)\sim{p_{data}}}[L(x,y,W)]+\lambda\nabla_WR(W)
\\\quad~~\approx\cfrac{1}{N}\displaystyle\sum_{i=1}^N{\nabla_WL(x_i, y_i, W)}+\lambda\nabla_WR(W)
$$

즉 손실 함수와 이의 그래디언트를 데이터셋의 **미니배치 (minibatch)** 를 이용하여 근사적으로 구한다. 이때 미니배치의 사이즈를 배치 크기 (batch size) 라고 하며, 주로 32, 64, 그리고 128를 값으로 선택한다. 확률적 경사 하강법의 하이퍼파라미터는 경사 하강법의 것과 동일하며, 배치 사이즈와 데이터 샘플링 기법을 추가로 가진다.

확률적 경사 하강법의 알고리즘은 다음과 같다:

```python
# Stochastic gradient descent
w = initialize_weights()
for t in range(num_steps):
	minibatch = sample_data(data, batch_size)
	dw = compute_gradient(loss_fn, data, w)
	w -= learning_rate * dw
```

### 4. 경사 하강법의 변형 (Variations of Gradient Descent)

**SGD의 한계**

1. 함수가 한 방향으로는 천천히 변화하고 다른 방향으로는 가파르게 변화하는 형태를 가진다면, SGD에서는 기울기가 낮은 방향으로는 천천히 진행되고 기울기가 급한 방향으로는 진동할 수 있다 (과하게 진전했다가 되돌아오는 형태). 손실 함수가 높은 조건수 (condition number; 파라미터에서의 작은 비율에 대해 함수가 얼마나 변화할 수 있는지) 를 가진다. (?)
2. 그래프에서 안장점 (saddle point; 쪽 그림. 마치 말의 안장과 같이 한 방향으로는 감소하고 다른 방향으로는 증가하는 점으로, 해당 점의 그래디언트는 0) 이나 극소점 (local minimum; 함수의 최솟값은 아니나 해당 점의 주변 값보다 낮은 지점) 이 나타난다면, 해당 점에 갇힐 수 있다.
    
    ![Screen Shot 2022-03-09 at 5.57.18 PM.png](/assets/images/2022-03-09-lecture-4-optimization/resource-3.png)
    
    ![Screen Shot 2022-03-09 at 5.57.36 PM.png](/assets/images/2022-03-09-lecture-4-optimization/resource-4.png)
    
3. SGD에서 각 그래디언트는 미니배치로부터 얻은 근사값이기 때문에 정확하지 않고 잡음 (noise) 이 있다.

다음 알고리즘들은 이러한 SGD의 한계를 극복하기 위해 고안된 것들이다.

**SGD+Momentum**

SGD+Momentum 기법은 관성이라는 물리적 현상에 착안하여 만들어진 것으로 (?), 속도 (velocity) 를 계산하여 해당 방향으로 진전하는 기법이다. 속도는 그래디언트의 지수 이동 평균 (exponential moving average, exponentially weighted moving average) 이다.

```python
v = 0
for t in range(num_steps):
	dw = compute_gradient(w)
	v = rho * v + dw # rho: gives "friction" typically 0.9 or 0.99
	w -= learning_rate * v
```

SGD+Momentum 기법에서는 꾸준히 진행해온 방향으로 나아가려는 성질이 있기 때문에 극소점이나 안장점에서 벗어날 수 있는 가능성이 존재한다.

**Nesterov Momentum**

$$
\begin{matrix}
v_{t+1} = \rho v_t -\alpha\nabla f(x_t+\rho v_t) & & v_{t+1} = \rho v_t -\alpha\nabla f(\tilde{x}_t) \\
x_{t+1} = x_t + v_{t+1} & \Rightarrow & \tilde{x}_{t+1} = \tilde{x}_t -\rho v_t + (1+\rho)v_{t+1} \\
& & ~~~~~~~~~~~= \tilde{x}_t + v_{t+1} + \rho(v_{t+1} - v_t)
\end{matrix}\\
\text{where }\tilde{x}_t=x_t+\rho v_t
$$

네스테로프 모멘텀 (Nesterov momentum) 기법은 SGD+Momentum 기법과 동일하나 그래디언트를 구하는 지점이 다르다. SGD+Momentum 기법에서는 현재 위치에서 그래디언트를 구하지만, 네스테로프 모멘텀 기법에서는 현재 지점에서 속도만큼 나아간 지점에서 그래디언트를 구한다. 알고리즘은 다음과 같다:

```python
v = 0
for t in range(num_steps):
	dw = compute_gradient(w)
	old_v = v
	v = rho * v - learning_rate * dw
	w -= rho * old_v - (1 + rho) * v
```

SGD+Momentum 기법과 네스테로프 모멘텀 기법은 바닐라 SGD 기법보다 학습 과정을 가속화하지만, 급격하게 나아갔다가 되돌아오는 경향 (overshoot) 이 있다.

**AdaGrad**

SGD 기법에서는 그래디언트의 크기로 인해 완만한 곳은 천천히 가고 가파른 곳은 빠르게 가는 단점이 있었다. 이를 완화하기 위해 **AdaGrad 기법**에서는 그래디언트의 제곱값을 모두 더하여 이 값의 제곱근으로 학습률를 나눈다. (경사 하강 과정에서 그래디언트 내지 스텝의 크기를 일정하게 유지하기 위함인 것 같다. (?)) 그래디언트의 크기에 따라 학습률을 조절한다는 점에서 (?) 파라미터별 학습률 (Per-parameter learning rates) 혹은  적응형 학습률 (Adaptive learning rates) 라고도 부른다.

AdaGrad 기법의 알고리즘은 다음과 같다:

```python
grad_squared = 0
for t in range(num_steps):
	dw = compute_gradient(w)
	grad_squared += dw * dw
	w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
```

**RMSProp**

그러나 AdaGrad 기법에서는 그래디언트의 제곱값을 계속 더하기 때문에 시간이 지남에 따라 이 합이 너무 커진다는 단점이 존재한다. RMSProp 기법에서는 그래디언트의 제곱값의 지수 이동 평균을 대신 사용하여 이러한 단점을 해결한다.

```python
beta = 0
grad_squared = 0
for t in range(num_steps):
	dw = compute_gradient(w)
	grad_squared = beta * grad_squared + (1 - beta) * dw * dw
	w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
```

**Adam**

Adam 기법은 RMSProp 기법과 모멘텀 기법을 결합한 것이다. RMSProp 기법은 현재 위치의 경사에 따라 속도가 급격하게 변하는 것을 방지하고, 모멘텀 기법은 꾸준히 진행해온 방향으로 갈 수 있도록 하는데,  이 두 기법을 조합한 것이다. 알고리즘은 다음과 같다:

```python
moment1 = 0
moment2 = 0
for t in range(num_steps):
	dw = compute_gradient(w)
	moment1 = beta1 * moment1 + (1 - beta1) * dw # Momentum
	moment2 = beta2 * moment2 + (1 - beta2) * dw * dw # RMSProp
	w -= learning_rate * moment1 / (moment2.sqrt() + 1e-7) # Momentum & RMSProp
```

그러나 위의 알고리즘에서는 $t=0$ 이고 $\beta_1 \approx 1$일 때 마지막 줄에서 $0$에 가까운 값으로 나누게 된다. 이 경우 경사 하강 초반에 스텝의 크기가 과도하게 커지기 때문에, **편향 조정 (Bias correction)** 을 도입하여 초반 $moment_1$과 $moment2$의 크기가 $0$에 가까울 수 있도록 해야 한다.

```python
moment1 = 0
moment2 = 0
for t in range(num_steps):
	dw = compute_gradient(w)
	moment1 = beta1 * moment1 + (1 - beta1) * dw # Momentum
	moment2 = beta2 * moment2 + (1 - beta2) * dw * dw # RMSProp
	moment1_unbias = moment1 / (1 - beta1 ** t)
	moment2_unbias = moment2 / (1 - beta2 ** t)
	w -= learning_rate * moment1 / (moment2.sqrt() + 1e-7) # Momentum & RMSProp
```

Adam 기법은 다양한 태스크에서 잘 동작하는 최적화 기법으로 자주 사용된다. 실제 사용 시 $\beta_1=0.9$, $\beta_2=0.999$, 그리고 학습률은 $1e-3$, $5e-4$, $1e-4$ 중에서 하나로 선택한다.

**2차 최적화 (2nd order optimization)**

위 알고리즘들과 같이 BFGS, L-BFGS 기법과 같이 1차 근사를 이용하지 않고 2차 미분값을 이용하는 기법이 있다. 이들은 스텝 크기를 더 정확하게 구할 수 있고 잘 동작하지만, 시간 복잡도가 크고 미니배치에서는 잘 동작하지 않는다는 단점이 있다. 따라서 기본적으로 Adam이나 SGD+Momentum 기법을 채택하되 풀배치 (full batch) 가 가능하다면 L-BFGS 기법을 고려하는 것도 좋을 것 같다.