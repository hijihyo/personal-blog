---
title: Summary of Michigan DL4CV Lecture 8
date: 2022-03-30 18:27:00 +0900
category: lecture-summary
tags: michigan dl4cv
---

*미시간대학교 컴퓨터 비전을 위한 딥러닝 (Deep Learning for Computer Vision) 의 8강 CNN 구조를 듣고 정리하였습니다.*

- **CNN의 구성 요소들**을 어떻게 조합할 것인가?
    - e.g. convolutional layers, pooling layers, fully-connected layers, activation functions, normalization, etc.
- ImageNet Classification Challenge를 기준으로 CNN 구조의 발전 과정을 살펴보자.
    1. **AlexNet**
        - 뉴럴 네트워크를 도입하여 해당 대회를 우승한 첫 번째 사례
        - 총 8개의 레이어로 구성됨
    2. **ZFNet**
        - 시행착오를 거쳐 AlextNet보다 큰 네트워크를 개발함
    3. **VGGNet**
        - 이전에는 필터의 크기와 개수, 레이어의 구성 등을 시행착오를 거쳐 직접 선택해야 했다.
        - 레이어별 특성을 파악하여 일종의 디자인 규칙을 정리했다.
            - convolutional layer - 3x3 stride 1 pad 1
            - max-pool layer - 2x2 stride 2
            - double #channels after pooling
    4. **GoogLeNet**
        - 효율적인 뉴럴 네트워크를 구성하는 것에 초점을 두었다.
        - stem network - 데이터를 초반에 강하게 다운샘플링하여 연산량을 줄였다.
        - inception module - 서로 다른 커널 크기를 가진 convolutional layer와 max-pool layer를 병렬적으로 두었다.
        - 뉴럴 네트워크의 마지막에 fully-connected layer 대신 global average pooling layer를 사용하여 연산량을 줄였다.
        - 네트워크가 깊어질수록 그래디언트가 잘 전달이 되지 않는 문제가 있었다. auxiliary classifier를 두어 중간에서 손실과 그래디언트를 계산하고 전달하였다.
    5. **ResNet**
        - 뉴럴 네트워크가 깊어질수록 학습을 시키기 어려웠다.
        - 레이어 간에 skip-connection을 두어 residual block을 구성하였다. 이로 인해 항등 함수를 잘 학습해낼 수 있었으며, 그래디언트를 잘 전달할 수 있었다.
        - 무려 152개의 레이어로 구성된 네트워크를 학습시킬 수 있었다.
        - GoogLeNet과 같이 초반에 강하게 다운샘플링하고 마지막에는 global average pooling을 사용하였다.
- 이후 개선된 모델들: ResNeXt,  SENet, DenseNet, MobileNet, etc.
- Neural architecture search와 같이 뉴럴 네트워크 디자인을 자동화하려는 시도도 나타났다.

<aside>
💡 <b>Which architecture should I use?</b><br>
Don’t be a hero. For most problems you should use an off-the-shelf architecture; don’t try to design your own!
1. If you just care about accuracy, ResNet-50 or ResNet-101 are great choices.
2. If you want an efficient network (real-time, run on mobile, etc) try MobileNets and ShuffleNets.

</aside>

- 참고 자료
    - Slides and videos from Michigan EECS 498-007 / 598-005 Deep Learning for Computer Vision (Fall 2019) by Justin Johnson [[Link]](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/)