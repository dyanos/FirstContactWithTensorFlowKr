번역자 : 이형도 Hyeongdo Lee(mylovercorea@gmail.com)
#A PRACTICAL APPROACH
#실용적 접근

> Tell me and I forget. Teach me and I remember. Involve me and I learn.
Benjamin Franklin
말하면 잊을 것이다. 가르치면 기억할 것이다. 체험하면 배울 것이다.
벤자민 프랭클린

One of the common applications of Deep Learning includes pattern recognition. Therefore, in the same way as when you start programming there is sort of a tradition to start printing “Hello World”, in Deep Learning a model for the recognition of handwritten digits is usually constructed[1].
딥러닝의 응용 분야 중 하나는 패턴 인식이다. 프로그래밍을 처음 배우기 시작할 때 "Hello World" 를 출력하는 것으로 시작하는 일종의 전통이 있는 것처럼, 딥러닝을 처음 공부할 때는 필기체 숫자를 인식하기 위한 모델을 생성하는 것으로 시작한다.

The first example of a neural network that I will provide, will also allow me to introduce this new technology called TensorFlow.
신경망의 첫 번째 예제를 통해 TensorFlow라는 새로운 기술을 소개할 수 있을 것이다.

However, I do not intend to write a research book on Machine Learning or Deep Learning, I only want to make this new Machine Learning’s package, TensorFlow, available to everybody, as soon as possible.
하지만 머신 러닝이나 딥러닝에 대한 연구 도서(reaserch book)을 쓰려고 하지는 않는다. 단지 TensorFlow라는 새로운 머신 러닝 패키지를 가능한 모든 사람이 사용할 수 있기를 원한다.

Therefore I apologise in to my fellow data scientists for certain simplifications that I have allowed myself in order to share this knowledge with the general reader.
따라서 일반 독자들과 이런 지식을 공유하기 위해 일정 부분 내용을 단순화한 것에 대해 동료 데이터 과학자에게 사과한다.

The reader will find here the regular structure that I use in my classes; that is inviting you to use your computer’s keyboard while you learn. We call it “learn by doing“, and my experience as a professor at UPC tells me that it is an approach that works very well with engineers who are trying to start a new topic.
독자는 내가 강의에서 사용한 일반적인 형식을 발견할 수 있을 것이다; 그것은 학습하는 동안 각자의 컴퓨터에 있는 키보드를 사용하는 것이다. 우리는 그것을 "체험 학습"(learn by doing) 이라고 부른다. 체험 학습은 새로운 주제를 공부하려는 엔지니어에게 적합한 방법이다.

For this reason, the book is of a practical nature, and therefore I have reduced the theoretical part as much as possible. However certain mathematical details have been included in the text when they are necessary for the learning process.
이런 이유로, 본 도서는 실용적인 성격의 책이며 가능한 이론적인 부분의 설명은 최소화했다. 다만, 학습 과정에서 필요한 경우 수식에 대한 상세 설명을 포함했다.

I assume that the reader has some basic underestanding of Machine Learning, so I will use some popular algorithms to gradually organize the reader’s training in TensorFlow.
독자는 머신 러닝(machine learning)에 대해 기본적인 개념을 이해하고 있다고 가정한다. 따라서 TensorFlow에 대한 단계적 훈련 구성을 위해 유명한 알고리즘을 사용할 것이다.

In the first chapter, in addition to an introduction to the scenario in which TensorFlow will have an important role, I take the opportunity to explain the basic structure of a TensorFlow program, and explain briefly the data it maintains internally.
1 장에서는 TensorFlow의 중요 역할에 대한 시나리오와 기본 구조에 대해 설명한다. 또한 내부적으로 사용하는 데이터에 대해 대략적으로 소개한다.

In chapter two, through an example of linear regression, I will present some code basics and, at the same time, how to call various important components in the learning process, such as the cost function or the gradient descent optimization algorithm.
2 장에서는 선형 회귀(linear regression) 예제를 통해 기본 코드를 살펴본다. 아울러 비용 함수(cost function), 경사 하강 최적화 알고리즘(gradient descent optimization algorithm) 같은 학습 단계(learning process)에서 사용할 수 있는 다양한 구성 요소(components)를 호출하는 방법을 설명한다.

In chapter three, where I present a clustering algorithm, I go into detail to present the basic data structure of TensorFlow called tensor, and the different classes and functions that the TensorFlow package offers to create and manage the tensors.
3 장에서는 군집화 알고리즘(clustering algorithm)을 설명하면서 TensorFlow의 기본 자료 구조인 tensor에 대해 자세히 살펴본다. 또한 tensor를 생성하고 관리하기 위해 TensorFlow 패키지가 제공하는 다양한 클래스와 함수도 살펴본다.

In chapter four, how to build a neural network with a single layer to recognize handwritten digits is presented in detail. This will allow us to sort all the concepts presented above, as well as see the entire process of creating and testing a model.
4 장에서는 필기체 숫자(handwritten digits)를 인식하기 위한 단일 신경망을 구성하는 방법을 상세히 설명한다. 이번 장을 통해 앞에서 설명한 모든 개념을 잘 정리할 수 있을 뿐만 아니라 모델을 생성하고 테스트하는 전과정을 볼 수 있을 것이다.

The next chapter begins with an explanation based on neural network concepts seen in the previous chapter and introduces how to construct a multilayer neural network to get a better result in the recognition of handwritten digits. What it is known as convolutional neural network will be presented in more detail.
5 장은 이전 장에서 살펴 본 신경망 개념에 대한 설명과 함께 시작한다. 필기체 숫자 인식 성능을 개선하기 위해 다층 신경망을 생성하는 방법을 소개한다. convolutional neural network 라고 알려진 방법이 무엇인지 보다 상세하게 살펴 본다.

In chapter six we look at a more specific issue, probably not of interest to all readers, harnessing the power of calculation presented by GPUs. As introduced in chapter 1, GPUs play an important role in the training process of neural networks.
6 장에서는 아마 모든 독자가 관심이 있지는 않겠지만 GPU를 이용해 연산 속도를 향상하는 좀 더 특정한 주제를 살펴본다. 1 장에서 소개한 것 처럼 GPU는 신경망의 훈련 단계(trainning process)에서 중요한 역할을 한다.

The book ends with closing remarks, in which I highlight some conclusions. I would like to emphasize that the examples of code in this book can be downloaded from the github repository of the book[2].
본 도서는 몇 가지 결론을 강조하면서 마무리 한다. 본 도서의 예제 소스 코드는 github에서 다운 받을 수 있음을 강조하고 싶다.

