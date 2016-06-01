번역자 : 신성진(sungjin7127@gmail.com)

4. SINGLE LAYER NEURAL NETWORK IN TENSORFLOW

In the preface, I commented that one of the usual uses of Deep Learning includes pattern recognition. Given that, in the same way that beginners learn a programming language starting by printing “Hello World” on screen, in Deep Learning we start by recognizing hand-written numbers.

이번 서문에서는, 패턴 인식을 포함한 딥러닝의 대표적인 사용사례에 대해 소개 하겠다.  초보자들이 처음 프로그래밍 언어를 배울때 "Hello, World"를 화면에 출력하는 것과 같이, 딥러닝은 손글씨로 쓰여진 숫자를 인식하는 것으로 시작한다.

In this chapter I present how to build, step by step, a neural network with a single layer in TensorFlow. This neural network will recognize hand-written digits, and it’s based in one of the diverse examples of the beginner’s tutorial of Tensor-Flow[27].

이번 챕터에서는, 텐서플로우 상에서 신경망을 단계별로 빌드하는 방법을 설명하고자 한다. 많은 예제들 중에서, 텐서플로우[27] 사이트에 나와있는 입문자를 위한 숫자 손글씨 예제를 활용할 것이다. 우리가 구축하는 신경망은 이 손글씨에 대한 숫자를 인식하는 용도이다.

Given the introductory style of this book, I chose to guide the reader while simplifying some concepts and theoretical justifications at some steps through the example.

이 책은 입문서의 용도로 제작되었기 때문에, 예제에 나온 개념들과 이론적 증명들을 단계별로 예제를 통해 독자들에게 간단하게 가이드 하려고 한다.

If the reader is interested in learn more about the theoretical concepts of this example after reading this chapter, I suggest to read Neural Networks and Deep Learning [28], available online, presenting this example but going in depth with the theoretical concepts.

만약 독자들 중에서, 이번 장을 읽은 후 이론적인 개념들을 배우는 것에 흥미가 있다면, Neural Networks and Deep Learning [28]이란 책을 읽도록 권하겠다. 이 책은 온라인에 공개 되 있으며, 이번 예제에 심화된 이론적 개념들이 추가 된 내용들이 있기 때문이다.


The MNIST data-set

The MNIST data-set is composed by a set of black  and white images containing hand-written digits, containing more than 60.000 examples for training a model, and 10.000 for testing it. The MNIST data-set can be found at the MNIST database[29].

MNIST 데이터 셋은 손글씨 숫자들이 흑백 이미지로 구성되 있으며, 학습 모델을 위한 60,000개의 예제들 및 검증을 위한 10,000개 이상의 예제로 구성되어 있다.  MNIST 데이터셋은 MNIST 데이터베이스[29]에서 찾을 수 있다.


This data-set is ideal for most of the people who begin with pattern recognition on real examples without having to spend time on data pre-processing or formatting, two very important steps when dealing with images but expensive in time.

이 데이터 셋은 패턴인식을 시작하는 사람들에게 적합한 데이터이다. 이 데이터셋은 보통 이미지 데이터를 다룰 때, 중요하지만 시간 소모량이 큰 2가지 단계인 전처리와 형식 통일을 할 필요가 없다.

The black and white images (bilevel) have been normalized into 20×20 pixel images, preserving the aspect ratio. For this case, we notice that the images contain gray pixels as a result of the anti-aliasing [30] used in the normalization algorithm (reducing the resolution of all the images to one of the lowest levels). After that, the images are centered in 28×28 pixel frames by computing the mass center and moving it into the center of the frame. The images are like the ones shown here:

흑백 이미지 (이중 레벨)는 형상비를 유지하면서 20 × 20 픽셀의 이미지로 정규화 되어있다. 이 경우에는 정규화 알고리즘에 활용되는 엘리어싱 제거 [30] (모든 이미지의 해상도를 최저 이미지의 해상도로 감소)를 적용한 결과로 회색 픽셀들이 포함되 있다.  이후에 이미지들은 질량의 중심을 계산하고 프레임의 중심으로 이동하는 방법을 활용하여 28 × 28 픽셀 프레임에 중앙화 하였다. 처리된 이미지들은 아래와 같은 방식으로 표현 되어있다.

image034

Also, the kind of learning required for this example is supervised learning; the images are labeled with the digit they represent. This is the most common form of Machine Learning.

이번 예제에서 요구되는 학습방법은 지도학습이다. 각각의 이미지들은 의미하는게 무엇인지 숫자로 표기되있다. 이 학습법은 머신러닝에서 가장 많이 활용되는 형태 중 하나이다.

In this case we first collect a large data set of images of numbers, each labelled with its value. During the training, the model is shown an image and produces an output in the form of a vector of scores, one score for each category. We want the desired category to have the highest score of all categories, but this is unlikely to happen before training.

먼저, 각각 표기가 되어 있는 숫자 이미지들의 대량의 데이터 셋을 수집한다. 데이터 학습 중에는, 모델이 이미지와 결과물을 도출해 낼 것이다. 이 결과물들은 각각의 점수를 카테고리별로 나누어진 백터 형태의 점수형태로 구성되어 있다. 전체 카테고리에서 원하는 카테고리가 가장 높은 점수를 얻기를 원하지만, 훈련 전에는 발생할 가능성이 높지 않다.

We compute an objective function that measures the error (as we did in previous chapters) between the output scores and the desired pattern of scores. The model then modifies its internal adjustable parameters , called weights, to reduce this error. In a typical Deep Learning system, there may be hundreds of millions of these adjustable weights, and hundreds of millions of labelled examples with which to train the machine. We will consider a smaller example in order to help the understanding of how this type of models work.

우선, 결과 점수와 예상 패턴 점수 값들의 오류를 나타내는 목적함수를 계산한다. 그 다음, 오류를 감소 시키기 위해 내부 조정 가능한 파라미터(매개 변수) 값의 가중치(Weight)을 수정한다.  일반적인 딥러닝 시스템에서는 기계 (컴퓨터) 에 학습시키기 위해, 아마도 수억개의 조정 가능한 가중치들과 수억개의 레이블이 있는 예제들을 보유하고 있을 것이다.  필자는 이해를 돕기 위하여 이러한 종류의 모델들이 어떻게 작동하는지 알기 위해 소규모의 예제들을 사용하였다.

To download easily the data, you can use the script input_data.py [31], obtained from Google’s site [32] but uploaded to the book’s github for your comodity. Simply download the code input_data.py in the same work directory where you are programming the neural network with TensorFlow. From your application you only need to import and use in the following way:

Google 사이트 [32]에서 input_data.py [31] 스크립트를 활용하면 쉽게 데이터를 받을 수 있다. 물론, 편의성을 위해서 필자의 github에 업로드도 해 놓았다.  신경망을 사용한 텐서플로우로 프로그래밍을 한 작업 디렉토리에 input_data.py를 다운로드 하기만 하면 된다. 그 이후, 어플리케이션에서 아래와 같은 방법으로 import 하기만 하면 된다:

```
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

After executing these two instructions you will have the full training data-set in mnist.train and the test data-set in mnist.test. As I previously said, each element is composed by an image, referenced as “xs”, and its corresponding label “ys”, to make easier to express the processing code. Remember that all data-sets, training and testing, contain “xs” and “ys”; also, the training images are referenced in mnist.train.images and the training labels in mnist.train.labels.

위에 나온 2단계의 설명을 따르면, 전체 학습 데이터셋인 mnist.train과 테스트 데이터셋인 mnist.test를 보유하고 있다. 언급한대로, 각각의 요소들은 "xs"는 구성되어 있는 이미지, "ys"는 프로세싱 코드의 표현을 좀 더 쉽게 하기 위한 이미지들의 래이블로 표현된다.  모든 데이터셋, 학습 및 검증 과정들은 "xs"나 "ys"를 포함하고 있고, 학습 이미지는 mnist.train.images에 학습 래이블은 mnist.train.labels에 있다.

As previously explained, the images are formed by 28×28 pixels, and can be represented as a numerical matix. For example, one of the images of number 1 can be represented as:

이전에 설명했던 대로, 이미지들은 28*28 픽셀들로 이루어져 있고, 각각 행렬의 수치값으로 표현 될 수 있다. 예를 들면, 숫자 1을 표현하는 이미지 중 하나는 아래와 같이 표현 할 수 있다:

image036

Where each position indicates the level of lackness of each pixel between 0 and 1. This matrix can be represented as an array of 28×28 = 784 numbers. Actually, the image has been transformed in a bunch of points in a vectorial space of 784 dimensions. Only to mention that when we reduce the structure to 2 dimensions, we can be losing part of the information, and for some computer vision algorithms this could affect their result, but for the simplest method used in this tutorial this will not be a problem.

이미지에 표현된 각 위치의 0 과 1 사이의 픽셀 값은, 명암의 정도로 표시되어 있다. 또한, 전체 이미지의 행렬은 28×28 = 784의 배열로 표현된다. 실질적으로 이 이미지는 784 차원의 백터 공간 값 안에 있는 수 많은 점들로 변환 되 있다.  만약에 이 이미지를 2차원 구조로 축소시킨다면, 일부 관련 정보들을 잃을 수도 있다.  몇몇 컴퓨터 영상 알고리즘에서는 결과에도 영향을 미치지만, 이번 튜토리얼은 간단한 접근법이기 때문에 문제가 되지 않는다.


Summarizing, we have a tensor mnist.train.images in 2D in which calling the fuction get_shape() indicates its shape:

요약하자면, TensorShape([Dimension(60000), Dimension(784)])으로 구성된 get_shape()의 함수를 포함 한 2D 텐서인 mnist.train.images를 보유 한 것이다.


```
TensorShape([Dimension(60000), Dimension(784)])
```

The first dimension indexes each image and the second each pixel in each image. Each element of the tensor is the intensity of each pixel between 0 and 1.

첫번째 차원은 이미지 각각에 대한 인덱스이며, 두번째 차원은 이미지 안의 픽셀을 나타낸다. 텐서의 각각의 요소들은 0과 1사이의 화소의 강도 값 (백:0, 흑:1) 으로 이루어져 있다.

Also, we have the labels in the form of numbers between 0 and 9, indicating which digit each image represents. In this example, we are representing labels as a vector of 10 positions, which the corresponding position for the represented number contains a 1 and the rest 0. So mnist.train.labelses is a tensor shaped as TensorShape([Dimension(60000), Dimension10)]).

또한, 각각의 이미지를 어떤 숫자인지 표현하기 위해 0에서 9까지 숫자에 라벨을 표기 하였다. 이번 예제에서는, 숫자 위치의 포함 여부에 따라 1이나 그 외에는 0으로 표현 된 10가지 위치들로 구성된 백터값을 라벨화 했다. 위의 사항들로 인해, 전체 구성도를 보면, mnist.train.labels은 TensorShape([Dimension(60000), Dimension10)])으로 이루어져 있다.


An artificial neutron

인공 신경망


Although the book doesn’t focus on the theoretical concepts of neural netwoks, a brief and intuitive introduction of how neurons work to learn the training data will help the reader to undertand what is happening. Those readers that already know the theory and just seek how to use TensorFlow can skip this section.
Let’s see a simple but illustrative example of how a neuron learns. Suppose a set of points in a plane labeled as “square” and “circle”. Given a new point “X”, we want to know which label corresponds to it:

이 책은 신경망의 이론적 개념에 집중하고 있지 않지만, 독자들의 이해를 돕기 위하여 간단하고 쉽게 신경망이 어떻게 학습 데이터를 학습하는 과정에 대해서 소개하겠다. 만약 신경망에 대한 이론을 알고 있는 독자들은 바로 Tensorflow 설명 부분으로 넘어가는 것을 추천한다.
신경망이 어떻게 학습하는 것에 대한 간단한 삽화와 함께 확인해보자. 2차원 평면에 "사각형"과 "원"으로 레이블 된 점들이 있다고 가정해보자. 새로운 포인트 "X"가 주어졌을 때, 어떤 레이블이 연관 되었는지 알고 싶다.


Screen Shot 2016-02-16 at 09.30.14

A usual approximation could be to draw a straight line dividing the two groups and use it as a classifier:

일반적인 근사치로 표현하면, 그룹을 2개로 나누는 선을 긋고 이 것을 분류의 기준으로 활용한다.

Screen Shot 2016-02-16 at 09.30.09


In this situation, the input data is represented by vectors shaped as (x,y) representing the coordinates in this 2-dimension space, and our function returning ‘0’ or ‘1’ (above or below the line) to know how to classify it as a “square” or “circle”. Mathematically, as we learned in the linear regression chapter, the “line” (classifier) can be expressed as y= W*x+b.

이 상황에서는 입력 데이터는 2차원 공간의 (x,y)로 표현되는 백터 값으로 표현된다. 또한, "사각형" 인지 "원"인지 분류되는 과정을 알기 위해 '0' 또는 '1' (직선의 윗부분과 아랫부분)으로 나누어진다.  선형회귀 장에서 배운 것처럼, 수학적으로 "선" (분류기)는 y = W*x + b로 표현된다.

Generalizing, a neuron must learn a weight W (with the same dimension as the input data X) and an offset b (called bias in neural networks) to learn how to classify those values. With them, the neuron will compute a weighted sum of the inputs in X using weight W, and add the offset b; and finally the neuron will apply an “activation” non-linear function to produce the result of “0” or “1”.
The function of the neuron can be more formally expressed as:

일반적으로, 신경은 점들의 값을 분류하기 위해서 가중치인 W (입력 데이터 X와 같은 차원)와 오프셋 b (신경망에서는 bias라고 표현)들을 학습시켜야 한다.  이러한 사항들을 기반으로, 신경은 가중치 W를 사용하여 입력 데이터 X의 총 합을 구하고, 오프셋 b를 더한다.  마지막으로 신경은 결과값인 "0"과 "1"을 생성하기 위해 비선형 함수인 "활성화 (Activation)" 함수를 적용한다.  뉴런의 함수를 좀 더 형식적으로 표현하면 다음과 같다.

image043

Having defined this function for our neuron, we want to know how the neuron can learn those parameters W and b from the labeled data with “squares” and “circles” in our example, to later label the new point “X”.

위에 신경의 함수를 정의한 것을 활용하여, 신경이 어떻게 "사각형"과 "원"으로 레이블 된 데이터에서 생성된 파라메터 W와 b값으로 새로운 점인 "X"를 학습하는 과정을 예제로 통해 알아 볼 것이다.

A first approach could be similar to what we did with the linear regression, this is, feed the neuron with the known labeled data, and compare the obtained result with the real one. Then, when iterating, weights in W and b are adjusted to minimize the error, as shown again in chapter 2 for the linear regression line.

 시작은 선형회귀 때 접근했던 방법과 유사하다.  주어진 레이블 된 데이터를 뉴런에게 제공한 결과에 실제 값과 비교하는 것이다.  그 다음 이 과정을 반복하면서, 2장에서 회귀분석 선의 오류율을 최소화 한 것처럼 가중치인 W와 b 값을 조절한다.

Once we have the W and b parameters we can compute the weighted sum, and now we need the function to turn the result stored in z into a ‘0’ or ‘1’. There are several activation functions available, and for this example we can use a popular one called sigmoid [33], returning a real value between 0 and 1

W와 b의 파라메터 값을 사용해서 가중치의 합을 구하고 나면, z에 ‘0’과 ‘1’로 저장시킬 함수가 필요하다.  여러 종류의 함수가 있겠지만, 유명한 것들 중 0과 1사이의 실수값을 돌려주는 시그모이드 [33] 함수를 활용 하겠다.

image046

Looking at the formula we see that it will tend to return values close to 0 or 1. If input z is big enough and positive, “e” powered to minus z is zero and then y is 1. If the input z is big enough and negative, “e” powered to a large positive number becomes also a large positive number, so the denominator becomes large and the final y becomes 0. If we plot the function it would look like this:

위의 공식을 보면 0과 1의 가까운 값으로 수렴하는 경향을 보인다.  만약 z가 충분히 큰 양수이면, “e의 -z승 [e^(-z)]은 0에 가까워지고, y는 1이 된다. 만약에 입력값 z에 충분히 큰 음수라면, “e”
는 거대한 양수가 되고 부모는 커지므로 결국 0이 된다.  이 함수를 그려보면 아래와 같다.


image045

Since here we have presented how to define a neuron, but a neural network is actually a composition of neurons connected among them in a different ways and using different activation functions. Given the scope of this book, I’ll not enter into all the extension of the neural networks universe, but I assure you that it is really exciting.

우리는 하나의 신경에 대해서 정의하는 방법을 소개 했지만, 신경망은 사실상 다양한 방식으로 연결되어 있고 각기 다른 활성화 함수들을 활용한다. 이 책의 범위상, 신경망의 모든 면을 살펴 볼 수는 없지만, 매우 흥미로운 분야이다.
 
Just to mention that there is a specific case of neural networks (in which Chapter 5 is based on) where the neurons are organized in layers, in a way where the inferior layer (input layer) receives the inputs, and the top layer (output layer) produces the response values. The neural network can have several intermediate layers, called hidden layers. A visual way to represent this is:

몇가지 특별한 경우의 신경망들 (제 5장에 소개 될)이 있다. 이 신경망은 입력을 받는 하위 층 (Input layer)와 결과 값을 생성하는 상위 층 (output layer)으로 신경들이 각 층마다 구성 되어 있다. 신경망은 숨겨진 층 (Hidden Layer)이라 불리우는 층을 중간에 가질 수도 있다.  지금 사항들을 시각화 한다면 아래와 같다:


image049

In these networks, the neurons of a layer communicate to the neurons of the previous layer to receive information, and then communicate their results to the neurons of the next layer.

이와 같은 신경망들은, 신경들이 정보를 교환하기 위해, 전 층에서는 정보를 받고 다음 층에서는 그 결과 값들을 다음 신경들과 소통을 한다.

As previously said, there are more activation functions apart of the Sigmoid, each one with different properties. For example, when we want to classify data into more than two classes at the output layer, we can use the Softmax[34] activation function, a generalization of the sigmoid function. Softmax allows obtaining the probability of each class, so their sum is 1 and the most probable result is the one with higher probability.

앞서 이야기 한 것처럼, 시그모이드 이외에도 각각 다른 성질들을 가지고 있는 함수들이 많다.  예를 들면, 출력 층에 2가지 이상의 클래스 데이터들을 분류하고 싶을 때는 시그모이드 함수의 일반화 형태인 소프트맥스[34]라는 활성화 함수를 사용한다.  소프트맥스는 각 클래스의 합을 확률로 나타내고, 그 총합을 1로 만든다.  따라서, 이 함수는 가장 높은 가능성을 가진 클래스가 가장 높은 확률을 가지게 한다.

A easy example to start: Softmax

간단한 예제로 시작하는 소프트맥스


Remember that the problem to solve is that, given an input image, we get the probability that it belongs to a certain digit. For example, our model could predict a “9” in an image with an 80% certainty, but give a 5% of chances to be an “8” (due to a dubious lower trace), and also give certain low probabilities to be any other number. There is some uncertainty on recognizing hand-written numbers, and we can’t recognize the digits with a 100% of confidence. In this case, a probability distribution gives us a better idea of how much confidence we have in our prediction.

여기서 풀려고 하는 문제는, 입력 이미지가 주어지면 그것이 어떤 특정 숫자에 속하는 확률을 구하는 것을 기억하자.  예를들면, 우리 모델이 80%정도의 확률로 이미지가 “9”일 것을 예측하지만, 5%의 확률로 “8” (하단의 불확실한 모양으로 인한), 그리고 낮은 확률로 몇몇 다른 숫자를 줄 수도 있다.  손글씨를 예측하는 데는 약간의 불확실성이 있기 때문에, 우리는 숫자를 100% 예측 할 수 가 없다.  이런 경우에는 확률 분포가 예측에 대해 얼마나 신뢰성이 있는지 좋은 정보를 제공 한다.

So, we have an output vector with the probability distribution for the different output labels, mutully exclusive. This is, a vector with 10 probability values, each one corresponding to each digit from 0 to 9, and all probabilities summing 1.

그래서, 상호 배타적인 서로 다른 출력 레이블을 위한 확률 분포를 가진 출력 백터 가진다. 10개의 확률 값을 가진 이 벡터는 각각 0 에서 9까지의 숫자에 대응되고, 총 확률의 합은 1 이다.

As previously said, we get to this by using an output layer with the softmax activation function. The output of a neuron with a softmax function depends on the output of the other neurons of its layer, as all of their outputs must sum 1.

앞서 언급한 것처럼, 이 백터 값은 출력 층과 소프트맥스 활성화 함수를 사용하여 얻어진다.  소프트맥스 함수의 출력 신경은 또 다른 뉴런 층의 출력값에 영향을 미친다.  또한 총 출력 값의 합을 1이 되어야 한다.


The softmax function has two main steps: first, the “evidences” for an image belonging to a certain label are computed, and later the evidences are converted into probabilities for each possible label.

소프트맥스 함수는 2가지 주요 단계가 있는데, 첫번째는 이미지가 어떤 래이블에 속하는지 “근거”값을 계산하는 것과, 이 근거들을 활용하여 각 레이블에 대한 확률로 변환하는 과정이 있다.


Evidence of belonging

Measuring the evidence of a certain image to belong to a specific class/label, a usual approximation is to compute the weighted sum of pixel intensities. That weight is negative when a pixel with high intensity happens to not to be in a given class, and positive if the pixel is frequent in that class.

Let’s show a graphical example: suppose a learned model for the digit “0” (we will see how this is learned later). At this time, we define a model as “something” that contains information to know whether a number belongs to a specific class. In this case, we chose a model like the one below, where the red (or bright gray for the b/n edition) represents negative examples (this is, reduce the support for those pixels present in “0”), while the blue (the darker gray for b/n edition) represents the positive examples. Look at it:

image050

Think in a white paper sheet of 28×28 pixels and draw a “0” on it. Generally our zero would be drawn in the blue zone (remember that we left some space around the 20×20 drawing zone, centering it later).

It is clear that in those cases where our drawing crosses the red zone, it is most probably that we are not drawing a zero. So, using a metric based in rewarding those pixels, stepping on the blue region and penalizing those stepping the red zone seems reasonable.

Now think of a “3”: it is clear that the red zone of our model for “0” will penalize the probabilities of it being a “0”. But if the reference model is the following one, generally the pixels forming the “3” will follow the blue region; also the drawing of a “0” would step into a red zone.

image052

I hope that the reader, seeing those two examples, understands how the explained approximation allows us to estimate which number represent those drawings.
The following figure shows an example of the ten different labels/classes learned from the MNIST data-set (extracted from the examples of Tensorflow[35]). Remember that red (bright gray) represents negative weights and blue (dark gray) represents the positive ones.

 

image054

In a more formal way, we can say that the evidence for a class i given an input x is expressed as:

image056

Where i indicates the class (in our situation, between 0 and 9), and j is an index to sum the indexes of our input image. Finally Wi represents the aforementioned weights.

Remember that, in general, the models also include an extra parameter representing the bias, adding some base uncertainty. In our situation, the formula would end like this:

image058

For each i (between 0 and 9) we have a matrix Wi of 784 elements (28×28), where each element j is multiplied by the corresponding component j of the input image, with 784 components, then added bi. A graphical view of the matrix calculus and indexes is this:

image061

Probability of belonging

We commented that the second step consisted on computing probabilities. Specifically we turn the sum of evidences into predicted probabilities, indicated as y, using the softmax function:

image062

Remember that the output vector must be a probability function with sum equals 1. To normalize each component, the softmax function uses the exponential value of each of its inputs and then normalizes them as follows:

image064

The obtained effect when using exponentials is a multiplication effect in weights. Also, when the evidence for a class is small, this class support is reduced by a fraction of its previous weight. Furthermore, softmax normalizes the weights making them to sum 1, creating a probability distribution.

The interesting fact of such function is that a good prediction will have one output with a value near 1, while all the other outputs will be near zero; and in a weak prediction, some labels may show similar support.

Programming in TensorFlow

After this brief description of what the algorithm does to recognize digits, we can implement it in TensorFlow. For this, we can make a quick look at how the tensors should be to store our data and the parameters for our model. For this purpose, the following schema depicts the data structures and their relations (to help the reader recall easily each piece of our problem):

image066First of all we create two variables to contain the weights W and the bias b:

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
Those variables are created using the tf.Variable function and the initial value for the variables; in this case we initialize the tensor with a constant tensor containing zeros.

We see that W is shaped as [Dimension(784), Dimension(10)], defined by its argument, a constant tensor tf.zeros[784,10] dimensioned like W. The same happens with the bias b, shaped by its argument as [Dimension(10)].

Matrix W has that size because we want to multiply the image vector of 784 positions for each one of the 10 possible digits, and produce a tensor of evidences after adding b.

In this case of study using MNIST, we also create a tensor of two dimensions to keep the information of the x points, with the following line of code:

x = tf.placeholder("float", [None, 784])
The tensor x will be used to store the MNIST images as a vector of 784 floating point values (using None we indicate that the dimension can be any size; in our case it will be equal to the number of elements included in the learning process).

Now we have the tensors defined, and we can implement our model. For this, TensorFlow provides several operations, being tf.nn.softmax(logits, name=None) one of the available ones, implementing the previously described softmax function. The arguments must be a tensor, and optionally, a name. The function returns a tensor of the same kind and shape of the tensor passed as argument.

In our case, we provide to this function the resulting tensor of multiplying the image vector x and the weight matrix W, adding b:

y = tf.nn.softmax(tf.matmul(x,W) + b)
Once specified the model implementation, we can specify the necessary code to obtain the weights for W and bias b using an iterative training algorithm. For each iteration, the training algorithm gets the training data, applies the neural network and compares the obtained result with the expected one.

To decide when a model is good enough or not we have to define what “good enough” means. As seen in prevous chapters, the usual methodology is to define the opposite: how “bad” a model is using cost functions. In this case, the goal is to obtain values of W and b that minimizes the function indicating how “bad” the model is.

There are different metrics for the degree of error between resulting outputs and expected outputs on training data. One common metric is the mean squared error or the squared Euclidean distance, which have been previously seen. Even though, some research lines propose other metrics for such purpose in neural networks, like the cross entropy error, used in our example. This metric is computed like this:

image068

Where y is the predicted distribution of probability and y’ is the real distribution, obtained from the labeling of the training data-set. We will not enter into details of the maths behind the cross-entropy and its place in neural networks, as it is far more complex than the intended scope of this book; just indicate that the minimum value is obtained when the two distributions are the same. Again, if the reader wants to learn the insights of this function, we recommend reading Neural Networks and Deep Learning [36].

To implement the cross-entropy measurement we need a new placeholder for the correct labels:

y_ = tf.placeholder("float", [None,10])
Using this placeholder, we can implement the cross-entropy with the following line of code, representing our cost function:

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
First, we calculate the logarithm of each element y with the built-in function in TensorFlow tf.log(), and then we multiply it for each y_ element. Finally, using tf.reduce_sum we sum all the elements of the tensor (later we will see that the images are visited in bundles, and in this case the value of cross-entropy corresponds to the bundle of images y and not a single image).

Iteratively, once determined the error for a sample, we have to correct the model (modifying the parameters W and b, in our case) to reduce the difference between computed and expected outputs in the next iteration.

Finally, it only remains to specify this iterative minimization process. There are several algorithms for this purpose in neural networks; we will use the backpropagation (backward propagation of errors) algorithm, and as its name indicates, it propagates backwards the error obtained at the outputs to recompute the weights of W, especially important for multi-layer neural network.

This method is used together with the previously seen gradient descent method, which using the cross-entropy cost function allows us to compute how much the parameters must change on each iteration in order to reduce the error using the available local information at each moment. In our case, intuitively it consist on changing the weights W a little bit on each iteration (this little bit expressed by a learning rate hyperparameter, indicating the speed of change) to reduce the error.

Due that in our case we only have one layer neural network we will not enter into backpropagation methods. Only remember that TensorFlow knows the entire computing graph, allowing it to apply optimization algorithms to find the proper gradients for the cost function to train the model.

So, in our example using the MNIST images, the following line of code indicates that we are using the backpropagation algorithm to minimize the cross-entropy using the gradient descent algorithm and a learning rate of 0.01:

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
Once here, we have specified all the problem and we can start the computation by instantiating tf.Session() in charge of executing the TensorFlow operations in the available devices on the system, CPUs or GPUs:

sess = tf.Session()
Next, we can execute the operation initializing all the variables:

sess.run(tf.initialize_all_variables())
From this moment on, we can start training our model. The returning parameter for train_step, when executed, will apply the gradient descent to the involved parameters. So training the model can be achieved by repeating the train_step execution. Let’s say that we want to iterate 1.000 times our train_step; we have to indicate the following lines of code:

for i in range(1000):
   batch_xs, batch_ys = mnist.train.next_batch(100)
   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
The first line inside the loop specifies that, for each iteration, a bundle of 100 inputs of data, randomly sampled from the training data-set, are picked. We could use all the training data on each iteration, but in order to make this first example more agile we are using a small sample each time. The second line indicates that the previously obtained inputs must feed the respective placeholders.

Finally, just mention that the Machine Learning algorithms based in gradient descent can take advantage from the capabilities of automatic differentiation of TensorFlow. A TensorFlow user only has to define the computational architecture of the predictive model, combine it with the target function, and then just add the data.

TensorFlow already manages the associated calculus of derivatives behind the learning process. When the minimize() method is executed, TensorFlow identifies the set of variables on which loss function depends, and computes gradients for each of these. If you are interested to know how the differentiation is implemented you can inspect the ops/gradients.py file [37].

Model evaluation

A model must be evaluated after training to see how much “good” (or “bad”) is. For example, we can compute the percentage of hits and misses in our prediction, seeing which examples were correctly predicted. In previous chapters, we saw that the tf.argmax(y,1) function returns the index of the highest value of a tensor according a given axis. In effect, tf.argmax(y,1) is the label in our model with higher probability for each input, while tf.argmax(y_,1) is the correct label. Using tf.equal method we can compare if our prediction coincides with the correct label:

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
This instruction returns a list of Booleans. To determine which fractions of predictions are correct, we can cast the values to numeric variables (floating point) and do the following operation:

```
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```

For example, [True, False, True, True] will turn into [1,0,1,1] and the average will be 0.75 representing the percentage of accuracy. Now we can ask for the accuracy of our test data-set using the mnist.test as the feed_dict argument:

```
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
```

I obtained a value around 91%. Are these results good? I think that they are fantastic, because this means that the reader has been able to program and execute her first neural network using TensorFlow.

Another problem is that other models may provide better accuracy, and this will be presented in the next chapter with a neural network containing more layers.

The reader will find the whole code used in this chapter in the file RedNeuronalSimple.py, in the book’s github [38]. Just to provide a global view of it, I’ll put it here together:

```
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

matm=tf.matmul(x,W)
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
```

[contents link]
