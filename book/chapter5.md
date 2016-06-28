5. MULTI-LAYER NEURAL NETWORKS IN TENSORFLOW

In this chapter I will program, with the reader, a simple Deep Learning neural network using the same MNIST digit recognition problem of the previous chapter.
이 장에서 나는 독자, 이전 장에 다룬 바 있는 MNIST 숫자 인식 문제를 이용하여 간단한 형태의 딥러닝 신경망을 프로그래밍 하고자 한다.

As I have advanced, a Deep Learning neural network consists of several layers stacked on top of each other. Specifically, in this chapter we will build a convolutional network, this is, an archetypal example of Deep Learning. Convolution neural networks were introduced and popularized in 1998 by Yann LeCunn and others. These convolutional networks have recently led state of the art performance in image recognition; for example: in our case of digit recognition they achieve an accuracy higher than 99%.
이전에 언급한 바 있지만, 딥러닝 신경망은 겹겹이 쌓인 여러 층의 레이어(layer)로 이루어져있다. 이번 장에서 우리는 딥러닝의 전형적인 예로  컨볼루션 신경망을 구축할 것이다. 컨볼루션 신경망을 도입 Yann LeCunn을 비롯한 여러 연구자들에 의해 1998 년에 대중화되었다. 이 컨볼루션 신경망은 최근 이미지 인식에서 최고의 성능을 보이고 있다; 예를 들면 : 숫자 인식의 우리의 경우에 그들은 99 % 이상의 정확도 높은 얻을 수 있습니다.

In the rest of this chapter, I will use an example code as the backbone, alongside which I will explain the two most important concepts of these networks: convolutions and pooling without entering in the details of the parameters, given the introductory nature of this book. However, the reader will be able to run all the code and I hope that it will allow you understand to global ideas behind convolutional networks.
이 책의 소개 자연 주어진 매개 변수의 세부 사항을 입력하지 않고 convolutions 및 pooling이 장의 나머지 부분에서 나는 가장 중요한 두 이러한 네트워크의 개념을 설명 할 함께 백본, 같은 예제 코드를 사용합니다 . 그러나 독자는 모든 코드를 실행할 수 있습니다 나는 당신이 컨볼루션 신경망 뒤에 글로벌 아이디어를 이해할 수 있기를 바랍니다.

* Convolutional Neural Networks
* 컨볼루셔널 뉴럴 네트웍

Convolutional Neural Nets (also known as CNN’s or CovNets) are a particular case of Deep Learning and have had a significant impact in the area of computer vision.
콘볼루셔널 뉴럴 네트웍(또한 CNN 나 CovNets라고도 함)은 딥 러닝의 특별한 경우이며, 컴퓨터 비전 분야에 큰 영향을 주었다.

A typical feature of CNN’s is that they nearly always have images as inputs, this allows for more efficient implementation and a reduction in the number of required parameters. Let’s have a look at our MNIST digit recognition example: after reading in the MNIST data and defining the placeholders using TensorFlow as we did in the previous example:
CNN의 전형적인 기능들은 거의 항상 입력으로 이미지를 가진다는 점인데, 이것은 요구되는 매개변수의 수에서 더 효율적인 구현과 매개변수의 적은 수를 통하여 동작을 허용한다는 점이다. 우리의 MNIST 숫자 인식 예제를 살펴보기로 하자: MNIST데이터에서 읽고, 이전 예제에서 했던 것처럼 텐서플로우를 사용하여 플레이스홀더(Placeholders)를 정의하자.

```
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
```

We can reconstruct the original shape of the images of the input data. We can do this as follows:
우리는 입력 데이터 이미지의 원래의 형상을 재구성 할 수있다. 다음과 같이 우리는 이 작업을 수행 할 수 있다 :

```
x_image = tf.reshape(x, [-1,28,28,1])
```

Here we changed the input shape to a 4D tensor, the second and third dimension correspond to the width and the height of the image while the last dimension corresponding number of color channels, 1 in this case.
여기서, 입력데이터를 4차원 텐서로 변환했다. 이때 두번째와 세번째 차원은 이미지의 넓이와 높이와 동일하며, 마지막 차원은 색상 채널들의 수와 동일한데, 이 경우에서는 1이다. 

"""This way we can think of the input to our neural network being a 2 dimensional space of neurons with size of 28×28 as depicted in the figure.
우리는 그림에 나타낸 바와 같이 우리의 신경 네트워크는 28 × 28의 크기를 가진 뉴런들(neurons)의 2 차원 공간에 존재하는 우리의 신경망으로 생각할 수 있다. 인에 입력 생각할 수있는이 방법."""

image072


There are two basic principles that define convolution neural networks: the filters and the characteristic maps. These principles can be expressed as groups of specialized neurons, as we will see shortly. But first, we will give a short description of these two principles given their importance in CNN’s.
컨볼루션 뉴럴 네트워크를 정의하는 두 가지 기본 원칙이 있다:필터 및 특성 맵. 이러한 원칙들은 특수화된 뉴런의 그룹으로 표현 될 수 있다. 우리는 이를 짧게나마 볼 것이다. 하지만 먼저, 우리는 CNN의 중요한 이들 두 가지 원칙에 대한 간단한 설명을 제공 할 것이다.

Intuitively, we could say that the main purpose of a convolutional layer is to detect characteristics or visual features in the images, think of edges, lines, blobs of color, etc. This is taken care of by a hidden layer that is connected by to input layer that we just discussed. In the case of CNN’s, in which we are interested, the input data is not fully connected to the neurons of the first hidden layer; this only happens in a small localized space in the input neurons that store the pixels values of the image. This can be visualized as follows:
직관적으로, 우리는 컨볼루셔널 레이어(Convolutional Layer)의 주요 목적은, 이미지의 특성이나 시각적 특징을 검출이에 의해 연결되어 숨겨진 레이어에 의해 처리되는 등 가장자리, 선, 색의 얼룩, 생각하는 것을 말할 수 있습니다 우리가 논의 입력 층. CNN의 우리가 관심이있는 경우, 입력 데이터는 제 은닉층의 뉴런 완전히 연결되지; 이것은 이미지의 픽셀 값을 저장하는 입력 뉴런의 작은 국부적 공간에서 일어난다. 이는 다음과 같이 시각화 될 수있다 :

image074

To be more precise, in the given example each neuron of our hidden layer is connected with a small 5×5 region (hence 25 neurons) of the input layer.
더 정확하게 하기 위해서, 주어진 예제에서 우리의 은닉층(hidden layer)의 각 뉴런은 입력 층(input layer)의 작은 5 × 5 영역 (따라서 25 뉴런)에 접속되어있다.

We can think of this being a window of size 5×5 that passes over the entire input layer of size 28×28 that contains the input image. The window slides over the entire layer of neurons. For each position of the window there is a neuron in the hidden layer that processes that information.
우리는이 입력 이미지를 포함하는 28 × 28 크기의 전체 입력 층 통과 크기 5 × 5 윈도우 인 생각할 수있다. 창은 뉴런의 전체 층 위에 슬라이드. 윈도우의 각 위치 정보를 처리 은닉층의 뉴런있다.

We can visualize this by assuming that the window starts in the top left corner of the image; this provides the information to the first neuron of the hidden layer.  The window is then slid right by one pixel; we connect this 5×5 region with the second neuron in the hidden layer. We continue like this until the entire space from top to bottom and from left to right has been convered by the window.
우리는 창 이미지의 왼쪽 상단에서 시작한다고 가정하여이를 시각화 할 수 있습니다; 이것은 숨겨진 레이어의 첫 번째 뉴런에 정보를 제공한다. 창은 하나의 픽셀에 의해 오른쪽으로 슬라이드; 우리는 숨겨진 레이어에서 두 번째 신경 세포와이 5 × 5 영역을 연결합니다. 우리는 위에서 아래로 전체 공간까지 이런 식으로 계속 왼쪽에서 오른쪽 창에 의해 변환 된 것되었습니다.

image076

Analyzing the concrete case that we have proposed, we observe that given an input image of size 28×28 and a window of size 5×5 leads to a 24×24 space of neurons in the first hidden layer due to the fact we can only move the window 23 times down and 23 times to the right before hitting the bottom right edge of the input image. This assumes that window is moved just 1 pixel each time, so the new window overlaps with the old one expect in line that has just advanced.
우리가 제안 구체적인 사례를 분석, 우리는 사실에 기인 사이즈 28 × 28의 입력 화상과 상기 제 은닉층 뉴런의 24 × 24 공간 크기가 5 × 5의 리드 창 주어진 관찰 우리는 단지 수 입력 화상의 우측 하단 모서리를 치기 전에 창을 오른쪽 아래 23 번, 23 번 이동한다. 이 창은 단지 1 픽셀마다 이동하는 것으로 가정하기 때문에 새 창이 이전 단지 고급 라인에 기대와 겹칩니다.

It is, however, possible to move more than 1 pixel at a time in a convlution layer, this parameter is called the ‘stride’ length. Another extension is to pad the edges with zeroes (or other values) so that the window can slide over the edge of the image, which may lead to better results. The parameter to control this feature is know as padding [39], with which you can determine the size of the padding. Given the introductory nature of the book we will not go into further detail about these two parameters.
그것은 convlution 층을 한 번에 1 개 이상의 화소를 이동하지만, 가능하며,이 파라미터는 "스트라이드"길이라고한다. 윈도우가 더 좋은 결과로 이어질 수있는 이미지의 가장자리 위로 밀어 수 있도록 또 다른 확장 패드에 제로 (또는 다른 값)와 가장자리입니다. 이 기능을 제어하는​​ 파라미터는 패딩의 크기를 결정할 수있는 패딩 [39]으로 알고있다. 우리는이 두 가지 매개 변수에 대한 자세한 세부 사항으로 가지 않을 것이다 책의 소개 성격을 감안할 때.

Given our case of study, and following the formalism of the previous chapter, we will need a bias value b and a 5×5 weight matrix W to connect the neurons of the hidden layer with the input layer. A key feature of a CNN is that this weight matrix W and bias b are shared between all the neurons in the hidden layer; we use the same W and b for neurons in the hidden layer. In our case that is 24×24 (576) neurons. The reader should be able to see that this drastically reduces the amount weight parameters that one needs when compared to a fully connected neural network. To be specific, this reduces from 14000 (5x5x24x24) to just 25 (5×5) due to the sharing of the weight matrix W.
연구의 우리의 경우 감안할 때, 이전 장에서의 형식주의에 따라, 우리는 입력 층과 은닉층의 뉴런을 연결하는 바이어스 값 b를 5 × 5 중량 행렬 W가 필요합니다. CNN과의 주요 기능이 가중치 행렬 W와 바이어스 b는 은닉층의 모든 신경 세포간에 공유된다는 것이다; 우리는 은닉층 뉴런에 대해 동일한 W와 b를 사용합니다. 우리의 경우 그 24 × 24 (576) 뉴런이다. 독자는이 대폭 완전히 연결된 신경망에 비해 하나의 필요한 양 가중치 파라미터를 감소시키는 것으로 볼 수있을 것이다. 즉,이 14000 (5x5x24x24)에서 감소 단지 25 (5 × 5)으로 인해 가중치 매트릭스 (W)의 공유에

This shared matrix W and the bias b are usually called a kernel or filter in the context of CNN’s. These filters are similar to those used image processing programs for retouching images, which in our case are used to find discriminating features. I recommend looking at the examples found in the GIMP [40] manual to get a good idea on how process of convolution works.
이 공유 행렬 W 바이어스 B는 일반적 CNN의 컨텍스트에서 커널 또는 필터라고한다. 이 필터는 우리의 경우 식별 기능을 찾는 데 사용되는 리터칭 이미지에 대한 그 사용 이미지 처리 프로그램과 유사합니다. 나는 컨볼 루션 작품의 방법 과정에 대한 좋은 아이디어를 얻을 수있는 GIMP [40] 설명서에있는 예제를보고하는 것이 좋습니다.

A matrix and a bias define a kernel. A kernel only detects one certain relevant feature in the image so it is, therefore, recommended to use several kernels, one for each characteristic we would like to detect. This means that a full convolution layer in a CNN consists of several kernels.  The usual way of representing several kernels is as follows:
행렬과 바이어스는 커널을 정의합니다. 그것은 그래서 커널은 따라서, 우리가 검출하고자하는 각각의 특성에 대해 하나, 몇 가지 커널을 사용하는 것이 좋습니다 이미지에서 하나의 특정 관련 기능을 감지합니다. 이는 CNN에서 완전 컨볼 루션 층은 여러 가지 커널 구성되어 있다는 것을 의미한다. 다음과 같이 몇 가지 커널을 표현하는 일반적인 방법은 다음과 같습니다

image078

The first hidden layer is composed by several kernels. In our example, we use 32 kernels, each one defined by a 5×5 weight matrix W and a bias b that is also shared between the neuros of the hidden layer.
첫 번째 숨겨진 층은 여러 가지 커널에 의해 구성되어있다. 우리의 예에서 우리는 32 커널, 5 × 5 중량 행렬 W에 의해 정의 된 각각 그 또한 숨겨진 층의 뉴 로스간에 공유 B A 바이어스를 사용합니다.

In order to simplify the code, I define the following two functions related to the weight matrix W and bias b:
코드를 단순화하기 위해, I은 중량 매트릭스 W​​ 바이어스 (B)과 관련된 두 개의 함수를 정의

```
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```

Without going into the details, it is customary to initialize the weights with some random noise and the bias values slightly positive.
세부 사항에 가지 않고, 어떤 임의의 노이즈와 바이어스 값을 약간 긍정적으로 가중치를 초기화하는 것이 관례입니다.

In addition to the convolution layers that we just described, it is usual for the convolution layer to be followed by a so called pooling layer. The pooling layer simply condenses the output from the convolutional layer and creates a compact version of the information that have been put out by the convolutional layer. In our example, we will use a 2×2 region of the convolution layer of which we summarize the data into a single point using pooling:
컨벌루션 층은 소위 풀링 층 따라야하는 방금 설명한 컨벌루션 층 이외에, 그것은 일반적이다. 풀링 층은 단순히 길쌈 층의 출력을 응축하고 길쌈 층에 의해 넣어 한 정보의 컴팩트 버전을 만듭니다. 이 예에서, 우리는 우리가 풀링을 사용하여 단일 점으로 데이터를 요약하는 회선의 층의 2 × 2 영역을 사용한다 :

image080

There are several ways to perform pooling to condense the information; in our example we will use the method called max-pooling. This consists of condensing the information by just retaining the maximum value in the 2×2 region considered.
정보를 응축 풀링을 수행하는 방법은 여러 가지가 있습니다; 우리의 예에서 우리는 최대-풀링이라는 방법을 사용합니다. 이것은 단지 고려하는 2 × 2 영역에서의 최대 값을 유지하여 집광 정보로 구성된다.

As mentioned above, the convolutional layer consists of many kernels and, therefore, we will apply max-pooling to each of those separately. In general, there can be many layers of pooling and convolutions:
상술 한 바와 같이, 컨볼 루션 커널 계층은 여러 구성되고, 따라서 개별적으로 그 각각 맥스 풀링 적용된다. 일반적으로 풀링 및 회선의 많은 레이어가있을 수있다 :

image082

This leads that the 24×24 convolution result is transformed to a 12×12 space by the max-pooling layer that correspond to the 12×12 tiles, of which each originates from a 2×2 region. Note that, unlike in the convolutional layer, the data is tiled and not created by a sliding window.
이 24 × 24 컨볼 루션 결과는 각각 2 × 2 영역으로부터 유래 한의 12 × 12 타일에 대응하는 MAX-풀링 층에 의해 12 × 12 공간으로 변환되는 것으로 이끈다. 길쌈 층 달리, 데이터는 타일 및 슬라이딩 윈도우에 의해 생성되지 않습니다 있습니다.

Intuitively, we can explain max-pooling as finding out if a particular feature is present anywhere in the image, the exact location of the feature is not as important as the relative location with respect to others features.
특정한 특징 어디서나 이미지에 존재하는 경우 직관적으로, 우리가 알아내는 최대로 풀링을 설명 할 수 있으며, 기능의 정확한 위치는 다른 기능에 대하여 상대적인 위치만큼 중요하지 않다.

* Implementation of the model
* 모델의 구현

In this section, I will present the example code on how to write a CNN based on the advanced example (Deep MNIST for experts) that can be found on the TensorFlow [41] website. As I said in the beginning, there are many details of the parameters that require a more detailed treatment and theoretical approach than the given in this book. I hence will only give an overview of the code without going into to many details of the TensorFlow parameters.
이 섹션에서는, 나는 TensorFlow [41] 웹 사이트에서 찾을 수 있습니다 고급 예 (전문가 깊은 MNIST)을 기반으로 CNN을 작성하는 방법에 대한 예제 코드를 발표 할 예정이다. 내가 처음에 말했다 바와 같이,이 책에서 주어진보다 자세한 처리 및 이론적 접근을 필요로하는 매개 변수의 많은 세부 사항이있다. 나는 따라서 만 TensorFlow 매개 변수의 많은 세부 사항에 가지 않고 코드에 대한 개요를 제공합니다.

As we have already seen, there are several parameters that we have to define for the convolution and pooling layers.  We will use a stride of size 1 in each dimension (this is the step size of the sliding window) and a zero padding model. The pooling that we will apply will be a max-pooling on block of 2×2. Similar to above, I propose using the following two generic functions to be able to write a cleaner code that involves convolutions and max-pooling.
우리가 이미 보았 듯이, 우리가 회선 및 풀링 레이어 정의해야합니다 여러 매개 변수가 있습니다. 우리 및 제로 패딩 모델 (이 슬라이딩 윈도우의 스텝 크기), 각 차원의 크기 (1)의 보폭을 사용할 것이다. 우리가 적용 할 풀링은 2 × 2 블록에 최대-풀링 될 것입니다. 위에서와 마찬가지로, 나는 회선 및 최대-풀링을 포함하는 청소기 코드를 작성할 수 있도록 다음과 같은 두 가지 일반적인 기능을 사용하여 제안한다.

```
def conv2d(x, W):
 return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
 return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

Now it is time to implement the first convolutional layer followed by a pooling layer. In our example we have 32 filters, each with a window size of 5×5. We must define a tensor to hold this weight matrix W with the shape [5, 5, 1, 32]: the first two dimensions are the size of the window, and the third is the amount of channels, which is 1 in our case. The last one defines how many features we want to use. Furthermore, we will also need to define a bias for every of 32 weight matrices. Using the previously defined functions we can write this in TensorFlow as follows:
이제 풀링 층 뒤에 제 컨벌루션 층을 구현하는 시간이다. 우리의 예에서 우리는 5 × 5의 창 크기 (32) 필터, 각이있다. 우리는 형상을 가진 (W)이 가중치 행렬을 보유 텐서를 정의한다 [5, 5, 1, 32] : 처음 두 치수는 윈도우의 크기이며, 세 번째는 우리의 경우 1이고 채널의 양이며 . 마지막 하나는 우리가 사용하는 방법을 많은 기능을 정의합니다. 또한, 우리는 또한 32 중량 행렬의 모든에 대한 편견을 정의해야합니다. 다음과 같이 우리가 TensorFlow이 쓸 수 이전에 정의 된 함수를 사용 :

```
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

The ReLU (Rectified Linear unit) activation function has recently become the default activation function used in the hidden layers of deep neural networks. This simple function consist of returning max(0, x), so it return 0 for negative values and x otherwise.  In our example, we will use this activation function in the hidden layers that follow the convolution layers.
ReLU (정류 선형 장치) 활성화 기능은 최근에 깊은 신경망의 숨겨진 레이어에 사용되는 기본 활성화 기능이되고있다. 이 간단한 함수는 음수 값 0을 반환하고, 그렇지 않으면 X, 그것 때문에 최대 (X 0) 반환로 구성되어 있습니다. 우리의 예에서 우리는 컨볼 루션 층을 따라 숨겨진 층이 활성화 함수를 사용합니다.

The code that we are writing will first apply the convolution to the input images x_image, which returns the results of the convolution of the image in a 2D tensor W_conv1 and then it sums the bias to which finally the ReLU activation function is applied. As a final step, we apply max-pooling to the output:
우리는 먼저 2D 텐서 W_conv1의 화상의 컨볼 루션의 결과를 반환 입력 이미지 x_image로 콘볼 루션을 적용 작성되고이 바이어스를 합산 코드는 마지막 ReLU 활성화 함수를 적용한다. 마지막 단계로, 우리는 출력에 최대-풀링을 적용

```
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

When constructing a deep neural network, we can stack several layers on top of each other. To demonstrate how to do this, I will create a secondary convolutional layer with 64 filters with a 5×5 window. In this case we have to pass 32 as the number of channels that we need as that is the output size of the previous layer:
딥 뉴럴 네트워크를 구성 할 때, 우리는 서로의 상단에 여러 레이어 스택 수 있습니다. 이 작업을 수행하는 방법을 설명하기 위해, 나는 5 × 5 창 (64) 필터와 보조 길쌈 레이어를 생성합니다. 이 경우에 우리는 그 이전 층의 출력 크기이기 때문에 필요한 채널 수로 (32)을 통과해야한다 :

```
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

The resulting output of the convolution has a dimension of 7×7 as we are applying the 5×5 window to a 12×12 space with a stride size of 1. The next step will be to add a fully connected layer to 7×7 output, which will then be fed to the final softmax layer like we did in the previous chapter.
콘볼루션의 출력결과는 12x12크기의 공간에 5x5크기의 윈도우를 1의 보폭크기(stride size)으로 적용한 것으로 7x7의 차원을 가진다. 다음 단계로써 완전히 연결된 층에 7x7 출력을 더할 것이다. 이것은 우리가 이전 장에서 했던것과 같이 마지막으로 소프트맥스(softmax) 레이어를 추가할 것(be fed)이기 때문이다.

We will use a layer of 1024 neurons, allowing us to to process the entire image. The tensors for the weights and biases are as follows:
우리는 1024개의 뉴런들로 구성된 층을 사용할 것이다. 이것은 전체 이미지를 처리할 수 있도록 한다. 가중치와 바이어스들에 대한 텐서는 다음과 같이 선언 할 수 있다.

```
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
```

Remember that the first dimension of the tensor represents the 64 filters of size 7×7 from the second convolutional layer, while the second parameter is the amount of neurons in the layer and is free to be chosen by us (in our case 1024).
두번째 매개변수가 우리에 의해 선택될 수 없지만, 그 레이어에서 뉴런의 개수를 나타내는 동안, 첫번째 차원은 두번째 컨볼루셔널 레이어로부터 크기 7x7의 64개의 필터들을 나타냄을 기억하라.

Now, we want to flatten the tensor into a vector. We saw in the previous chapter that the softmax needs a flattened image in the form as a vector as input. This is achieved by multiplying the weight matrix W_fc1 with the flattend vector, adding the bias b_fc1 after which we apply the ReLU activation function:
이제, 우리는 텐서를 벡터로 평평하게 하기를 원한다. 우리는 소프트맥스(softmax)가 벡터와 같은 형태로 병합된 일차원적인 이미지 형태를 필요로 한다는 것을 이전장에서 보았다. 이것은 평평화된 벡터에 가중치 행렬 W_fc1을 곱하고, 바이어스(bias) b_fc1을 곱한 후 ReLU 활성화 함수를 적용하므로써 이루어진다.:

```
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

The next step will be to reduce the amount of effective parameters in the neural network using a technique called dropout. This consists of removing nodes and their incoming and outgoing connections. The decision of which neurons to drop and which to keep is decided randomly.  To do this in a consistent manner we will assign a probability to the neurons being dropped or not in the code.
다음 단계는 신경망에서 드랍아웃(dropout)이라 불리우는 기술을 사용하여 효과적인 매개변수의 양을 줄일 것이다. 이것은 노드들과 그들의 입력(수신,incoming)/출력(발신,outgoing) 연결들을 제거하는 것으로 구성된다. 어느 뉴런을 제거(drop)할지, 어떤 것을 유지할지는 무작위로 결정된다. 일관된 방식으로 이 작업을 수행하기 위해서, 우리는 코드에서 떨어뜨릴지 아닐지에 대해서 뉴런에 확률을 할당할 것이다.

Without going into too many details, dropout reduces the risk of the model of overfitting the data. This can happen when the hidden layers have large amount of neurons and thus can lead to very expressive models; in this case it can happen that random noise (or error) is modelled. This is known as overfitting, which is more likely if the model has a lot of parameters compared to the dimension of the input. The best is to avoid this situation, as overfitted models have poor predictive performance.
너무 많은 세부 사항의 설명없이, 드롭 아웃(dropout)은 모델이 과적화되는 것에 대한 위험을 줄일 수 있다. 이것은 숨겨진 층이 큰 양의 뉴런들을 가지고 있을때 매우 다체로운(expressive,?) 모델로 이어지는 일이 발생할 수 있다. 이 경우에 랜덤 노이즈 (또는 오류)가 모델링되는 일이 발생할 수 있다. 이는 과적화로 알려져 있다. 이것은 만일 모델이 입력의 차원과 비교하여 더 많은 매개변수를 가진다면 많이 접할 수 있는 상황이다. 이 상황을 피하는 것이 가장 좋다. 왜냐하면 과적화된 모델들은 빈약한 예측 성능을 가진다.

In our model we apply dropout, which consists of using the function dropout tf.nn.dropout before the final softmax layer. To do this we construct a placeholder to store the probability that a neuron is maintained during dropout:
우리의 모델에서 우리는 최종 softmax를 층 전에 함수 드롭 아웃 tf.nn.dropout를 사용하여 구성 드롭 아웃을 적용 할 수 있습니다. 이를 위해 우리는 신경 세포가 드롭 아웃 동안 유지되는 확률을 저장하는 자리를 만들 :

```
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

Finally, we add the softmax layer to our model like have been done in the previous chapter. Remember that softmax returns the probability of the input belonging to each class (digits in our case) so that the total probability adds up to 1. The softmax layer code is as follows:
마지막으로, 우리는 이전 장에서 해왔던 것과 같이 우리의 모델에 소프트맥스 레이어(Softmax Layer)를 추가한다. 소프트맥스는 각각의 클래스(우리의 경우 숫자들)를 나타내는 입력의 확률을 돌려줌을 기억하라. 이는 총 확률을 1이되게 한다. 소프트맥스 레이어 코드는 다음과 같다. :

```
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
```

* Training and evaluation of the model
* 모델의 교육 및 평가

We are now ready to train the model that we have just defined by adjusting all the weights in the convolution, and fully connected layers to obtain the predictions of the images for which we have a label. If we want to know how well our model performs we must follow the example set in the previous chapter.
우리는 하나의 레이블(label)에 대한 이미지들의 예측결과를 얻기 위해서 컨볼루션과 모두 연결된 층들에 있는 모든 가중치를 조절하므로써 정의된 모델을 훈련시킬 준비가 되었다. 만일 우리가 우리의 모델이 얼마나 잘 동작되는지 알기를 원한다면, 우리는 이전 장의 예제들을 가지고 확인해봐야 한다.

The following code is very similar to the one in the previous chapter, with one exception: we replace the gradient descent optimizer with the ADAM optimizer, because this algorithm implements a different optimizer that offers certain advantages according to the literature [42].
다음 코드는 하나를 제외하고는 이전 장에 있는 것과 매우 유사하다: 우리는 경사하강법 최적화를 ADAM 최적화 기법으로 교채했는데, 이 알고리즘은 문헌 [42]에 따라 확실한 이점들을 제공하는 하나의 다른 최적화이기 때문이다.

We also need to include the additional parameter keep_prob in the feed_dict argument, which controls the probability of the dropout layer that we discussed earlier.
우리는 또한 free_dict 매개변수에 추가적인 매개변수인 keep_prob를 포함할 필요가 있다. 이는 우리가 앞서 논의했던 드랍아웃 층의 확률을 조절하는 것이다.

```
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = sess.run( accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"% sess.run(accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

Like in the previous models, the entire code can be found on the Github page of this book, one can verify that this model achieves 99.2% accuracy.
이전 모델에서와 같이, 전체 코드는 이 책의 Github 페이지에서 볼 수 있다. 당신은 이 모델이 99.2 %의 정확도를 얻을 수 있는지 확인할 수 있다.

Here is when the brief introduction to building, training and evaluating deep neural networks using TensorFlow comes to an end. If the reader have managed to run the provided code, he or she has noticed that the training of this network took noticeably longer time than the one in the previous chapters; you can imagine that a network with much more layers will take a lot longer to train. I suggest you to read the next chapter, where is explained how to use a GPU for training, which will vaslty decrease your training time.
여기서 텐서플로우를 사용하여 딥뉴럴 네트워크를 만들고, 훈련시키고, 평가하기(evaluating) 위한 간략한 소개를 끝마치겠다. 만일 독자가 제공된 코드를 실행하기위해 만져봤다면, 당신은 망의 훈련이 이전 장에서의 그것보다 더 긴 시간이 걸렸다는 것을 알아챘을 것이다. 당신은 더 많은 레이어와 네트워크 훈련에 많은 시간이 더 걸릴 것이라고 상상할 수 있다. 나는 당신이 vaslty 훈련 시간을 감소시키기 위해 GPU를 어떻게 사용하지에 대한 내용이 있는 다음 장을 읽기를 추천한다.

The code of this chapter can be found in CNN.py on the github page of this book [43], for studying purposes the code can be found in its entirity below:
이 장의 코드는 이 책의 GitHub의 페이지에 있는 CNN.py에서 찾을 수 있다. 학습을 위하여, 아래에서 그 코드의 전체 내용을 볼 수 있다. :

```
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])
print "x_image="
print x_image

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.initialize_all_variables())

for i in range(200):
   batch = mnist.train.next_batch(50)
   if i%10 == 0:
     train_accuracy = sess.run( accuracy, feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
     print("step %d, training accuracy %g"%(i, train_accuracy))
   sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"% sess.run(accuracy, feed_dict={ 
       x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

[cotents link]
