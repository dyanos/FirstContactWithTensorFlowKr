1. TENSORFLOW BASICS
텐서플로우 기본

In this chapter I will present very briefly how a TensorFlow’s code and their programming model is. At the end of this chapter, it is expected that the reader can install the TensorFlow package on their personal computer.
이 장에서는 텐서플로우의 코드와 프로그래밍 모델을 간략히 소개한다. 후반부에 가면 독자는 본인 컴퓨터에 텐서플로우를 설치할 수 있을 것이다.

An Open Source Package
오픈소스 패키지

Machine Learning has been investigated by the academy for decades, but it is only in recent years that its penetration has also increased in corporations. This happened thanks to the large volume of data it already had and the unprecedented computing capacity available nowadays.
머신러닝은 수 십년간 학계가 주도 해왔지만 최근 수 년간 기업에서도 많이 사용되고 있다. 아마도 기업이 보유한 대량의 데이터와 전례없는 최신 컴퓨터 성능 덕택일 것이다. 이것은 기업이 기 보유한 대량의 데이터와 전례없는 최신의 컴퓨팅 성능 덕분이다.

In this scenario, there is no doubt that Google, under the holding of Alphabet, is one of the largest corporations where Machine Learning technology plays a key role in all of its virtual initiatives and products.
이런 상황에서, 알파벳의 자회사인 구글이 머신러닝 기술을 자사의 모든 virtual initiatives(?) 제품에 핵심 역할로 사용하는 가장 큰 회사 중에 하나임에 틀림 없다. 

Last October, when Alphabet announced its quarterly Google’s results, with considerable increases in sales and profits, CEO Sundar Pichai said clearly: “Machine learning is a core, transformative way by which we’re rethinking everything we’re doing”.
지난 10월 알파벳이 구글의 매출과 이익이 크게 늘어난 분기 실적을 발표할 때 선다 피차이 CEO는 “머신러닝은 우리가 하는 모든 일에 대해 다시 생각하게 만드는 핵심적이고 변화시키는 방법이다”라고 분명하게 말했다. 

Technologically speaking, we are facing a change of era in which Google is not the only big player. Other technology companies such as Microsoft, Facebook, Amazon and Apple, among many other corporations are also increasing their investment in these areas.
기술적으로 말하자면 구글 뿐만 아니라 우리는 격변의 시기를 마주하고 있다. 마이크로소프트, 페이스북, 아마존, 애플 같은 다른 기술 기반 회사들도 이 분야에 투자를 늘리고 있다. 

In this context, a few months ago Google released its TensorFlow engine under an open source license (Apache 2.0). TensorFlow can be used by developers and researchers who want to incorporate Machine Learning in their projects and products, in the same way that Google is doing internally with different commercial products like Gmail, Google Photos, Search, voice recognition, etc.
이런 맥락에서, 몇 달전 구글은 텐서플로우 엔진을 오픈소스 라이센스(Apache 2.0)로 공개했다. 구글이 지메일, 구글 포토스, 검색, 음성인식 같은 다양한 상용 제품에 텐서플로우를 사용한 것 처럼 자신의 프로젝트나 제품에 머신러닝을 적용하고 싶은 개발자나 연구원들이 텐서플로우를 사용할 수 있게 되었다. 

TensorFlow was originally developed by the Google Brain Team, with the purpose of conducting Machine Learning and deep neural networks research, but the system is general enough to be applied in a wide variety of other Machine Learning problems.
텐서플로우는 원래 머신러닝과 딥 뉴럴 네트워크(deep neural network) 연구를 수행하는 구글 브레인 팀에서 개발했지만 다양한 머신러닝 문제에도 적용하기에 충분하다. 

Since I am an engineer and I am speaking to engineers, the book will look under the hood to see how the algorithms are represented by a data flow graph. TensorFlow can be seen as a library for numerical computation using data flow graphs. The nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors), which interconnect the nodes.
필자 엔지니어고 또 엔지니어에게 말하고 있기 때문에, 이 책에서는 어떻게 알고리즘이 데이터 플로우 그래프(data flow graph)로 표현되는지 자세히 알아 보겠다. 텐서플로우는 데이터 플로우 그래프를 이용하여 수치 연산을 하기 위한 라이브러리로 볼 수 있다.  그래프상의 노드(node)는 수학적인 연산(operations)을 의미하고, graph의 엣지(edge)는 노드들 사이를 연결하는 다차원 데이터 행렬(tensor)이다.

TensorFlow is constructed around the basic idea of building and manipulating a computational graph, representing symbolically the numerical operations to be performed. This allows TensorFlow to take advantage of both CPUs and GPUs right now from Linux 64-bit platforms such as Mac OS X, as well as mobile platforms such as Android or iOS.
텐서플로우는 computational graph를 만들고 처리한다는 기본 개념을 바탕으로 만들어졌다. 그래서 텐서플로우는 안드로이드나 iOS 같은 모바일 플랫폼은 물론 Mac OS X와 같은 64비트 리눅스에서 CPU, GPU의 장점을 모두 이용할 수 있다. 

Another strength of this new package is its visual TensorBoard module that allows a lot of information about how the algorithm is running to be monitored and displayed. Being able to measure and display the behavior of algorithms is extremely important in the process of creating better models. I have a feeling that currently many models are refined through a little blind process, through trial and error, with the obvious waste of resources and, above all, time.
텐서플로우의 또 하나의 강점은 알고리즘이 어떻게 수행되는지 알려주기 위한 많은 정보를 모니터링하고 표현 해주는 텐서보드 모듈이다. 더 좋은 모델을 만드는 프로세스에서 알고리즘의 동작을 측정하고 표현하는 것은 매우 중요하다. 내 생각에 최근 시행착오를 통해 많은 모델을 만들고 있는데 이건 분명히 리소스 (무엇보다도 시간) 낭비이다.

TensorFlow Serving
텐서플로우 서빙

Recently Google launched TensorFlow Serving[3], that helps developers to take their TensorFlow machine learning models (and, even so, can be extended to serve other types of models) into production. TensorFlow Serving is an open source serving system (written in C++) now available on GitHub under the Apache 2.0 license.
구글은 개발자가 텐서플로우 머신러닝 모델(게다가 심지어 다른 종류의 모델로까지로도 확장)을 배포 할 수 있도록 도와주는 텐서플로우 서빙을 런칭했다. 텐서플로우 서빙은 C++로 구현된 오픈소스 서빙 시스템이며, 아파치 2.0 라이센스로 깃허브에 공개되어 있다.

What is the difference between TensorFlow and TensorFlow Serving?  While in TensorFlow it is easier for the developers to build machine learning algorithms and train them for certain types of data inputs, TensorFlow Serving specializes in making these models usable in production environments. The idea is that developers train their models using TensorFlow and then they use TensorFlow Serving’s APIs to react to input from a client.
텐서플로우와 텐서플로우 서빙의 차이점은 무엇일까? 텐서플로우가 입력된 데이터로 개발자가 쉽게 머신러닝 알고리즘을 만들고 모델을 훈련시키는 걸 도와주는 것이라면 텐서플로우 서빙은 이 모델을 운영환경에서 사용할 수 있도록 특화되어있다. 이 아이디어는 개발자가 텐서플로우를 이용하여 모델을 훈련시키고 클라이언트로부터 들어오는 입력 데이터에 대응하기 위해 텐서플로우 서빙 API를 이용하는 것이다.

This allows developers to experiment with different models on a large scale that change over time, based on real-world data, and maintain a stable architecture and API in place.
이것은 개발자들이 실 데이터로 대규모의 모델들을 바꿔가면서 실험해볼 수 있고, 안정적인 시스템과 API를 유지할 수 있다.

The typical pipeline is that a training data is fed to the learner, which outputs a model, which after being validated is ready to be deployed to the TensorFlow serving system. It is quite common to launch and iterate on our model over time, as new data becomes available, or as you improve the model. In fact, in the google post [4] they mention that at Google, many pipelines are running continuously, producing new model versions as new data becomes available.
전형적인 파이프라인(pipeline)은 학습기에 훈련 데이터를 공급하고 모델을 만든 다음 검증 절차를 거치면 텐서플로우 서빙 시스템에 배포할 준비가 된다. 모델을 런칭하고 반복적으로 수행할 때 새로운 데이터가 들어오거나 모델을 개선하기위해 이와 같은 작업은 일반적이다. 실제로 구글이 포스트에서 언급했듯이 많은 파이프라인이 지속적으로 수행되고 있고 새로운 데이터가 들어옴에 따라 새로운 버전의 모델을 만들고 있다.

Developers use to communicate with TensorFlow Serving a front-end implementation based on gRPC, a high performance, open source RPC framework from Google.
개발자는 텐서플로우 서빙과 통신하기 위해서 구글에서 오픈소스로 공개한 고성능 RPC 프레임워크인 gRPC를 사용한다. 

If you are interested in learning more about TensorFlow Serving, I suggest you start by by reading the Serving architecture overview [5] section, set up your environment and start to do a basic tutorial[6] .
텐서플로우 서빙에 대해 더 알고 싶다면 서빙 아키텍처 개요를 읽어보고 환경설정을 한 뒤에 기본 튜토리얼을 따라하길 추천한다.


TensorFlow Installation
텐서플로우 설치

It is time to get your hands dirty. From now on, I recommend that you interleave the reading with the practice on your computer.
이제 직접 해 볼 시간이다. 지금부터는 책을 읽으면서 직접 컴퓨터에서 실습을 해보기를 추천한다.

TensorFlow has a Python API (plus a C / C ++) that requires the installation of Python 2.7 (I assume that any engineer who reads this book knows how to do it).
텐서플로우는 파이썬 API(C/C++도 있음)가 있으며 이것을 사용하려면 파이썬 2.7 버전을 설치해야 한다. (이 책을 읽는 엔지니어라면 어떻게 설치하는지 알고 있을것이라 생각한다.)

In general, when you are working in Python, you should use the virtual environment virtualenv. Virtualenv is a tool to keep Python dependencies required in different projects, in different parts of the same computer. If we use virtualenv to install TensorFlow, this will not overwrite existing versions of Python packages from other projects required by TensorFlow.
일반적으로 파이썬으로 작업을 할 때는 virtualenv이라는 가상환경을 사용해야 한다. Virtualenv는 동일 컴퓨터에서 여러 프로젝트를 작업할 때 파이썬 패키지의 의존성을 독립 관리해주는 툴 이다. 즉 텐서플로우를 설치하기 위해 virtualenv를 사용하면, 텐서플로우에서 필요한 다른 프로젝트에서 같이 설치한 패키지를 덮어쓰지 않는다.

First, you should install pip and virtualenv if they are not already installed, like the follow script shows:
우선 pip와 virtualenv가 설치되어 있지 않다면, 아래의 스크립트로 설치해보자. 

```
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev python-virtualenv 

# Mac OS X 
$ sudo easy_install pip
$ sudo pip install --upgrade virtualenv
~/tensorflow 디렉토리에 virtualenv 환경을 만든다.

$ virtualenv --system-site-packages ~/tensorflow
다음은 아래처럼 virtualenv를 활성화시키는 것이다. 아래와 같이 될 수 있다. 

$ source ~/tensorflow/bin/activate #  with bash 
$ source ~/tensorflow/bin/activate.csh #  with csh
(tensorflow)$
```

The name of the virtual environment in which we are working will appear at the beginning of each command line from now on. Once the virtualenv is activated, you can use pip to install TensorFlow inside it:
이제는 명령줄 시작 부분에 현재 작업하고 있는 가상환경 이름이 나타나게 된다. virtualenv가 활성화 되었으므로 pip를 이용해 텐서플로우를 설치할 수 있다.

```
# Ubuntu/Linux 64-bit, CPU only:
(tensorflow)$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl 

# Mac OS X, CPU only:
(tensorflow)$ sudo easy_install --upgrade six
(tensorflow)$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
```

I recommend that you visit the official documentation indicated here, to be sure that you are installing the latest available version.
최신 버전을 설치하려면 공식 문서인 여기를 참고하는 것을 추천한다.

If the platform where you are running your code has a GPU, the package to use will be different. I recommend that you visit the official documentation to see if your GPU meets the specifications required to support Tensorflow. Installing additional software is required to run Tensorflow GPU and all the information can be found at Download and Setup TensorFlow[7] web page. For more information on the use of GPUs, I suggest reading chapter 6.
만약 코드를 수행하고 있는 플랫폼에 GPU가 있다면 설치해야할 패키지가 다르다. 공식 문서를 참고해서 당신의 GPU가 텐서플로우를 지원하는 스펙인지 확인하길 추천한다. 텐서플로우 GPU를 실행하기 위해서는 추가 소프트웨어가 설치되어야 하고 모든 정보는 텐서플로우 다운로드와 설치 웹 페이지에서 찾을 수 있다. GPU 사용에 대한 자세한 정보는 6장을 읽어보길 권장한다.

Finally, when you’ve finished, you should disable the virtual environment as follows:
마지막으로 작업을 끝났을때는 아래와 같이 가상 환경을 빠져나와야 한다.

```
(tensorflow)$ deactivate
```

Given the introductory nature of this book, we suggest that the reader visits the mentioned official documentation page to find more information about other ways to install Tensorflow.
이 책은 입문서로 쓰여 졌기에 텐서플로우를 설치하는 다른 방법에 대한 정보를 더 찾기 위해서는 공식 문서를 추천 한다.

My first code in TensorFlow
첫 번째 텐서플로우 코드

As I mentioned at the beginning, we will move in this exploration of the planet TensorFlow with little theory and lots of practice. Let’s start!
시작부에서 이야기했듯이, 텐서플로우라는 행성을 탐험하기 위해 약간의 이론만 다루고 많은 실습을 하겠다. 

From now on, it is best to use any text editor to write python code and save it with extension “.py” (eg test.py). To run the code, it will be enough with the command python test.py.
지금부터는 파이썬 코드를 작성하고 “.py” 확장자로 저장(예, test.py)하기 위해서 텍스트 에디터를 사용하는 것이 좋다.

To get a first impression of what a TensorFlow’s program is, I suggest doing a simple multiplication program; the code looks like this:
텐서플로우 프로그래밍 무엇인지 처음 느껴보기 위해서 간단한 곱셈 프로그램을 만들어보는 것을 추천한다. 코드는 아래와 같다.

```
import tensorflow as tf
  
a = tf.placeholder("float")
b = tf.placeholder("float")
  
y = tf.mul(a, b)

sess = tf.Session()

print sess.run(y, feed_dict={a: 3, b: 3})
```

In this code, after importing the Python module tensorflow, we define “symbolic” variables, called placeholder in order to manipulate them during the program execution. Then, we move these variables as a parameter in the call to the function multiply that TensorFlow offers. tf.mul is one of the many mathematical operations that TensorFlow offers to manipulate the tensors. In this moment, tensors can be considered dynamically-sized, multidimensional data arrays.
이 코드에서 tensorflow라는 python 모듈을 import 하고 나서 프로그램 실행중에 다루기 위해서 placeholder를 호출하는 "symbolic" 변수를 정의하였다. 그리고 나서 이 변수들을 텐서플로우가 제공하는 곱셈연산 함수의 파라미터로 넘긴다. tf.mul은 텐서를 다루기 위해 텐서플로우가 제공하는 많은 수학 연산 함수 중 하나이다. 여기서 텐서는 동적 사이즈 다차원 데이터 배열이라고 생각하면 된다.


The main ones are shown in the following table:
중요 함수들은 다음 테이블과 같다.
              
| Operation | Description |
| ----------|-------------|
| tf.add    | sum | 
| tf.sub |    substraction | 
| tf.mul |    multiplication | 
| tf.div |    division | 
| tf.mod |    module | 
| tf.abs |    return the absolute value | 
| tf.neg |    return negative value | 
| tf.sign |   return the sign | 
| tf.inv   |  returns the inverse | 
| tf.square | calculates the square | 
| tf.round |  returns the nearest integer | 
| tf.sqrt |   calculates the square root | 
| tf.pow |    calculates the power | 
| tf.exp |    calculates the exponential | 
| tf.log |    calculates the logarithm | 
| tf.maximum |    returns the maximum | 
| tf.minimum |    returns the minimum | 
| tf.cos |    calculates the cosine | 
| tf.sin |    calculates the sine | 


TensorFlow also offers the programmer a number of functions to perform mathematical operations on matrices. Some are listed below:
텐서플로우는 행렬 연산을 위한 함수도 제공한다. 몇 가지를 나열하면 아래와 같다.

Operation | Description
----------|------------
tf.diag | returns a diagonal tensor with a given diagonal values
tf.transpose | returns the transposes of the argument
tf.matmul |    returns a tensor product of multiplying two tensors listed as arguments
tf.matrix_determinant |    returns the determinant of the square matrix specified as an argument
tf.matrix_inverse |    returns the inverse of the square matrix specified as an argument
               
The next step, one of the most important, is to create a session to evaluate the specified symbolic expression. Indeed, until now nothing has yet been executed in this TensorFlowcode. Let me emphasize that TensorFlow is both, an interface to express Machine Learning’s algorithms and an implementation to run them, and this is a good example.
다음 단계로 가장 중요한 것 중에 하나인 심볼릭 표현을 평가하기 위해 세션을 만드는 것이다. 실제로 텐서플로우 코드에서 아직 아무것도 시작되지 않았다. 텐서플로우가 머신러닝 알고리즘을 표현하기 위한 인터페이스와 알고리즘을 실행하는 프로그램으로서의 측면을 가지고 있다고 강조하고 싶다. 이것은 좋은 예제이다.

Programs interact with Tensorflow libraries by creating a session with Session(); it is only from the creation of this session when we can call the run() method, and that is when it really starts to run the specified code. In this particular example, the values of the variables are introduced into the run() method with a feed_dict argument. That’s when the associated code solves the expression and exits from the display a 9 as a result of multiplication.
프로그램은 Session()을 통해 세션을 생성함으로써 텐서 플로우와 상호작용하게 된다. 세션을 만든 다음 run() 함수를 호출 할 때 진짜로 입력된 코드가 실행된다. 이 예제에서 run() 함수에 전달될 변수들의 값은 feed_dict의 인자로 넘기면 된다. 입력된 코드가 수행되면 곱셈의 결과인 9를 화면에 출력하고 종료된다.

With this simple example, I tried to introduce the idea that the normal way to program in TensorFlow is to specify the whole problem first, and eventually create a session to allow the running of the associated computation.
간단한 이 예제에서 텐서플로우 프로그램의 일반적인 방법은 전체 알고리즘을 정의하고 그 다음 최종적으로 관련 계산을 수행하기 위해 세션을 생성시킨다.

Sometimes however, we are interested in having more flexibility in order to structure the code, inserting operations to build the graph with operations running part of it. It happens when we are, for example, using interactive environments of Python such as IPython [8]. For this purpose, TesorFlow offers the tf.InteractiveSession() class.
그러나 때로는 일부의 부분만 계산을 수행하면서 그래프를 만들기 위해 연산을 추가하는 등의 구조화된 코드를 위해서 더 복잡한 것이 필요할 때가 있다. 예를 들어 IPython과 같은 인터액티브 환경을 사용하여 개발할 때 이런 일은 빈번히 발생할 수 있다. 이런 목적으로 텐서플로우는 tf.InteractiveSession() 클래스를 제공한다.

The motivation for this programming model is beyond the reach of this book. However, to continue with the next chapter, we only need to know that all information is saved internally in a graph structure that contains all the information operations and data.
이런 프로그래밍 모델을 채택하게 된 이유를 설명하는 것은 이 책의 범위를 벗어난다. 그러나 다음 장으로 넘어가기 위해서는 그래프 구조에 모든 정보 (operation과 data)를 내부적으로 저장하고 있다는 것은 기억하고 넘어가도록 하자.

This graph describes mathematical computations. The nodes typically implement mathematical operations, but they can also represent points of data entry, output results, or read/write persistent variables. The edges describe the relationships between nodes with their inputs and outputs and at the same time carry tensors, the basic data structure of TensorFlow.
이 그래프 구조는 수학적인 계산을 묘사한다. 노드는 일반적으로 수학 연산을 나타내고 데이터 엔트리의 포인트와, 아웃풋 결과 또는 저장된 변수의 입.출력을 한다. 엣지는 입력과 출력의 노드 사이에 관계를 표현하고 그와 동시에 텐서플로우의 기본 데이터 구조인 텐서를 옮긴다.

The representation of the information as a graph allows TensorFlow to know the dependencies between transactions and assigns operations to devices asynchronously, and in parallel, when these operations already have their associated tensors (indicated in the edges input) available.
텐서플로우는 그래프로 표현된 정보를 이용하여 트랜잭션간의 의존성을 파악하고, 연산들이 관련된 텐서가 이미 있을 경우 병렬로 디바이스 비동기적 연산을 할당한다.

Parallelism is therefore one of the factors that enables us to speed up the execution of some computationally expensive algorithms, but also because TensorFlow has already efficiently implemented a set of complex operations. In addition, most of these operations have associated kernels which are implementations of operations designed for specific devices such as GPUs. The following table summarizes the most important operations/kernels[9]:
병렬처리는 계산비용이 많이 드는 복잡한 알고리즘을 빠르게 실행할 수 있는 특징이 있고, 또한 텐서 플로우는 이미 복잡한 연산들을 효율적으로 구현해 놓았다. 게다가 이런 연산들은 대부분 연산들의 대부분은 GPU와 같은 특정 디바이스를 위해 연산이 구현된 관련 커널을 가지고 있다. 다음 표는 중요한 연산/커널을 요약했다.

Operations groups | Operations
------------------|-------------
Maths |   Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal
Array |   Concat, Slice, Split, Constant, Rank, Shape, Shuffle
Matrix |  MatMul, MatrixInverse, MatrixDeterminant
Neuronal | Network SoftMax, Sigmoid, ReLU, Convolution2D, MaxPool
Checkpointing |   Save, Restore
Queues and syncronizations  | Enqueue, Dequeue, MutexAcquire, MutexRelease
Flow control | Merge, Switch, Enter, Leave, NextIteration
               
Display panel Tensorboard
디스플레이 패널 텐서보드

To make it more comprehensive, TensorFlow includes functions to debug and optimize programs in a visualization tool called TensorBoard. TensorBoard can view different types of statistics about the parameters and details of any part of the graph computing graphically.
더 범용적인 툴로 만들기 위해 텐서플로우는 텐서보드라고 불리는 시각화 툴에 디버깅하고 프로그램을 최적화하는 기능을 포함시켰다. 텐서보드에서는 도식화한 그래프의 각 부분의 파라미터와 상세 정보에 대한 여러가지 통계를 볼 수 있다.


The data displayed with TensorBoard module is generated during the execution of TensorFlow and stored in trace files whose data is obtained from the summary operations. In the documentation page[10] of TensorFlow, you can find detailed explanation of the Python API.
텐서보드 모듈에 나타나는 데이터는 텐서플로우가 실행되는 동안 생성되며 summary 연산으로 얻을 수 있는 데이터로서 추적 파일에 저장된다. 텐서플로우의 도큐먼트 페이지에서 파이썬 API의 자세하게 설명문서를 참고하자.


The way we can invoke it is very simple: a service with Tensorflow commands from the command line, which will include as an argument the file that contains the trace.
텐서보드를 실행하는 것은 간단한다. 커맨드라인에서 trace가 담겨있는 파일을 인자로 지정하여 실행시키면 된다.

```
(tensorflow)$ tensorboard --logdir=&lt;trace file&gt;
```

You simply need to access the local socket 6006 from the browser[11] with http://localhost:6006/ .
당신은 인터넷 브라우저를 이용하여 로컬 소켓 6006에 접근할 필요가 있다.

The visualization tool called TensorBoard is beyond the reach of this book. For more details about how Tensorboard works, the reader can visit the section TensorBoard Graph Visualization[12]from the TensorFlow tutorial page.
텐서보드에 대한 자세한 설명은 이 책의 범위를 벗어난다. 텐서보드가 어떻게 동작하는지에 대한 자세한 것은 텐서플로우 튜토리얼 페이지의 텐서보드 그래프 시각화 부분을 참고하자.

[contents link]

