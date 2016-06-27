3. CLUSTERING IN TENSORFLOW 텐서 플로우의 군집화
===============================
Linear regression, which has been presented in the previous chapter, is a supervised learning algorithm in which we use the data and output values (or labels) to build a model that fits them. But we haven’t always tagged data, and despite this we also want analyze them in some way. In this case, we can use an unsupervised learning algorithm as clustering. The clustering method is widely used because it is often a good approach for preliminary screening data analysis.
이전 장에서 설명했던 선형 회귀는, 데이터와 결과물(또는 라벨들)로 적합(fitting)할 모델을 생성하는 지도학습을 말하는 것이었다. 그러나 데이터가 항상 키워드로 분류되어 있는것은 아니다. 그럼에도, 우리는 이러한 데이터를 어떻게 해서든 분석하고 싶어한다. 이러한 경우 우리는 군집화라는 비지도학습 알고리즘을 이용할 수 있다. 군집화 방식이 두루 쓰이고 있는데, 예비 선별하는 데이터 분석에 유용한 방식이기 때문이다. 

In this chapter, I will present the clustering algorithm called K-means. It is surely the most popular and widely used to automatically group the data into coherent subsets so that all the elements in a subset are more similar to each other than with the rest. In this algorithm, we do not have any target or outcome variable to predict estimations.
이번 장에서, K평균 알고리즘을 설명 할 것이다. 이것은 데이터를 자동적으로 질서있는 집합으로 무리지을 때 가장 인기 있고 폭 넓게 쓰이고 있다. 각 부분 집합들의 모든 요소들은 나머지 집합의 요소들 보다 해당 집합내에서 보다 유사성을 갖는다. 

I will also use this chapter to achieve progress in the knowledge of TensorFlow and go into more detail in the basic data structure called tensor. I will start by explaining what this type of data is like and present the transformations that can be performed on it. Then, I will show the use of K-means algorithm in a case study using tensors.
나는 이번 장을 통해서 텐서 플로우에 대한 지식을 확장 시킬 것이고 텐서라고 불리는 기본 데이터 구조에 대해 좀 더 구체적으로 접근할 것이다. 나는 이 데이터 형태가 어떠한 것인지 설명하고 그것에 대한 변환 이행을 보여줄 것이다. 그 뒤, 나는 텐서를 이용한 K평균 알고리즘의 사용을 보여줄 것이다.

## Basic data structure: tensor 기본 데이터 구조 : 텐서

TensorFlow programs use a basic data structure called tensor to represent all of their datum. A tensor can be considered a dynamically-sized multidimensional data arrays that have as a properties a static data type, which can be from boolean or string to a variety of numeric types. Below is a table of the main types and their equivalent in Python.
텐서플로우 프로그램들은 모든 데이터 자료를 표현하는데 있어서 텐서라는 기본 데이터형을 이용한다. 텐서는 논리형 연산자나 문자열, 다양한 수치형 데이터 같은 정적 데이터 속성을 갖는 동적 크기의 다차원 배열이라고 말할 수 있다.  아래 표는 주요 데이터 형과 파이썬에 대응하는 데이터형에 대한 것이다.
 
 Type in TensorFlow | Type in Python | Description
--------------------|----------------|---------------
 DT_FLOAT           | tf.float32     | Floating point of 32 bits
 DT_INT16           | tf.int16       | Integer of 16 bits
 DT_INT32           | tf.int32       | Integer of 32 bits
 DT_INT64           | tf.int64       | Integer of 64 bits
 DT_STRING          | tf.string      | String
 DT_BOOL            | tf.bool        | Boolean
 

 TensorFlow에서의 타입 | Python에서의 타입 | 설명 
-----------------------|----------------- -|---------------
 DT_FLOAT              | tf.float32        | 32 비트의 실수 
 DT_INT16              | tf.int16          | 16 비트의 정수 
 DT_INT32              | tf.int32          | 32 비트의 정수
 DT_INT64              | tf.int64          | 64 비트의 정수
 DT_STRING             | tf.string         | 문자열  
 DT_BOOL               | tf.bool           | 논리값(boolean)  
 
 In addition, each tensor has a rank, which is the number of its dimensions. For example, the following tensor (defined as a list in Python) has rank 2:
 그리고 각 텐서는 차원을 나타내는 랭크(rank)를 갖는다. 예를 들어 다음의 텐서는 (파이썬 list로 표 현된) 랭크 2를 갖는다. 

 t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

 Tensors can have any rank. A rank 2 tensor is usually considered a matrix, and a rank 1 tensor would be a vector. Rank 0 is considered a scalar value.
 텐서는 어느 랭크나 가질수 있다. 랭크 2를 갖는 텐서는 행렬을 의미한다. 랭크 1을 갖는 텐서는 벡터라 할 수 있다. 랭크 0은  스칼라값을 말한다. 

 TensorFlow documentation uses three types of naming conventions to describe the dimension of a tensor: Shape, Rank and Dimension Number. The following table shows the relationship between them in order to make easier the Tensor Flow documentation’s traking easier:
 텐서플로우 문서는 형태, 랭크, 차수 라는 3 가지 형태의 호칭 규칙을 쓰고 있는다.  아래 표는 텐서 플로우 문서를 쉽게 이해할 수 있도록 이들의 관계를 표현한 것이다. 

 Shape             | Rank | Dimension Number
 --------------    |------|--------------------
 []                |    0 | 0-D
 [D0]              |    1 | 1-D
 [D0, D1]          |    2 | 2-D
 [D0, D1, D2]      |    3 | 3-D
           ~       |    ~ |  ~
 [D0, D1,  ~ Dn]   |    n | n-D
 
  형태             | 랭크 | 차원 
 --------------    |------|--------------------
 []                |    0 | 0-D
 [D0]              |    1 | 1-D
 [D0, D1]          |    2 | 2-D
 [D0, D1, D2]      |    3 | 3-D
           ~       |    ~ |  ~
 [D0, D1,  ~ Dn]   |    n | n-D
 
 These tensors can be manipulated with a series of transformations that supply the TensorFlow package. Below, we discuss some of them in the next table.
 이들 텐서들은 텐서플로우 패키지를 변환하는 일련의 작업으로 다뤄질 수 있다. 하단에서, 우리는 이들 중 일부를 다음 표에서 다룬다.

 Throughout this chapter we will go into more detail on some of them. A comprehensive list of transformations and details of each one can be found on the official website of TensorFlow, Tensor Transformations[18].
이번 장 전반에 걸쳐, 우리는 이들 중 일부를 더 자세히 다룰 것이다. 변환에 대한 포괄적인 항목이나 각 항목의 자세한 사항은 텐서플로우의 공식 웹 사이트 중 텐서 변환[18] 에서 찾을 수 있다.
 

 Operation      | Description
 ---------------|--------------------------------------------------------------------------
 tf.shape       | To find a shape of a tensor
 tf.size        | To find the size of a tensor
 tf.rank        | To find a rank of a tensor
 tf.reshape     | To change the shape of a tensor keeping the same elements contained
 tf.squeeze     | To delete in a tensor dimensions of size 1
 tf.expand_dims | To insert a dimension to a tensor 
 tf.slice       | To remove a portions of a tensor
 tf.split       | To divide a tensor into several tensors along one dimension
 tf.tile        | To create a new tensor replicating a tensor multiple times
 tf.concat      | To concatenate tensors in one dimension
 tf.reverse     | To reverse a specific dimension of a tensor
 tf.transpose   | To transpose dimensions in a tensor
 tf.gather      | To collect portions according to an index

    연산        |    설명
 ---------------|--------------------------------------------------------------------------
 tf.shape       | tensor의 형태를 알고싶을 때 
 tf.size        | tensor의 크기를 알고싶을 때
 tf.rank        | tensor의 랭크를 알고싶을 때
 tf.reshape     | tensor안의 요소(element)를 유지한 채 형태를 변경하고자 할 때 
 tf.squeeze     | tensor에서 크기가 1인 차원을 삭제하고자 할 때 
 tf.expand_dims | tensor에 차원을 추가하고자 할 때  
 tf.slice       | tensor의 일부분을 삭제하고자 할 때 
 tf.split       | tensor를 한 차원을 기준으로 여러개의 tensor로 나누고자 할 때
 tf.tile        | 한 tensor를 여러번 중복으로 늘려 새 tensor를 만들자 할 때 
 tf.concat      | 한 tensor를 기준으로 tensor를 이어서 붙이고자 할 때 
 tf.reverse     | tensor의 지정된 차원을 역으로 바꾸고자 할 때 
 tf.transpose   | tensor의 지정된 차원을 전치(transpose)시키고자 할 때 
 tf.gather      | 인덱스에 대응되는 부분을 모으고자 할 때
 
 For example, suppose that you want to extend an array of 2×2000 (a 2D tensor) to a cube (3D tensor). We can use the tf.expand_ dims function, which allows us to insert a dimension to a tensor:
 실례로, 2x2000 의 2차원 텐서 어레이를 3차원 텐서 형태로 확장한다고 해보자. 우리는 텐서 안에 차원을 추가해주는 tf.expand_dims 함수를 사용할 수 있다.

```python
vectors = tf.constant(conjunto_puntos)
extended_vectors = tf.expand_dims(vectors, 0)
In this case, tf.expand_dims inserts a dimension into a tensor in the one given in the argument (the dimensions start at zero).
```
Visually, the above transformation is as follows:
위의 변환을 시각화하면 다음과 같다.

![image023](http://www.jorditorres.org/wp-content/uploads/2016/02/image023.gif)

As you can see, we now have a 3D tensor, but we cannot determine the size of the new dimension D0 based on function arguments.
보는바와 같이, 이제 우리는 3차원 텐서를 가지게 되었다. 그러나, 우리는 함수 인수에 기반한 새로운 차원의 D0 의 사이즈는 알 수 없다.

If we obtain the shape of this tensor with the get_shape() operation, we can see that there is no associated size:
get_shape() 함수를 통해 이 텐서의 모양을 알아낸다면, 우리는 관련된 사이즈가 없음을 알 수 있다.

print expanded_vectors.get_shape()

It appears on the screen like:
화면에 다음과 같이 나타난다.

```python
TensorShape([Dimension(1), Dimension(2000), Dimension(2)])
```
Later in this chapter, we will see that, thanks to TensorFlow shape broadcasting, many mathematical manipulation functions of tensors (as presented in the first chapter), are able to discover for themselves the size in the dimension which unspecific size and assign to it this deduced value.
고맙게도 플로우의 형태 전개(bradcasting)기능이 있어서, 이 장 다음에 우리는 (1장에서 다루어 졌던)  텐서의 많은 수학적 변형 함수들이 스스로 불특정한 크기의 차원의 크기를 스스로 발견할 수 있음과,  그것을 추론된 값에 할당할 수 있음을 보게될 것이다.


## Data Storage in TensorFlow 텐서 플로우에서의 데이터 저장

Following the presentation of TensorFlow’s package, broadly speaking there are three main ways of obtaining data on a TensorFlow program:
이어지는 텐서 플로우 패키지 안내를 통해, 텐서 플로우 프로그램에서 일반적으로 사용되는 3가지의 주된 데이터 적재 방법을 다룰 것이다.

From data files.
Data preloaded as constants or variables.
Those provided by Python code.
Below, I briefly describe each of them.
데이터 파일에서 적재하기
데이터는 상수나 변수로 사전에 적재된다
이러한 것들은 파이썬 코드로 제공된다
아래에, 각각의 방법을 간단히 기술한다

## Data files 데이터 파일
Usually, the initial data is downloaded from a data file. The process is not complex, and given the introductory nature of this book I invite the reader to visit the website of TensorFlow[19] for more details on how to download data from different file types. You can also review the Python code input_data.py[20](available on the Github book), which loads the MNIST data from files (I will use this in the following chapters).
일반적으로, 초기 데이터는 데이터 파일로부터 다운로드되어진다. 이 과정은 복잡하지 않다. 그리고 이 책의 서두에서 언급한것 처럼, 다른 파일 타입으로부터 어떻게 데이터를 다운받는지에 대한 자세한 사항에 대해서는 텐서플로우[19]의 웹사이트를 방문하길 추천한다. 파일에서 MNIST 데이터를 읽는 input_data.py[20](깃헙6 북 에서도 가능하다) 의 파이썬 코드를 읽어볼 수도 있다.(나는 다음 챕터에서 이것을 사용할 것이다)

## Variables and constants 변수와 상수
When it comes to small sets, data can also be found pre-loaded into memory; there are two basic ways to create them, as we have seen in the previous example:
작은 단위의 데이터를 쓸 경우, 메모리에 미리 적재하고 사용할 수도 있다; 앞의 예제에서 본 것 처럼, 이를 만드는데에는 두 가지 기본적인 방법이 있다:

As a constants using constant(…)
As a variable using Variable(…)
constant(…) 를 사용한 상수
Variable(…) 를 사용한 변수

TensorFlow package offers different operations that can be used to generate constants. In the table below you can find a summary of the most important:
텐서플로우 패키지는 상수를 생성할 수 있는 다양한 명령어들을 제공한다. 하단의 테이블에서, 가장 중요한 것들의 요약정보를 확인할 수 있다.

Operation      | Description
---------------|-------------------------------------------------------------------------------------
tf.zeros_like  | Creates a tensor with all elements initialized to 0
tf.ones_like   | Creates a tensor with all elements initialized to 1
tf.fill        | Creates a tensor with all elements initialized to a scalar value given as argument
tf.constant    | Creates a tensor of constants with the elements listed as an arguments
 
    연산       |    설명 
---------------|-------------------------------------------------------------------------------------
tf.zeros_like  | 모든 요소를 0으로 초기화한 tensor를 생성
tf.ones_like   | 모든 요소를 1로 초기화한 tensor를 생성
tf.fill        | 모든 요소를 인자로 주어진 스칼라 값으로 초기화한 tensor를 생성
tf.constant    | 인자로 주어진 리스트화 된 원소를 이용하여 상수 tensor를 생성

In TensorFlow, during the training process of the models, the parameters are maintained in the memory as variables. When a variable is created, you can use a tensor defined as a parameter of the function as an initial value, which can be a constant or a random value. TensorFlow offers a collection of operations that produce random tensors with different distributions:
텐서플로우에서, 모델이 훈련되는 동안에, 파라미터들은 메모리 안에 변수로서 존재한다. 변수가 생성될 때, 함수의 파라미터로 정의된 텐서를 사용할 수 있고, 이 텐서들은 상수나 랜덤값이 될 수 있다. 텐서플로우는 다른 분포를 가지는 임의의 텐서를 만드는 일련의 명령어를 제공한다:
     

Operation           | Description
--------------------|------------------------------------------------------------------------------------------------------------------------------------
tf.random_normal    | Random values with a normal distribution
tf.truncated_normal | Random values with a normal distribution but eliminating those values whose magnitude is more than 2 times the standard deviation
tf.random_uniform   | Random values with a uniform distribution
tf.random_shuffle   | Randomly mixed tensor elements in the first dimension
tf.set_random_seed  | Sets the random seed
 
      연산          | 설명 
--------------------|------------------------------------------------------------------------------------------------------------------------------------
tf.random_normal    | 정규 분포를 가지는 난수를 생성 
tf.truncated_normal | 정규 분포를 가지는 난수를 생성. 단 2표준편차 범위 바깥의 값은 제거.
tf.random_uniform   | 균등 분포를 가지는 난수를 생성
tf.random_shuffle   | 첫번째 차원을 기준으로 하여 tensor의 원소를 무작위로 섞음
tf.set_random_seed  | 난수 seed를 정함 

An important detail is that all of these operations require a specific shape of the tensors as the parameters of the function, and the variable that is created has the same shape. In general, the variables have a fixed shape, but TensorFlow provides mechanisms to reshape it if necessary.
이 동작들은 함수의 파라미터로써 텐서의 특별한 모습을 필요로 한다. 그리고 만들어지는 변수들도 같은 모습이어야 한다. 일반적으로, 변수들은 고정된 모습을 가지지만, 텐서플로우는 필요한 경우 이들의 형태를 바꿀 수 있는 방법을 제공한다.

When using variables, these must be explicitly initialized after the graph that has been constructed, and before any operation is executed with the run() function. As we have seen, it can be used tf.initialize_all_variables() for this purpose. Variables also can be saved onto disk during and after training model through TensorFlow tf.train.Saver() class, but this class is beyond the scope of this book.
변수를 사용하는 경우, 그래프의 정의 이후에 명시적으로 초기화되어야만 하며, 초기화 전에는 반드시 run() 함수를 통해 실행되어야만 한다. 우리가 봐온 것처럼, 동일한 목적을 위해, tf.initialize_all_variables() 도 사용될 수 있다. 변수들은 텐서플로우의 tf.train.Saver() 클래스를 이용하면 트레이닝 중이나 후에 디스크로 저장될 수 있다. 그러나 이 클래스를 이 책에서 다루지는 않는다.


## Provided by Python code
Finally, we can use what we have called “symbolic variable” or placeholder to manipulate data during program execution. The call is placeholder(), which includes arguments with the type of the elements and the shape of the tensor, and optionally a name.
파이썬 코드
마지막으로, 우리는 프로그램 수행 중에 데이터를 처리하는 "symbolic variable" 또는 placeholder 라고 불리는 것을 사용할 것이다. placeholder() 로 호출하면 되며, 원소의 타입과 텐서의 모양, 그리고 선택적으로 이름을 선언한 인수를 포함한다.

At the same time as making the calls to Session.run() or Tensor.eval() from the Python code, this tensor is populated with the data specified in the feed_dict parameter. Remember the first code in Chapter 1:
파이썬 코드에서, Session.run() 또는 Tensor.eval() 을 호출함과 동시에, 텐서는 feed_dict 의 매개변수에 지정된 데이터로 채워진다. 챕터 1의 첫번째 코드를 기억하라:

```python
import tensorflow as tf
a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.mul(a, b)
sess = tf.Session()
print   sess.run(y, feed_dict={a: 3, b: 3})
```
In the last line of code, when the call sess.run() is made, it is when we pass the values of the two tensors a and b through feed_dict parameter.
코드의 마지막 라인에서, sess,run() 이 호출될 때가 feed_dict 파라미터로 a 와 b 두 텐서의 값이 전달되는 때이다.

With this brief introduction about tensors, I hope that from now on the reader can follow the codes of the following chapters without any difficulty.
텐서에 대한 간략한 소개와 더불어, 나는 독자들이 지금부터 이어지는 챕터의 코드들을 어려움 없이 이해할 수 있으리라 기대한다.


K-means algorithm K-평균 알고리즘
----------------------------------

K-means is a type of unsupervised algorithm which solves the clustering problem. Its procedure follows a simple and easy way to classify a given data set through a certain number of clusters (assume k clusters). Data points inside a cluster are homogeneous and heterogeneous to peer groups, that means that all the elements in a subset are more similar to each other than with the rest.
K평균 알고리즘은 군집 문제를 해결하는 비지도학습의 종류 이다. 이것의 동작은 특정한 수의 클러스터(k cluster)를 통한 간단하고 쉬운 분석을 구분 작업 수행이다. 군집 내 데이터 좌표는 동질성이 있고, 타 그룹에 대하여 구분된 특질이 있다. 군집내 요소들은 나머지 요소들보다 동질성이 있다는 것을 의미한다.

The result of the algorithm is a set of K dots, called centroids, which are the focus of the different groups obtained, and the tag that represents the set of points that are assigned to only one of the K clusters. All the points within a cluster are closer in distance to the centroid than any of the other centroids.
이 알고리즘의 결과물로는 중심값(Centroid)라고 불리는 K점들이다. 이것은 구분되는 그룹들 간의 주요점이 된다. 그리고 K클러스터에만 유일하게 할당되는 점들의 집합을 대표하는 식별자가 된다.   

Making clusters is a computationally expensive problem if we want to minimize the error function directly (what is known as an NP-hard problem); and therefore it some algorithms that converge rapidly in a local optimum by heuristics have been created. The most commonly used algorithm uses an iterative refinement technique, which converges in a few iterations.
만일 군집화 작업을 우리가 직접적으로 에러를 최소화하는 방향으로 작업할 때, 계산에 드는 비용이 상당히 문제이다. (NP-hard라고 알려진 문제), 그러므로 휴리스틱하게 지역적 최적값으로 빠르게 수렴하는 알고리즘들이 몇가지 개발 되었다. 가장 일반적으로 쓰여지는 알고리즘은 수회의 반복을 통하여 반복적인 정제 작업을 수행한다. 

Broadly speaking, this technique has three steps:
개략적으로 말하면, 이 기술은 3가지 단계가 있다. 

Initial step (step 0): determines an initial set of K centroids.
Allocation step (step 1): assigns each observation to the nearest group.
Update step (step 2): calculates the new centroids for each new group.
There are several methods to determine initial K centroids. One of them is randomly choose K observations in the data set and consider them centroids; this is the one we will use in our example.
초기 단계 (step 0): 초기 K중심값 결정
할당 단계 (step 1): 가장 관측값에서 가까운 그룹에 할당
업데이트 단계 (step 2): 각 그룹에서 새로운 중심치 도출
초기 K중심값을 결정하는 방법은 여러가지가 있다. 그 방법 중 하나는 랜덤하게 K개 관측점을 데이터 집합에서 선별하여 중심값으로 고려한다. 이 방법을 우리의 예제에서 이용할 것이다.    

The steps of allocation (step 1) and updating (step 2) are being alternated in a loop until it is considered that the algorithm has converged, which may be for example when allocations of points to groups no longer change.
핟당 단계(step 1)과 업데이트 단계(step 2)는 알고리즘이 수렴되었다고 여겨질 때 까지 번갈아가며 수행된다. 수렴은 예를들어 할당 결과가 더이상 바뀌지 않을 때가 될 수 있다.

Since this is a heuristic algorithm, there is no guarantee that it converges to the global optimum, and the outcome depends on the initial groups. Therefore, as the algorithm is generally very fast, it is usual to repeat executions multiple times with different values of the initials centroides, and then weigh the result.
이것은 휴리스틱한 알고리즘이므로, 전체적으로 최적값으로 수렴한다는 보장을 할 수 없다. 그리고 결과물은 초기 그룹에 영향을 받는다.  그러므로 (이 알고리즘이 일반적을 빠른 종류이므로 ) 대개 다양한 초기 중심치값들을 두고 많은 회차를 수행하면서 결과를 비교한다.

To start coding our example of K-means in TensorFlow I suggest to first generate some data as a testbed. I propose to do something simple, like generating 2,000 points in a 2D space in a random manner, following two normal distributions to draw up a space that allows us to better understand the outcome. For example, I suggest the following code:
우리의 텐서플로우를 이용한 K평균 예제 코딩을 시작하기 위하여, 나는 먼저 테스트 데이터를 생성하는 것을 권한다. 나는 간단하게 2차 평면에서 랜덤하게 2,000개의 포인트를 생성할 것을 제시한다. 해당 포인트는 2개의 정규 분포가 공간에 분포 될 것이고, 이것을 통해 결과물에 대한 이해를 높이게 될 것이다. 예로 아래와 같이 제시한다. 

```python
num_puntos = 2000
conjunto_puntos = []
for i in xrange(num_puntos):
   if np.random.random() &gt; 0.5:
        conjunto_puntos.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
   else:
        conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])
```
As we have done in the previous chapter, we can use some Python graphic libraries to plot the data. I propose that we use matplotlib like before, but this time we will also use the visualization package Seaborn based on matplotlib and the data manipulation package pandas, which allows us to work with more complex data structures.
이전 장에서 작업했던 것과 같이, 우리는 데이터를 도식화 할 때 파이썬 그래픽 라이브러리를 이용할 수 있다. 나는 이전과 같이 matplotlib을 이용할 것을 제안하며, matplotlib에 기반한 Seaborn 이라는 시각화 패키지를 이용하고, pandas라는 데이터 처리 패키지를 이용할 것이다. 우리는 보다 복잡한 구조의 데이터를 pandas를 통해 해결할 것이다. 

If you do not have these packages installed, you must do it with the pip value before you can run the following codes.
만을 이러한 패키지가 설치 되지 않을 경우, 아래 코드를 실행하기 전에 pip 과정을 통해 반드시 완료하여야 한다.

To display the points that have been generated randomly I suggest the following code:
랜덤하게 생성된 포인트들이 보여지기 위하여 아래와 같이 코드를 제시한다.

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame({"x": [v[0] for v in set_points],
"y": [v[1] for v in set_points]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()
```
This code generates a graph of points in a two dimensional space like the following screenshot:
이 코드는 아래 그림과 같이 2차 평면에 포인트 그래프를 생성한다.

![image024](http://www.jorditorres.org/wp-content/uploads/2016/02/image024.png)

A k-means algorithm implemented in TensorFlow to group the above points, for example in four clusters, can be as follows (based on the model proposed by Shawn Simister in his blog[21]):
위에 생성된 포인트들을 텐서플로우를 이용한 K평균 알고리즘으로 4개의 군집을 구현한 예는 아래와 같다. ( Shawn Simister의 블로그에서 구현된 모델 기반) 
```python
import numpy as np
vectors = tf.constant(conjunto_puntos)
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2), 0)

means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in xrange(k)])

update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in xrange(100):
   _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
   I suggest the reader checks the result in the assignment_values tensor with the following code, which generates a graph as above:

   data = {"x": [], "y": [], "cluster": []}

   for i in xrange(len(assignment_values)):
     data["x"].append(conjunto_puntos[i][0])
       data["y"].append(conjunto_puntos[i][1])
         data["cluster"].append(assignment_values[i])

         df = pd.DataFrame(data)
         sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
         plt.show()
```         
The screenshot with the result of the execution of my code it is shown in the following figure:
이 코드를 수행한 결과물의 스크린샷은 아래 그림에서 보여준다.

![image026](http://www.jorditorres.org/wp-content/uploads/2016/02/image026.png)

## New groups 새 그룹

I assume that the reader might feel a little overwhelmed with the K-means code presented in the previous section. Well, I propose that we analyze this in detail, step by step, and especially watch the tensors invoved and how they are transformed during the program.
나는 독자들이 이전 섹션에서 보여진 K평균 코드로 인해 약간의 어려움을 겪을 것이라 느낀다. 그렇지만 우리는 이것을 좀더 자세히 단계별로 특히 구현외 관여된 텐서들이 프로그램이 실행되는 과정에서 변환되는 과정을 분석 할 것이다. 
The first thing to do is move all our data to tensors. In a constant tensor, we keep our entry points randomly generated:
첫 번 째로 할 일은 모든 우리의 데이터를 텐서로 옮기는 것이다. 상수텐서로 우리가 랜덤하게 생성한 포인트들을 치환한다.
```python
vectors = tf.constant(conjunto_vectors)
```
Following the algorithm presented in the previous section, in order to start we must determine the initial centroids. As I advanced, an option may be randomly choose K observations from the input data. One way to do this is with the following code, which indicates to TensorFlow that it must shuffle randomly the entry point and choose the first K points as centroids:
이전장에서 보여주었듯이 다음 알고리즘은, 시작하기 위하여 초기 중심값을 결정하는 것이다. 시작하기 전에 입력 데이터 중 K개 관측값이 랜덤하게 결정 될 수 있다. 이러한 과정이 다음 코드에 나타나 있는데, 초기 중심치 포인트 K를 결정하는 과정을 텐서플로우가 랜덤하게 섞는 것을 나타낸 것이다. 

```python
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))
```
These K points are stored in a 2D tensor. To know the shape of those tensors K
we can use tf.Tensor.get_shape():
K 포인트들은 2D 텐서에 보관되어 있다. 이러한 텐서 K의 형태를 알고 싶으면 
tf.Tensor.get_shape() 를 이용하면된다.

```python
print vectors.get_shape()
print centroides.get_shape()

TensorShape([Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(2)])
```
We can see that vectors is an array that dimension D0 contains 2000 positions, one for each, and D1 contains the position x,y for each point. Instead, centroids is a matrix of four positions in the dimension D0, one position for each centroid, and the dimension D1 is equivalent to the dimension D1 of vectors.
우리는 벡터가 D0차원에서 2000개의 각각의 위치를, D1차원에서 좌표 x, y를 담고 있는 배열임을 볼 수 있다. 반면에 중심치는 4개의 위치 정보를 D0차원에 각각 담고, D1차원에 벡터의 D1과 동등한 정보를 갖고 있다. 

Next, the algorithm enters in a loop. The first step is to calculate, for each point, its closest centroid by the Squared Euclidean Distance[22] (which can only be used when we want to compare distances):
다음으로, 알고리즘이 반복 구문으로 진입한다. 첫번째 단계는 각 포인트들에 대해서 연산하는 것인이고, 유클리디안 제곱 거리를 이용한 최 근접 중심치를 연산한다. (이것은 우리가 거리를 비교할 때 이용할 수 있다.)
![image028](http://www.jorditorres.org/wp-content/uploads/2016/02/image028.jpg)

To calculate this value tf.sub(vectors, centroides) is used. We should note that, although the two subtract tensors have both 2 dimensions, they have different sizes in one dimension (2000 vs 4 in dimension D0), which, in fact, also represent different things.
이 계산을 하기 위해서 tf.sub(vectors, centroides)가 이용된다. 반드시 알아야 할 점은 감산을 할 텐서가 2개의 차원을 가졌는데, 1차원에서 다른 길이를 갖는 점이다. (D0 내 2000 대 4), 사실 D0은 각각 다른 정보를 담고 있다. 

To fix this problem we could use some of the functions discussed before, for instance tf.expand_dims in order to insert a dimension in both tensors. The aim is to extend both tensors from 2 dimensions to 3 dimensions to make the sizes match in order to perform a subtraction:
이러한 문제를 고치기 위하여 우리는 이전에 몇가지 함수에 대해서 논의 할 수 있었다. 예를들어, tf.expand_dims를 이용하여 양측 텐서에 새로운 차원을 삽입한다. 이것의 목적은 양측 텐서를 2차원에서 3차원으로 확장하여 감산 연산을 수행할 수 있도록 크기를 맞추는 것이다.  

```python
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)
```
tf.expand_dims inserts one dimension in each tensor; in the first dimension (D0) of vectors tensor, and in the second dimension (D1) of centroids tensor. Graphically, we can see that in the extended tensors the dimensions have the same meaning in each of them:
tf.expand_dim은 각 텐서에 1개 차원을 삽입한다. 벡터 텐서에는 1차 차원을(D0), 중심치 텐서에는 2차 차원(D1)을 넣는다. 우리는 여기서 그래픽 하게 각 차원의 의미가 각 텐서에서의 같은 의미를 갖는 것을 볼 수 있다.

![image031](http://www.jorditorres.org/wp-content/uploads/2016/02/image031.gif)

It seems to be solved, but actually, if you look closely (outlined in bulk in the illustration), in each case there are dimensions that have not been able to determinate the sizes of those dimensions. Remember that the with get_shape() function we can find out:
문제는 풀린 것으로 보이지만 그림에서 빗금친 영역을 자세히 들여다 보면, 각 차원에서 각각의 사이즈가 결정 되지 않은 것을 볼 수 있다.
get_shape()함수를 이용하여 상황을 확인할 수 있다는 것을 기억해야 한다.

```python
print expanded_vectors.get_shape()
print expanded_centroides.get_shape()
```
The output is as follows:
결과물은 아래와 같다.

```python
TensorShape([Dimension(1), Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(1), Dimension(2)])
```
With 1 it is indicating a no assigned size.
1이 의미하는 바는 사이즈가 할당 되지 않았다는 것을 의미한다.

But I have already advanced that TensorFlow allows broadcasting, and therefore the tf.sub function is able to discover for itself how to do the subtraction of elements between the two tensors.
그러나 나는 이전에 텐서플로우가 브로드캐스팅을 지원한다고 확인했었고, 이것은 tf.sub 함수는 2개의 텐서 사이에 어떻게 감산을 하여야 하는지 스스로 찾아낼 수 있다는 것이다. 

Intuitively, and observing the previous drawings, we see that the shape of the two tensors match, and in those cases both tensors have the same size in a certain dimension. These math, as happens in dimension D2. Instead, in the dimension D0 only has a defined size the expanded_centroides.
직관적으로, 이전 도해를 관찰 했을 때 우리는 두 텐서의 형태가 맞출 수 있는 형태이고, 이 형태는 특정한 차원 영역에서 같은 크기의 경우를 갖는다. 이 연산은 D2 차원에서 일어난다. 

In this case, TensorFlow assumes that the dimension D0 of expanded_vectors tensor have to be the same size if we want to perform a subtraction element to element within this dimension.
이러한 경우, 만일 우리가 이 차원(D2) 차원에서 점대 점 감산 연산을 하고자 한다면 텐서플로우는 expaneded_vectors의 텐서를 같은 사이즈로 가정하정 하도록 한다.

And the same happens with the size of the dimension D1 of expended_centroides tensor, where TensorFlow deduces the size of the dimension D1 of expanded_vectors tensor.
그리고 D1 차원의 expended_centroid 텐서공간에서 expandec_vector 텐서와 같은 사이즈를 도출할 때에도 같은 작업을 하게 된다. 

Therefore, in the allocation step (step 1) the algorithm can be expressed in these four lines of TensorFlow´s code, which calculates the Squared Euclidean Distance:
그러므로, 할당 단계(step 1) 알고리즘은 유클리디안 제곱 거리를 계산하는 4줄의 텐서플로우의 코드로 표현할 수 있다. 

```python
diff=tf.sub(expanded_vectors, expanded_centroides)
sqr= tf.square(diff)
distances = tf.reduce_sum(sqr, 2)
assignments = tf.argmin(distances, 0)
```
And, if we look at the shapes of tensors, we see that they are respectively for diff, sqr, distances and assignments as follows:
그리고, 텐서의 형태를 관찰한다면 우리는 'diff', 'sqr', 'distances', 'assignment' 각각의 형태를 아래와 같이 볼 수 있다.
```python
TensorShape([Dimension(4), Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(2000)])
TensorShape([Dimension(2000)])
```
That is, tf.sub function has returned the tensor dist, that contains the subtraction of the index values for centroids and vector (indicated in the dimension D1, and the centroid indicated in the dimension D0. For each index x,y are indicated in the dimension D2) .
즉, tf.sub 함수는 텐서 거리를 반환하고 이것은 중심치와 벡터의 감산들의 인덱스 값의 차를 갖고 있다. 
(이것은 D1 차원을 가리키고, 중심치는 D0 차원을 가리킨다. 각각 x, y 는 D2차원을 가리키고 있다.)

The sqr tensor contains the square of those. In the distance tensor we can see that it has already reduced one dimension, the one indicated as a parameter in tf.reduce_sum function.
sqr 텐서는 이것들의 제곱 값을 갖고 있고,  tf.reduce_sum 함수의 파라메터로 축약된 차원의 거리값 텐서의  결과물을 볼 수 있다. 

I use this example to explain that TensorFlow provides several operations which can be used to perform mathematical operations that reduce dimensions of a tensor as in the case of tf.reduce_sum. In the table below you can find a summary of the most important ones:
나는 이 예제를 텐서플로우의 제공된 다수의 연산들, 예를들어 tf.reduce_sum과 같이 차원 축약에 쓰이는 텐서의 수학적 연산들, 을 설명하는데 이용하였다. 아래 표에서 여러분은 이러한 종류의 중요한 연산들의 요약을 볼 수 있다. 

Operation      | Description
---------------|----------------------------------------------------------
tf.reduce_sum  | Computes the sum of the elements along one dimension
tf.reduce_prod | Computes the product of the elements along one dimension
tf.reduce_min  | Computes the minimum of the elements along one dimension
tf.reduce_max  | Computes the maximum of the elements along one dimension
tf.reduce_mean | Computes the mean of the elements along one dimension

    연산       |   설명 
---------------|----------------------------------------------------------
tf.reduce_sum  | 하나의 차원을 따라 원소들의 합을 구함
tf.reduce_prod | 하나의 차원을 따라 원소들의 곱을 구함
tf.reduce_min  | 하나의 차원을 따라 원소들의 최소값을 구함
tf.reduce_max  | 하나의 차원을 따라 원소들의 최대값을 구함
tf.reduce_mean | 하나의 차원을 따라 원소들의 평균을 구함

Finally, the assignation is achieved with tf.argmin, which returns the index with the minimum value of the tensor dimension (in our case D0, which remember that was the centroid). We also have the tf.argmax operation:
마지막으로 tf.argmin으로 부터 획득한 최소값의 텐서 차원(우리의 경우 중심치를 갖고 있는 D0 차원)의 인덱스를 할당하는 것이 달성 되었다. 

Operation  |  Description
-----------|-------------------------------------------------------------------------------------
tf.argmin  |  Returns the index of the element with the minimum value along tensor dimension
tf.argmax  |  Returns the index of the element with the maximum value of the tensor dimension

  연산     |   설명 
-----------|-------------------------------------------------------------------------------------
tf.argmin  | tensor의 차원을 따라 구한 최소값의 인덱스를 반환함
tf.argmax  | tensor의 차원을 따라 구한 최대값의 인덱스를 반환함


In fact, the 4 instructions seen above could be summarized in only one code line, as we have seen in the previous section:
사실 이전 섹션에서 봤던 저 4 가지 명령문은 한줄로 요약될 수 있다. 

```python
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2), 0)
```
But anyway, internal tensors and the operations that they define as nodes and execute the internal graph are like the ones we have described before.
이렇게 되더라도,  내부 텐서와 노드로 정의된 연산들 그리고 내부 그래프는 이전에 구술한것과 유사하다.

# . Computation of the new centroids 새 중심치 계산

Once we have created new groups on each iteration, we will have to remember that the new step of the algorithm consists in calculating the new centroids of the groups. In the code of the section before we have seen this line of code:
각 반복을 통해 우리가 새 그룹을 만들었다면, 명심해야 할 것은 다음 단계의 알고리즘은 해당 그룹에서 새로운 중심치를 계산하는 것으로 이루어 진다는 것이다. 이전 섹션에서 우리는 이 코드를 보았다.

```python
means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in xrange(k)])
```
On that piece of code, we can see that the means tensor is the result of the concatenation of the k tensors that correspond to the mean value of every point that belongs to each k cluster.
이 코드 한줄로 우리는 평균값 텐서를 볼 수 있다.  평균값 텐서는 각각의 k군집에 속해있는 모든 포인트들의 평균 값에 대응하는 k 텐서의 연립으로 된 결과물이다. 

Next, I will comment on each of the TensorFlow operations that are involved in the calculation of the mean value of every points that belongs to each cluster[23]:
각 군집에 속하여있는 모든 포인트들의 평균 값의 연산에 참여하는 텐서 연산에 대하여 설명할 것이다.

With equal we can obtain a boolean tensor (Dimension(2000)) that indicates (with true value) the positions where the assignment tensor match with the K cluster, which, at the time, we are calculating the average value of the points.
With where is constructed a tensor (Dimension(1) x Dimension(2000)) with the position where the values true are on the boolean tensor received as a parameter. i.e. a list of the position of these.
With reshape is constructed a tensor (Dimension(2000) x Dimension(1)) with the index of the points inside vectors tensor that belongs to this c cluster.
With gather is constructed a tensor (Dimension(1) x Dimension(2000)) which gathers the coordenates of the points that form the c cluster.
With reduce_mean it is constructed a tensor (Dimension(1) x Dimension(2)) that contains the average value of all points that belongs to the cluster c.
Anyway, if the reader wants to dig deeper into the code, as I always say, you can find more info for each of these operations, with very illustrative examples, on the TensorFlow API page[24].

equal 연산으로 우리는 논리형(bool) 텐서를 구할 것이고(길이 2000의 차원), 이것은 k군집과 대응하는 assignments 텐서의 위치를 나태고 있다. 이와 동시에 우리는 그 점들의 평균값을 계산하고 있다. 
where 연산을 통해, 파라메터로 받은 논리값(bool) 텐서의 참 값의 위치로 (길이 1 차원 x 길이 2000 차원)의 텐서를 형성하고 (이를테면 참값들의 list)
reshape 연산을 이용하여, c클러스터에 속한 벡터 텐서 내부에 있는 포인트들의 인덱스들을 구성한다 (길이 2000 차원 x 길이 1 차원)
gather 연산으로 c군집을 형성하는 포인트들의 좌표들을 수집하여 텐서를 형성하고 (길이 1 차원 x 길이 2000 차원)
reduce_mean 연산으로 c군집내 속한 모든 포인트들의 평균 값을 같는 텐서(길이 1 차원 x 길이 2 차원)을 형성한다. 
어쨌든 만일 독자중에 이 코드에 대해 좀더 자세히 들어가고 싶다면 나는 항상 말하는데, 독자 여러분은 TensorFlow API page[24]에서 아주 설명에 도움이 되는 실례를 찾을 수 있다.

## Graph execution 그래프 실행

Finally, we have to describe the part of the above code that corresponds to the loop and to the part that update the centroids with the new values of the means tensor.
마지막으로, 루프 및 평균 텐서의 새로운 값으로 중심값들을 갱신하는 부분에 대응하는 코드를 설명한다.

To do this, we need to create an operator that assigns the value of the variable means tensor into centroids in a way than, when the operation run() is executed, the values of the updated centroids are used in the next iteration of the loop:
이를 위해, 오퍼레이션 run() 이 실행될 때, 평균 텐서의 값들을 중심값들에 할당하는 연산자를 만들어야 한다. 업데이트 된 중심값들의 값은 루프의 다음 이터레이션에서 사용된다:

```python
update_centroides = tf.assign(centroides, means)
```
We also have to create an operator to initialize all of the variable before starting to run the graph:
또한 그래프의 실행을 시작하기 전에, 모든 변수를 초기화하는 연산자도 만들어야 한다:

```python
init_op = tf.initialize_all_variables()
```
At this point everything is ready. We can start running the graph:
모든 준비가 끝났다. 이제 그래프를 그려보자:

```python
sess = tf.Session()
sess.run(init_op)

for step in xrange(num_steps):
   _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
```
In this code, for each iteration, the centroids and the new allocation of clusters for each entry points are updated.
이 코드에선, 각 이터레이션에 대해 중심값과 각 지점들에 대한 클러스터의 새로운 할당이 갱신된다.

Notice that the code specifies three operators and it has to go look in the execution of the call run(), and running in this order. Since there are three values to search, sess.run() returns a data structure of three numpy array elements with the contents of the corresponding tensor during the training process.
세 개의 연산자를 지정하는 코드에 주의하라. run() 호출의 실행으로 가야하며 위의 순서대로 실행된다. 찾아야 할 값이 3개이므로, sess.run() 는 트레이닝 진행 과정에서 해당 텐서의 내용과 함께 3개의 numpy 배열 요소의 데이터 구조를 리턴한다.

As update_centroides is an operation whose result is not the parameter that returns, the corresponding item in the return tuple contains nothing, and therefore be ruled out, indicating it with “_” [25].
update_centroides 는 리턴되는 파라미터가 아닌 동작이기 때문에, 리턴되는 투플의 해당 아이템은 아무것도 포함하지 않는다. 따라서 "_" 로 지정하면 제외된다[25]

For the other two values, the centroids and the assigning points to each cluster, we are interested in displaying them on screen once they have completed all num_steps iterations.
중심값과 각 클러스터에 할당된 포인트들, 두 가지의 값에 대해, 우리는 모든 num_steps 이터레이션을 완료한 후 그것들을 화면에 표시하는데 관심이 있습니다.

We can use a simple print. The output is as it follows:
간단한 출력을 해보자. 결과는 다음과 같다:

```python
print centroid_values

[[ 2.99835277e+00 9.89548564e-01]
[ -8.30736756e-01 4.07433510e-01]
[ 7.49640584e-01 4.99431938e-01]
[ 1.83571398e-03 -9.78474259e-01]]
```
I hope that the reader has a similar values on the screen, since this will indicate that he has successfully executed the proposed code in this chapter of the book.
나는 독자들이 화면의 결과와 비슷한 값을 얻기를 바란다. 그것은 이번 챕터에서 제공한 코드의 실행 결과가 성공적임을 나타내주기 때문이다.

I suggest that the reader tries to change any of the values in the code, before advancing. For example the num_points, and especially the number of clustersk, and see how it changes the result in the assignment_values tensor with the previous code that generates a graph.
다음 단계로 나아가기 전에, 코드의 값들 중 어떤 것이라도 바뀌보기를 권장한다. 예를 들면 num_points, 특히 clustersk 의 값들이 그것이다. 그리고 그래프를 그리는 이전의 코드와 비교해서 assignment_values 텐서의 결과가 어떻게 바뀌는지 보기 바란다.

Remember that in order to facilitate testing the code described in this chapter, it can be downloaded from Github[26] . The name of the file that contains this code is Kmeans.py.
이 장에서 설명하는 코드의 테스트 용이하게하기 위해 기억하자, 그것은 Github[26]에서 다운로드 할 수 있다. 이 코드가 포함 된 파일의 이름은 Kmeans.py 이다.

In this chapter we have advanced some knowledge of TensorFlow, especially on basic data structure tensor, from a code example in TensorFlow that implements a clustering algorithm K-means.
이 장에서 우리는 클러스터링 알고리즘 K-평균을 구현한 TensorFlow의 코드 예제를 통해, 특히 기본 데이터 구조 텐서에 대해서  TensorFlow의 지식을 습득했다.

With this knowledge, we are ready to build a single layer neural network, step by step, with TensorFlow in the next chapter.
이러한 지식을 바탕으로, 우리는 다음 챕터에서 텐서플로우를 사용한 단일층의 신경 네트워크 구축을 다룰 준비가 되었다.

[contents link]
