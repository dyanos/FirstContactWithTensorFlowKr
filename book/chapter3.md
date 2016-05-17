3. CLUSTERING IN TENSORFLOW
3. 텐서 플로우의 군집화

Linear regression, which has been presented in the previous chapter, is a supervised learning algorithm in which we use the data and output values (or labels) to build a model that fits them. But we haven’t always tagged data, and despite this we also want analyze them in some way. In this case, we can use an unsupervised learning algorithm as clustering. The clustering method is widely used because it is often a good approach for preliminary screening data analysis.
이전 장에서 설명했던 선형 회귀는, 데이터와 결과물( 또는 라벨들)로 피팅(fitting)할 모델을 생성하는 지도학습을 말하는 것이었다. 그러나 데이터가 항상 키워드로 분류되어 있는것은 아니다. 그럼에도, 우리는 이러한 데이터를 어떻게 해서든 분석하고 싶어한다. 이러한 경우 우리는 군집화라고 불리는 비지도학습 알고리즘을 이용할 수 있다. 군집화 방식이 두루 쓰이고 있는데, 예비 선별하는 데이터 분석에 유용한 방식이기 때문이다. 

In this chapter, I will present the clustering algorithm called K-means. It is surely the most popular and widely used to automatically group the data into coherent subsets so that all the elements in a subset are more similar to each other than with the rest. In this algorithm, we do not have any target or outcome variable to predict estimations.
이번 장에서, K평균 알고리즘을 설명 할 것이다. 이것은 데이터를 자동적으로 질서있는 집합으로 무리지을 때 가장 인기 있고 폭 넓게 쓰이고 있다. 각 부분 집합들의 모든 요소들은 나머지 집합의 요소들 보다 해당 집합내에서 보다 유사성을 갖는다. 

I will also use this chapter to achieve progress in the knowledge of TensorFlow and go into more detail in the basic data structure called tensor. I will start by explaining what this type of data is like and present the transformations that can be performed on it. Then, I will show the use of K-means algorithm in a case study using tensors.
나는 이번 장을 통해서 텐서 플로우에 대한 지식을 확장 시킬 것이고 텐서라고 불리는 기본 데이터 구조에 대해 좀 더 구체적으로 진입할 것이다. 나는 이 데이터 형태가 어떠한 것인지 설명하고 그것에 대한 변환 이행을 보여줄 것이다. 그 뒤, 나는 텐서를 이용한 K평균 알고리즘의 사용을 바여줄 것이다.

Basic data structure: tensor
기본 데이터 구조 : 텐서

TensorFlow programs use a basic data structure called tensor to represent all of their datum. A tensor can be considered a dynamically-sized multidimensional data arrays that have as a properties a static data type, which can be from boolean or string to a variety of numeric types. Below is a table of the main types and their equivalent in Python.
텐서플로우 프로그램들은 모든 데이터 자료를 표현하는데 있어서 텐서라는 기본 데이터형을 이용한다. 텐서는 불 연산자나 문자열, 다양한 수치형 데이터 같은 정적 데이터 속성을 갖는 동적 크기의 다차원 배열이라고 말할 수 있다.  아래 표는 주요 데이터 형과 파이썬에 대응하는 데이터형에 대한 것이다.
 
 Type in TensorFlow | Type in Python | Description
--------------------|----------------|---------------
 DT_FLOAT           | tf.float32     | Floating point of 32 bits
 DT_INT16           | tf.int16       | Integer of 16 bits
 DT_INT32           | tf.int32       | Integer of 32 bits
 DT_INT64           | tf.int64       | Integer of 64 bits
 DT_STRING          | tf.string      | String
 DT_BOOL            | tf.bool        | Boolean

 In addition, each tensor has a rank, which is the number of its dimensions. For example, the following tensor (defined as a list in Python) has rank 2:
 참고로, 각기 텐서는 랭크를 갖는데 이것은 그것의 차수를 말한다. 예를 들어 아래 텐서에서 (파이썬 list로 정의된) 랭크 2를 갖는다. 

 t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

 Tensors can have any rank. A rank 2 tensor is usually considered a matrix, and a rank 1 tensor would be a vector. Rank 0 is considered a scalar value.
 텐서는 어느 랭크나 가질수 있다. 랭크 2를 갖는 텐서는 행렬을 의미한다. 랭크 1을 갖는 텐서는 벡터라 할 수 있다. 랭크 0 는 스칼라값을 말한다. 

 TensorFlow documentation uses three types of naming conventions to describe the dimension of a tensor: Shape, Rank and Dimension Number. The following table shows the relationship between them in order to make easier the Tensor Flow documentation’s traking easier:
 텐서플로우 문서는 형태, 랭크, 차수 라는 3 가지 형태의 네이밍 조항을 쓰고 있는다.  아래 표는 텐서 플로우 문서를 쉽게 이해할 수 있도록 이들의 관계를 표현한 것이다. 

 Shape             | Rank | Dimension Number
 --------------    |------|--------------------
 []                |    0 | 0-D
 [D0]              |    1 | 1-D
 [D0, D1]          |    2 | 2-D
 [D0, D1, D2]      |    3 | 3-D
           ~       |    ~ |  ~
 [D0, D1,  ~ Dn]   |    n | n-D
 
 These tensors can be manipulated with a series of transformations that supply the TensorFlow package. Below, we discuss some of them in the next table.
 이들 텐서들은 텐서플로우 패키지를 변환하는 일련의 작업으로 다뤄질 수 있다. 하단에서, 우리는 이들 줄 일부를 다음 표에서 다룬다.

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
   

   For example, suppose that you want to extend an array of 2×2000 (a 2D tensor) to a cube (3D tensor). We can use the tf.expand_ dims function, which allows us to insert a dimension to a tensor:
   실례로, 2x2000 의 2차원 텐서 어레이를 3차원 텐서 형태로 확장한다고 해보자. 우리는 텐서 안에 차원을 추가해주는 tf.expand_dims 함수를 사용할 수 있다.

```python
vectors = tf.constant(conjunto_puntos)
extended_vectors = tf.expand_dims(vectors, 0)
In this case, tf.expand_dims inserts a dimension into a tensor in the one given in the argument (the dimensions start at zero).
```
Visually, the above transformation is as follows:
위의 변환을 시각화하면 다음과 같다.

image023

As you can see, we now have a 3D tensor, but we cannot determine the size of the new dimension D0 based on function arguments.
보는바와 같이, 이제 우리는 3차원 텐서를 가지게 되었다. 그러나, 우리는 함수 인수에 기반한 새로운 차원의 D0 의 사이즈는 알 수 없다.

If we obtain the shape of this tensor with the get_shape() operation, we can see that there is no associated size:
get_shape() 함수를 통해 이 텐서의 모양을 알아낸다면, 우리는 관련된 사이즈가 없음을 알 수 있다.

print expanded_vectors.get_shape()

It appears on the screen like:
화면에 다음과 같이 나온다:

```python
TensorShape([Dimension(1), Dimension(2000), Dimension(2)])
```
Later in this chapter, we will see that, thanks to TensorFlow shape broadcasting, many mathematical manipulation functions of tensors (as presented in the first chapter), are able to discover for themselves the size in the dimension which unspecific size and assign to it this deduced value.
나중에 이 장에서, 우리는 (1장에서 다루어 졌던) 텐서의 많은 수학적 변형 함수들이 함수들 스스로 불특정한 크기의 차원의 크기를 발견할 수 있음과 그것을 추론된 값에 할당할 수 있음을 보게될 것이다.


Data Storage in TensorFlow
텐서 플로우에서의 데이터 저장

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

Data files
Usually, the initial data is downloaded from a data file. The process is not complex, and given the introductory nature of this book I invite the reader to visit the website of TensorFlow[19] for more details on how to download data from different file types. You can also review the Python code input_data.py[20](available on the Github book), which loads the MNIST data from files (I will use this in the following chapters).
데이터 파일
일반적으로, 초기 데이터는 데이터 파일로부터 다운로드되어진다. 이 과정은 복잡하지 않다. 그리고 이 책의 서두에서 언급한것 처럼, 다른 파일 타입으로부터 어떻게 데이터를 다운받는지에 대한 자세한 사항에 대해서는 텐서플로우[19]의 웹사이트를 방문하길 추천한다. 파일에서 MNIST 데이터를 읽는 input_data.py[20](깃헙6 북 에서도 가능하다) 의 파이썬 코드를 읽어볼 수도 있다.(나는 다음 챕터에서 이것을 사용할 것이다)

Variables and constants
When it comes to small sets, data can also be found pre-loaded into memory; there are two basic ways to create them, as we have seen in the previous example:
변수와 상수
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
 
In TensorFlow, during the training process of the models, the parameters are maintained in the memory as variables. When a variable is created, you can use a tensor defined as a parameter of the function as an initial value, which can be a constant or a random value. TensorFlow offers a collection of operations that produce random tensors with different distributions:

     

Operation           | Description
--------------------|------------------------------------------------------------------------------------------------------------------------------------
tf.random_normal    | Random values with a normal distribution
tf.truncated_normal | Random values with a normal distribution but eliminating those values whose magnitude is more than 2 times the standard deviation
tf.random_uniform   | Random values with a uniform distribution
tf.random_shuffle   | Randomly mixed tensor elements in the first dimension
tf.set_random_seed  | Sets the random seed
 

An important detail is that all of these operations require a specific shape of the tensors as the parameters of the function, and the variable that is created has the same shape. In general, the variables have a fixed shape, but TensorFlow provides mechanisms to reshape it if necessary.

When using variables, these must be explicitly initialized after the graph that has been constructed, and before any operation is executed with the run() function. As we have seen, it can be used tf.initialize_all_variables() for this purpose. Variables also can be saved onto disk during and after training model through TensorFlow tf.train.Saver() class, but this class is beyond the scope of this book.

Provided by Python code
Finally, we can use what we have called “symbolic variable” or placeholder to manipulate data during program execution. The call is placeholder(), which includes arguments with the type of the elements and the shape of the tensor, and optionally a name.

At the same time as making the calls to Session.run() or Tensor.eval() from the Python code, this tensor is populated with the data specified in the feed_dict parameter. Remember the first code in Chapter 1:

```python
import tensorflow as tf
a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.mul(a, b)
sess = tf.Session()
print   sess.run(y, feed_dict={a: 3, b: 3})
```
In the last line of code, when the call sess.run() is made, it is when we pass the values of the two tensors a and b through feed_dict parameter.

With this brief introduction about tensors, I hope that from now on the reader can follow the codes of the following chapters without any difficulty.

K-means algorithm

K-means is a type of unsupervised algorithm which solves the clustering problem. Its procedure follows a simple and easy way to classify a given data set through a certain number of clusters (assume k clusters). Data points inside a cluster are homogeneous and heterogeneous to peer groups, that means that all the elements in a subset are more similar to each other than with the rest.

The result of the algorithm is a set of K dots, called centroids, which are the focus of the different groups obtained, and the tag that represents the set of points that are assigned to only one of the K clusters. All the points within a cluster are closer in distance to the centroid than any of the other centroids.

Making clusters is a computationally expensive problem if we want to minimize the error function directly (what is known as an NP-hard problem); and therefore it some algorithms that converge rapidly in a local optimum by heuristics have been created. The most commonly used algorithm uses an iterative refinement technique, which converges in a few iterations.

Broadly speaking, this technique has three steps:

Initial step (step 0): determines an initial set of K centroids.
Allocation step (step 1): assigns each observation to the nearest group.
Update step (step 2): calculates the new centroids for each new group.
There are several methods to determine initial K centroids. One of them is randomly choose K observations in the data set and consider them centroids; this is the one we will use in our example.

The steps of allocation (step 1) and updating (step 2) are being alternated in a loop until it is considered that the algorithm has converged, which may be for example when allocations of points to groups no longer change.

Since this is a heuristic algorithm, there is no guarantee that it converges to the global optimum, and the outcome depends on the initial groups. Therefore, as the algorithm is generally very fast, it is usual to repeat executions multiple times with different values of the initials centroides, and then weigh the result.

To start coding our example of K-means in TensorFlow I suggest to first generate some data as a testbed. I propose to do something simple, like generating 2,000 points in a 2D space in a random manner, following two normal distributions to draw up a space that allows us to better understand the outcome. For example, I suggest the following code:

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

If you do not have these packages installed, you must do it with the pip value before you can run the following codes.

To display the points that have been generated randomly I suggest the following code:

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame({"x": [v[0] for v in conjunto_puntos],
"y": [v[1] for v in conjunto_puntos]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()
```
This code generates a graph of points in a two dimensional space like the following screenshot:

image024

A k-means algorithm implemented in TensorFlow to group the above points, for example in four clusters, can be as follows (based on the model proposed by Shawn Simister in his blog[21]):

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

image026

New groups

I assume that the reader might feel a little overwhelmed with the K-means code presented in the previous section. Well, I propose that we analyze this in detail, step by step, and especially watch the tensors invoved and how they are transformed during the program.

The first thing to do is move all our data to tensors. In a constant tensor, we keep our entry points randomly generated:

```python
vectors = tf.constant(conjunto_vectors)
```
Following the algorithm presented in the previous section, in order to start we must determine the initial centroids. As I advanced, an option may be randomly choose K observations from the input data. One way to do this is with the following code, which indicates to TensorFlow that it must shuffle randomly the entry point and choose the first K points as centroids:

```python
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))
These K points are stored in a 2D tensor. To know the shape of those tensors we can use tf.Tensor.get_shape():

print vectors.get_shape()
print centroides.get_shape()

TensorShape([Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(2)])
```
We can see that vectors is an array that dimension D0 contains 2000 positions, one for each, and D1 contains the position x,y for each point. Instead, centroids is a matrix of four positions in the dimension D0, one position for each centroid, and the dimension D1 is equivalent to the dimension D1 of vectors.

Next, the algorithm enters in a loop. The first step is to calculate, for each point, its closest centroid by the Squared Euclidean Distance[22] (which can only be used when we want to compare distances):

image028

To calculate this value tf.sub(vectors, centroides) is used. We should note that, although the two subtract tensors have both 2 dimensions, they have different sizes in one dimension (2000 vs 4 in dimension D0), which, in fact, also represent different things.

To fix this problem we could use some of the functions discussed before, for instance tf.expand_dims in order to insert a dimension in both tensors. The aim is to extend both tensors from 2 dimensions to 3 dimensions to make the sizes match in order to perform a subtraction:

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)
tf.expand_dims inserts one dimension in each tensor; in the first dimension (D0) of vectors tensor, and in the second dimension (D1) of centroids tensor. Graphically, we can see that in the extended tensors the dimensions have the same meaning in each of them:

image031

It seems to be solved, but actually, if you look closely (outlined in bulk in the illustration), in each case there are dimensions that have not been able to determinate the sizes of those dimensions. Remember that the with get_shape() function we can find out:

```python
print expanded_vectors.get_shape()
print expanded_centroides.get_shape()
```
The output is as follows:
```python
TensorShape([Dimension(1), Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(1), Dimension(2)])
```
With 1 it is indicating a no assigned size.

But I have already advanced that TensorFlow allows broadcasting, and therefore the tf.sub function is able to discover for itself how to do the subtraction of elements between the two tensors.

Intuitively, and observing the previous drawings, we see that the shape of the two tensors match, and in those cases both tensors have the same size in a certain dimension. These math, as happens in dimension D2. Instead, in the dimension D0 only has a defined size the expanded_centroides.

In this case, TensorFlow assumes that the dimension D0 of expanded_vectors tensor have to be the same size if we want to perform a subtraction element to element within this dimension.

And the same happens with the size of the dimension D1 of expended_centroides tensor, where TensorFlow deduces the size of the dimension D1 of expanded_vectors tensor.

Therefore, in the allocation step (step 1) the algorithm can be expressed in these four lines of TensorFlow´s code, which calculates the Squared Euclidean Distance:

```python
diff=tf.sub(expanded_vectors, expanded_centroides)
sqr= tf.square(diff)
distances = tf.reduce_sum(sqr, 2)
assignments = tf.argmin(distances, 0)
```
And, if we look at the shapes of tensors, we see that they are respectively for diff, sqr, distances and assignments as follows:

```python
TensorShape([Dimension(4), Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(2000)])
TensorShape([Dimension(2000)])
```
That is, tf.sub function has returned the tensor dist, that contains the subtraction of the index values for centroids and vector (indicated in the dimension D1, and the centroid indicated in the dimension D0. For each index x,y are indicated in the dimension D2) .

The sqr tensor contains the square of those. In the distance tensor we can see that it has already reduced one dimension, the one indicated as a parameter in tf.reduce_sum function.

I use this example to explain that TensorFlow provides several operations which can be used to perform mathematical operations that reduce dimensions of a tensor as in the case of tf.reduce_sum. In the table below you can find a summary of the most important ones:

Operation      | Description
---------------|----------------------------------------------------------
tf.reduce_sum  | Computes the sum of the elements along one dimension
tf.reduce_prod | Computes the product of the elements along one dimension
tf.reduce_min  | Computes the minimum of the elements along one dimension
tf.reduce_max  | Computes the maximum of the elements along one dimension
tf.reduce_mean | Computes the mean of the elements along one dimension

Finally, the assignation is achieved with tf.argmin, which returns the index with the minimum value of the tensor dimension (in our case D0, which remember that was the centroid). We also have the tf.argmax operation:

Operation  |  Description
-----------|-------------------------------------------------------------------------------------
tf.argmin  |  Returns the index of the element with the minimum value along tensor dimension
tf.argmax  |  Returns the index of the element with the maximum value of the tensor dimension

In fact, the 4 instructions seen above could be summarized in only one code line, as we have seen in the previous section:

```python
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2), 0)
```
But anyway, internal tensors and the operations that they define as nodes and execute the internal graph are like the ones we have described before.

Computation of the new centroids

Once we have created new groups on each iteration, we will have to remember that the new step of the algorithm consists in calculating the new centroids of the groups. In the code of the section before we have seen this line of code:

means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in xrange(k)])
On that piece of code, we can see that the means tensor is the result of the concatenation of the k tensors that correspond to the mean value of every point that belongs to each k cluster.

Next, I will comment on each of the TensorFlow operations that are involved in the calculation of the mean value of every points that belongs to each cluster[23]:

With equal we can obtain a boolean tensor (Dimension(2000)) that indicates (with true value) the positions where the assignment tensor match with the K cluster, which, at the time, we are calculating the average value of the points.
With where is constructed a tensor (Dimension(1) x Dimension(2000)) with the position where the values true are on the boolean tensor received as a parameter. i.e. a list of the position of these.
With reshape is constructed a tensor (Dimension(2000) x Dimension(1)) with the index of the points inside vectors tensor that belongs to this c cluster.
With gather is constructed a tensor (Dimension(1) x Dimension(2000)) which gathers the coordenates of the points that form the c cluster.
With reduce_mean it is constructed a tensor (Dimension(1) x Dimension(2)) that contains the average value of all points that belongs to the cluster c.
Anyway, if the reader wants to dig deeper into the code, as I always say, you can find more info for each of these operations, with very illustrative examples, on the TensorFlow API page[24].

Graph execution

Finally, we have to describe the part of the above code that corresponds to the loop and to the part that update the centroids with the new values of the means tensor.

To do this, we need to create an operator that assigns the value of the variable means tensor into centroids in a way than, when the operation run() is executed, the values of the updated centroids are used in the next iteration of the loop:

```python
update_centroides = tf.assign(centroides, means)
```
We also have to create an operator to initialize all of the variable before starting to run the graph:

```python
init_op = tf.initialize_all_variables()
```
At this point everything is ready. We can start running the graph:
```python
sess = tf.Session()
sess.run(init_op)

for step in xrange(num_steps):
   _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
```
In this code, for each iteration, the centroids and the new allocation of clusters for each entry points are updated.

Notice that the code specifies three operators and it has to go look in the execution of the call run(), and running in this order. Since there are three values to search, sess.run() returns a data structure of three numpy array elements with the contents of the corresponding tensor during the training process.

As update_centroides is an operation whose result is not the parameter that returns, the corresponding item in the return tuple contains nothing, and therefore be ruled out, indicating it with “_” [25].

For the other two values, the centroids and the assigning points to each cluster, we are interested in displaying them on screen once they have completed all num_steps iterations.

We can use a simple print. The output is as it follows:

```python
print centroid_values

[[ 2.99835277e+00 9.89548564e-01]
[ -8.30736756e-01 4.07433510e-01]
[ 7.49640584e-01 4.99431938e-01]
[ 1.83571398e-03 -9.78474259e-01]]
```
I hope that the reader has a similar values on the screen, since this will indicate that he has successfully executed the proposed code in this chapter of the book.

I suggest that the reader tries to change any of the values in the code, before advancing. For example the num_points, and especially the number of clustersk, and see how it changes the result in the assignment_values tensor with the previous code that generates a graph.

Remember that in order to facilitate testing the code described in this chapter, it can be downloaded from Github[26] . The name of the file that contains this code is Kmeans.py.

In this chapter we have advanced some knowledge of TensorFlow, especially on basic data structure tensor, from a code example in TensorFlow that implements a clustering algorithm K-means.

With this knowledge, we are ready to build a single layer neural network, step by step, with TensorFlow in the next chapter.

[contents link]
