번역자 : 송한나님

2. LINEAR REGRESSION IN TENSORFLOW
2. 텐서플로우에서의 선형회귀

In this chapter, I will begin exploring TensorFlow’s coding with a simple model: Linear Regression. Based on this example, I will present some code basics and, at the same time, how to call various important components in the learning process, such as the cost function or the algorithm gradient descent.
이 장에서는 간단한 모델인 선형회귀(linear regression)를 이용해 텐서플로우의 코딩을 향한 탐험을 시작할 것이다. 이 사례에 근거해 약간의 기초적인 코드를 보여주고, 동시에 비용함수(cost function)나 경사하강법(gradient descent)과 같이 학습 과정에 쓰이는 여러 중요한 요소들을 호출하는 법을 보여줄 것이다. 

* Model of relationship between variables
* 변수간 관계에 대한 모델
 
Linear regression is a statistical technique used to measure the relationship between variables. Its interest is that the algorithm that implements it is not conceptually complex, and can also be adapted to a wide variety of situations. For these reasons, I have found it interesting to start delving into TensorFlow with an example of linear regression.
선형회귀는 변수들 사이의 관계를 측정하는 데 쓰이는 통계기법이다. 선형회귀를 실행하는 알고리즘은 개념적으로 복잡하지 않으면서도 매우 다양한 상황에 적용가능한지에 관심을 둔다. 이런 이유들 때문에 나는 선형회귀의 사례를 통해 텐서플로우에 대한 탐색을 시작하는 것이 흥미롭다고 생각했다.

Remember that both, in the case of two variables (simple regression) and the case of more than two variables (multiple regression), linear regression models the relationship between a dependent variable, independent variables xi and a random term b.
2개의 변수를 다루는 경우(단순회귀 simple regression)와 2개 이상의 변수를 다루는 경우(다중회귀 multiple regression) 모두에서, 선형회귀는 종속변수(dependent variable), 독립변수(independent variable) xi 그리고 오차(random term) b 사이의 관계를 모델링한다는 사실을 기억하라. 

In this section I will create a simple example to explain how TensorFlow works assuming that our data model corresponds to a simple linear regression as y = W * x + b. For this, I use a simple Python program that creates data in a two-dimensional space, and then I will ask TensorFlow to look for the line that fits the best in these points.
이 장에서는 텐서플로우가 우리의 데이터 모델이 y = W * x + b 로 표현되는 단순한 선형회귀 모델과 일치한다고 추정하며 동작하는 방법을 설명하기 위해 간단한 사례를 만들어볼 것이다. 이 작업을 위해 2차원 공간에 데이터를 생성하는 간단한 파이썬 프로그램을 사용한 다음, 텐서플로우가 이 점들에 가장 잘 적합시킨(fit) 직선을 구하도록 할 것이다. 

The first thing to do is to import the NumPy package that we will use to generate points. The code we have created is as it follows:
가장 먼저 할 일은 점들을 생성하는 데 사용할 NumPy 패키지를 가져오는 일이다. 우리가 만든 코드는 다음과 같다: 

```
import numpy as np
 
num_points = 1000
vectors_set = []
for i in xrange(num_points):
  x1= np.random.normal(0.0, 0.55)
  y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
  vectors_set.append([x1, y1])
   
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
```

As you can see from the code, we have generated points following the relationship y = 0.1 * x + 0.3, albeit with some variation, using a normal distribution, so the points do not fully correspond to a line, allowing us to make a more interesting example.
코드에서 볼 수 있듯이, 정규분포를 사용하여 약간의 편차가 있긴 하지만 y = 0.1 * x + 0.3의 관계를 따르는 점들을 생성했기 때문에, 점들이 선과 완전히 일치하지 않아서 좀더 흥미로운 사례를 만들도록 해준다. 

In our case, a display of the resulting cloud of points is:
이 경우 수많은 점들이 배치된 결과는 다음 그림과 같다. 

image014

The reader can view them with the following code (in this case, we need to import some of the functions of matplotlib package, running pip install matplotlib[13]):
독자는 다음의 코드에서 그 점들을 볼 수 있다 (이 경우 pip install matplotlib을 실행시켜 matplotlib 패키지의 일부 함수들을 불러올 필요가 있다):

```
import matplotlib.pyplot as plt
 
plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()
```

These points are the data that we will consider the training dataset for our model.
이 점들은 모델을 위한 훈련 데이터 세트로 사용할 데이터이다. 

* Cost function and gradient descent algorithm
* 비용함수와 경사하강 알고리즘

The next step is to train our learning algorithm to be able to obtain output values y, estimated from the input data x_data. In this case, as we know in advance that it is a linear regression, we can represent our model with only two parameters: W and b.
다음 단계는 입력 데이터 x_data로부터 추정된 출력값 y를 얻기 위해 학습 알고리즘을 훈련하는 단계이다. 이 경우 알고리즘이 선형회귀인 것을 미리 알고 있기 때문에, 모델을 2개의 매개변수 W와 b만으로 표현할 수 있다. 

The objective is to generate a TensorFlow code that allows to find the best parameters W and b, that from input data x_data, adjunct them to y_data output data, in our case it will be a straight line defined by y_data = W * x_data + b . The reader knows that W should be close to 0.1 and b to 0.3, but TensorFlow does not know and it must realize it for itself.
목표는 입력 데이터 x_data로부터 출력 데이터 y_data로 부가되는 가장 좋은 매개변수 W와 b를 찾도록 하는 텐서플로우 코드를 생성하는 것인데, 이 경우 y_data = W * x_data + b로 정의되는 직선이 될 것이다. 독자는 W는 0.1에 가깝고 b는 0.3에 가까워야함을 알고 있지만, 텐서플로우는 이것을 모르므로 스스로 매개변수를 찾아내야한다. 

A standard way to solve such problems is to iterate through each value of the data set and modify the parameters W and b in order to get a more precise answer every time. To find out if we are improving in these iterations, we will define a cost function (also called “error function”) that measures how “good” (actually, as “bad”) a certain line is.
이런 문제를 푸는 표준적인 방법은 매번 더 정확한 답을 얻기 위해 데이터 세트에 있는 각각의 값을 전부 반복하여 매개변수 W와 b를 수정하는 것이다. 이러한 반복을 통해 향상되고 있는지를 알아내기 위해, 어떤 선이 얼마나 “좋은지"(사실은 얼마나 “나쁜지"로 측정된다)를 측정하는 비용 함수(cost function) (또는 “오차 함수(error function)"라고도 불린다)를 정의할 것이다. 

This function receives the pair of W and as parameters b and returns an error value based on how well the line fits the data. In our example we can use as a cost function the mean squared error[14]. With the mean squared error we get the average of the “errors” based on the distance between the real values and the estimated one on each iteration of the algorithm.
이 함수는 W와 매개변수로서의 b의 쌍을 받아서 그 직선이 데이터에 얼마나 잘 적합되는지에 근거해 오차값을 반환한다. 이 사례에서는 비용 함수로 평균 제곱 오차(mean squared error)를 사용할 수 있다. 평균 제곱 오차를 가지고 알고리즘의 각 반복에서 실제값과 추정값 사이의 거리에 근거해 “오차"의 평균을 얻는다.  

Later, I will go into more detail with the cost function and its alternatives, but for this introductory example the mean squared error helps us to move forward step by step.
나중에 비용 함수와 대안적 방법들을 더 자세히 설명할 것이지만, 이 입문용 사례를 위해서는 평균 제곱 오차  방법이 차근차근 나아가는데 도움이 된다. 

Now it is time to program everything that I have explained with TensorFlow. To do this, first we will create three variables with the following sentences:
이제 지금까지 설명한 모든 것을 텐서플로우를 이용해 프로그램해볼 시간이다. 이 작업을 하기 위해, 먼저 다음 문장에 있는 3개의 변수를 만들 것이다:

```
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
```

For now, we can move forward knowing only that the call to the method Variable is defining a variable that resides in the internal graph data structure of TensorFlow, of which I have spoken above. We will return with more information about the method parameters later, but for now I think that it’s better to move forward to facilitate this first approach.
지금은 메소드 변수에 대한 호출이 위에서 언급한 텐서플로우의 내부적인 그래프 자료 구조에 속한 변수를 정의하고 있음을 아는 것만으로도 진전이 가능하다. 나중에 메서드 매개변수에 관한 더 많은 정보를 다시 다룰 것이지만, 지금은 최초의 진입이 용이하도록 일단 앞으로 나아가는 것이 더 낫다고 생각한다. 

Now, with these variables defined, we can express the cost function that we discussed earlier, based on the distance between each point and the calculated point with the function y= W * x + b. After that, we can calculate its square, and average the sum. In TensorFlow this cost function is expressed as follows:
이제 이렇게 정의된 변수들을 가지고 각 점들과 y= W * x + b 함수로 계산된 점 사이의 거리에 근거하여 앞에서 논의한 비용함수를 표현할 수 있다. 그 다음에 거리의 제곱을 계산하여 그 총합의 평균을 낼 수 있다. 텐서플로우에서 이러한 비용 함수는 다음과 같이 표현된다:

```
loss = tf.reduce_mean(tf.square(y - y_data))
```
 
As we see, this expression calculates the average of the squared distances between the y_data point that we know, and the point y calculated from the input x_data.
여기서 보이듯 이러한 표현식은 우리가 알고 있는 점 y_data 와 x_data 입력에서 계산된 점 y 사이의 거리 제곱(squared distances)의 평균을 계산한다. 

At this point, the reader might already suspects that the line that best fits our data is the one that obtains the lesser error value. Therefore, if we minimize the error function, we will find the best model for our data.
이 지점에서 독자는 이미 우리의 데이터에 가장 잘 적합되는 선은 가장 적은 오차값을 얻는 선이라고 생각하고 있을지 모른다. 그러므로 만약 오차 함수를 최소화한다면 우리의 데이터에 가장 좋은 모델을 찾게 될 것이다. 

Without going into too much detail at the moment, this is what the optimization algorithm that minimizes functions known as gradient descent[15] achieves. At a theoretical level gradient descent is an algorithm that given a function defined by a set of parameters, it starts with an initial set of parameter values and iteratively moves toward a set of values that minimize the function. This iterative minimization is achieved taking steps in the negative direction of the function gradient[16]. It’s conventional to square the distance to ensure that it is positive and to make the error function differentiable in order to compute the gradient.
지금은 지나치게 깊이 세부사항을 다루지는 않을 텐데, 이 생각은 비용 함수를 최소화하는 최적화 알고리즘인 경사 하강법 알고리즘으로 얻으려는 결과이다. 이론적 수준에서 경사하강법은 일군의 매개변수들로 정의된 함수가 주어졌을 때, 매개변수값들의 초기 집합으로 시작하여 비용 함수를 최소화하는 값들의 집합을 향해 반복적으로 나아가는 알고리즘이다. 이러한 반복적 최소화는 함수의 경사도를 음의 방향으로 진행하면서 이뤄진다. 거리 값이 반드시 양수가 되도록 보장하기 위해 거리 값을 제곱하고 경사도를 계산하기 위해 오차 함수가 미분가능하도록 만드는 것이 관례이다. 

The algorithm begins with the initial values of a set of parameters (in our case W and b), and then the algorithm is iteratively adjusting the value of those variables in a way that, in the end of the process, the values of the variables minimize the cost function.
이 알고리즘은 일군의 매개변수(이 경우 W와 b)의 초기 값으로 시작한 다음, 마지막 과정에서 변수값들이 비용 함수를 최소화하는 방식으로 변수들의 값을 반복적으로 조정한다. 

To use this algorithm in TensorFlow, we just have to execute the following two statements:
텐서플로우에서 이 알고리즘을 사용하려면, 다음의 두 문구를 실행하기만 하면 된다. 

```
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
```

Right now, this is enough to have the idea that TensorFlow has created the relevant data in its internal data structure, and it has also implemented in this structure an optimizer that may be invoked by train, which it is a gradient descent algorithm to the cost function defined. Later on, we will discuss the function parameter called learning rate (in our example with value 0.5).
지금은 텐서플로우가 내부 자료구조에 있는 관련 데이터를 생성했다는 것과 이러한 자료 구조 내에서 훈련에 의해 작동될 수 있는 최적화기(optimizer)를 실행했다는 것을 아는 것으로 충분하며, 이 최적화기는 정의된 비용 함수에 대한 경사 하강 알고리즘이다. 나중에 학습율(learning rate)(우리의 사례에서는 0.5 값을 가진다) 이라고 불리는 함수의 매개변수에 대해 논의할 것이다.

* Running the algorithm
* 알고리즘 실행하기

As we have seen before, at this point in the code the calls specified to the library TensorFlow have only added information to its internal graph, and the runtime of TensorFlow has not yet run any of the algorithms. Therefore, like the example of the previous chapter, we must create a session, call the run method and passing train as parameter. Also, because in the code we have specified variables, we must initialize them previously with the following calls:
앞에서 본 것처럼, 현 시점에서 코드 안에서는 텐서플로우 라이브러리에 명시된 호출은 내부의 그래프에 정보를 추가하기만 했고, 텐서플로우의 런타임은 아직 어떤 알고리즘도 실행하지 않았다. 그러므로 앞 장의 사례와 마찬가지로, 세션을 생성하고 실행 메서드를 호출하고 매개변수로서 훈련을 통과해야한다. 또한 변수를 명시한 코드 때문에, 다음의 호출을 통해 변수들을 미리 초기화해야만 한다. 

```
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
```

Now we can start the iterative process that will allow us to find the values of W and b, defining the model line that best fits the points of entry. The training process continues until the model achieves a desired level of accuracy on the training data. In our particular example, if we assume that with only 8 iterations is sufficient, the code could be:
이제 입력된 점들에 가장 잘 적합시킨 모델의 직선을 정의하는 W와 b의 값을 찾도록 해줄 반복 과정를 시작할 수 있다. 훈련 과정은 모델이 훈련 데이터에 대해서 바람직한 수준의 정확성을 달성할 때까지 계속된다. 우리의 특별한 사례에서, 8번의 반복만으로 충분하다고 가정한다면 코드는 다음과 같다: 

```
for step in xrange(8):
   sess.run(train)
   print step, sess.run(W), sess.run(b)
```

The result of running this code show that the values of W and b are close to the value that we know beforehand. In my case, the result of the print is:
이 코드의 실행 결과는 W와 b의 값이 사전에 알고 있는 값에 근접함을 보여준다. 나의 경우, 출력 결과는 다음과 같다: 

```
(array([ 0.09150752], dtype=float32), array([ 0.30007562], dtype=float32))
```

And, if we graphically display the result with the following code:
그리고 다음의 코드를 가지고 시각적으로 결과를 표시한다면:

```
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.show()
```

We can see graphically the line defined by parameters W = 0.0854 and b = 0.299 achieved with only 8 iterations:
8번의 반복만으로 얻은 매개변수 W = 0.0854 와 b = 0.299 로 정의된 직선을 시각적으로 볼 수 있다.


image016

Note that we have only executed eight iterations to simplify the explanation, but if we run more, the value of parameters get closer to the expected values. We can use the following sentence to print the values of W and b:
설명을 단순화하기 위해 8번의 반복만을 실행했음에 주목하라. 그러나 더 많이 실행할수록 매개변수들의 값은 예측치에 더 가까와지게 된다. W와 b의 값을 출력하기 위해 다음의 구문을 사용할 수 있다:

```
print(step, sess.run(W), sess.run(b))
```

In our case the print outputs are:
이 경우 출력결과는 다음과 같다:

```
(0, array([-0.04841119], dtype=float32), array([ 0.29720169], dtype=float32))
(1, array([-0.00449257], dtype=float32), array([ 0.29804006], dtype=float32))
(2, array([ 0.02618564], dtype=float32), array([ 0.29869056], dtype=float32))
(3, array([ 0.04761609], dtype=float32), array([ 0.29914495], dtype=float32))
(4, array([ 0.06258646], dtype=float32), array([ 0.29946238], dtype=float32))
(5, array([ 0.07304412], dtype=float32), array([ 0.29968411], dtype=float32))
(6, array([ 0.08034936], dtype=float32), array([ 0.29983902], dtype=float32))
(7, array([ 0.08545248], dtype=float32), array([ 0.29994723], dtype=float32))
```
You can observe that the algorithm begins with the initial values of W= -0.0484 and b=0.2972 (in our case) and then the algorithm is iteratively adjusting in a way that the values of the variables minimize the cost function.
알고리즘이 초기값 W= -0.0484 와 b=0.2972(우리의 경우)로 시작한 다음, 변수값들이 비용 함수를 최소화하는 방식으로 알고리즘이 반복적으로 조정되고 있음을 관찰할 수 있다. 

You can also check that the cost function is decreasing with
비용 함수가 감소하고 있는지도 다음으로 점검할 수 있다.

```
print(step, sess.run(loss))
```

In this case the print output is:
이 경우 출력 결과는 이렇다:

```
(0, 0.015878126)
(1, 0.0079048825)
(2, 0.0041520335)
(3, 0.0023856456)
(4, 0.0015542418)
(5, 0.001162916)
(6, 0.00097872759)
(7, 0.00089203351)
```

I suggest that reader visualizes the plot at each iteration, allowing us to visually observe how the algorithm is adjusting the parameter values. In our case the 8 snapshots are:
나는 독자들에게 알고리즘이 매개변수 값들을 어떻게 조정하고 있는지 시각적으로 관찰할 수 있도록, 각 반복 단계에서 도표를 시각화할 것을 추천한다. 

image018
                                  
As the reader can see, at each iteration of the algorithm the line fits better to the data. How does the gradient descent algorithm get closer to the values of the parameters that minimize the cost function?
독자가 볼 수 있듯이, 알고리즘의 각 반복 단계마다 직선이 데이터에 더 잘 적합해진다. 경사하강 알고리즘은 어떻게 비용 함수를 최소화하는 매개변수 값들에 더 가까와지는가? 

Since our error function consists of two parameters (W and b) we can visualize it as a two-dimensional surface. Each point in this two-dimensional space represents a line. The height of the function at each point is the error value for that line. In this surface some lines yield smaller error values than others. When TensorFlow runs gradient descent search, it will start from some location on this surface (in our example the point W= -0.04841119 and b=0.29720169) and move downhill to find the line with the lowest error.
오차 함수가 2개의 매개변수(W와 b)로 구성되어 있기 때문에, 오차 함수를 2차원 표면으로 시각화할 수 있다. 이 2차원 공간에 있는 각 점은 하나의 선을 표현한다. 각 점에서 함수의 높이는 그 직선에 대한 오차값이다. 이 표면에서 어떤 선들은 다른 선들보다 더 작은 오차값을 산출한다. 텐서플로우가 경사 하강 탐색을 실행할 때, 이 표면 위의 어떤 위치로부터 탐색이 시작될 것이고(이 사례에서는 W= -0.04841119 와 b=0.29720169의 점에서) 가장 낮은 오차를 가진 선을 찾기 위해 내리막으로 움직일 것이다. 

To run gradient descent on this error function, TensorFlow computes its gradient. The gradient will act like a compass and always point us downhill. To compute it, TensorFlow will differentiate the error function, that in our case means that it will need to compute a partial derivative for W and b that indicates the direction to move in for each iteration.
이 오차 함수에서 경사 하강을 실행하기 위해, 텐서플로우는 함수의 경사도를 계산한다. 경사도는 나침반처럼 동작하여 항상 내리막을 가리킬 것이다. 경사도를 계산하기 위해 텐서플로우는 오차 함수를 미분할 것인데, 이 경우 오차 함수는 각 반복 단계 동안 움직여갈 방향을 지시하는 W와 b에 대해 편도함수를 계산할 필요가 있음을 의미한다. 

The learning rate parameter mentioned before, controls how large of a step TensorFlow will take downhill during each iteration. If we introduce a parameter too large of a step, we may step over the minimum. However, if we indicate to TensorFlow to take small steps, it will require much iteration to arrive at the minimum. So using a good learning rate is crucial. There are different techniques to adapt the value of the learning rate parameter, however it is beyond the scope of this introductory book. A good way to ensure that gradient descent algorithm is working fine is to make sure that the error decreases at each iteration.
학습율 매개변수는 텐서플로우가 얼마나 큰 단계로 각 반복 단계 동안 내리막으로 움직일지를 통제한다고 이전에 언급했다. 만약 매개변수를 너무 큰 단계로 도입한다면, 최소값을 넘어버릴 수도 있다. 그러나 만약 텐서플로우가 작은 단계를 취하도록 지시한다면, 최소값에 도달하는데 많은 반복이 필요할 것이다. 그러므로 좋은 학습률을 사용하는 것은 대단히 중요하다. 학습률 매개변수값을 조정하는 다른 기법들이 있지만, 그것은 이 입문서의 범위를 넘어서는 것이다. 경사하강 알고리즘이 잘 작동하고 있는지 보장하는 좋은 방법은 각 반복 단계마다 오차가 감소하는지를 확인하는 것이다. 

Remember that in order to facilitate the reader to test the code described in this chapter, you can download it from Github[17] of the book with the name of regression.py. Here you will find all together for easy tracking:
독자가 이 장에서 기술된 코드를 시험해보기 쉽도록, 이 책의 Github로부터 regression.py라는 이름으로 코드를 다운로드할 수 있음을 기억하라. 이곳에서 손쉽게 추적가능한 모든 코드를 함께 찾을 수 있을 것이다:

```
import numpy as np

num_points = 1000
vectors_set = []
for i in xrange(num_points):
  x1= np.random.normal(0.0, 0.55)
  y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
  vectors_set.append([x1, y1])

  x_data = [v[0] for v in vectors_set]
  y_data = [v[1] for v in vectors_set]

import matplotlib.pyplot as plt

#Graphic display
plt.plot(x_data, y_data, 'ro')
plt.legend()
plt.show()

import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(8):
  sess.run(train)
  print(step, sess.run(W), sess.run(b))
  print(step, sess.run(loss))

#Graphic display
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.xlabel('x')
plt.xlim(-2,2)
plt.ylim(0.1,0.6)
plt.ylabel('y')
plt.legend()
plt.show()

In this chapter we have begun to explore the possibilities of the TensorFlow package with a first intuitive approach to two fundamental pieces: the cost function and gradient descent algorithm, using a basic linear regression algorithm for their introduction. In the next chapter we will go into more detail about the data structures used by TensorFlow package.
이 장에서 2가지 핵심적인 부분들에 대해 직관적인 첫 접근법으로 텐서플로우 패키지가 가진 가능성을 탐험하기 시작했다: 그것은 비용 함수와 경사 하강 알고리즘으로, 기초적인 선형 회귀 알고리즘을 이용하여 소개하였다. 다음 장에서는 텐서플로우 패키지에 사용된 자료 구조에 대해서 더 상세하게 설명할 것이다. 

[contents index]
