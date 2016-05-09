2. LINEAR REGRESSION IN TENSORFLOW

In this chapter, I will begin exploring TensorFlow’s coding with a simple model: Linear Regression. Based on this example, I will present some code basics and, at the same time, how to call various important components in the learning process, such as the cost function or the algorithm gradient descent.

Model of relationship between variables

Linear regression is a statistical technique used to measure the relationship between variables. Its interest is that the algorithm that implements it is not conceptually complex, and can also be adapted to a wide variety of situations. For these reasons, I have found it interesting to start delving into TensorFlow with an example of linear regression.

Remember that both, in the case of two variables (simple regression) and the case of more than two variables (multiple regression), linear regression models the relationship between a dependent variable, independent variables xi and a random term b.

In this section I will create a simple example to explain how TensorFlow works assuming that our data model corresponds to a simple linear regression as y = W * x + b. For this, I use a simple Python program that creates data in a two-dimensional space, and then I will ask TensorFlow to look for the line that fits the best in these points.

The first thing to do is to import the NumPy package that we will use to generate points. The code we have created is as it follows:

import numpy as np
 
 num_points = 1000
 vectors_set = []
 for i in xrange(num_points):
          x1= np.random.normal(0.0, 0.55)
                   y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
                            vectors_set.append([x1, y1])
                             
                             x_data = [v[0] for v in vectors_set]
                             y_data = [v[1] for v in vectors_set]
                             As you can see from the code, we have generated points following the relationship y = 0.1 * x + 0.3, albeit with some variation, using a normal distribution, so the points do not fully correspond to a line, allowing us to make a more interesting example.

                             In our case, a display of the resulting cloud of points is:

                             image014

                             The reader can view them with the following code (in this case, we need to import some of the functions of matplotlib package, running pip install matplotlib[13]):

                             import matplotlib.pyplot as plt
                              
                              plt.plot(x_data, y_data, 'ro', label='Original data')
                              plt.legend()
                              plt.show()
                              These points are the data that we will consider the training dataset for our model.

                              Cost function and gradient descent algorithm

                              The next step is to train our learning algorithm to be able to obtain output values y, estimated from the input data x_data. In this case, as we know in advance that it is a linear regression, we can represent our model with only two parameters: W and b.

                              The objective is to generate a TensorFlow code that allows to find the best parameters W and b, that from input data x_data, adjunct them to y_data output data, in our case it will be a straight line defined by y_data = W * x_data + b . The reader knows that W should be close to 0.1 and b to 0.3, but TensorFlow does not know and it must realize it for itself.

                              A standard way to solve such problems is to iterate through each value of the data set and modify the parameters W and b in order to get a more precise answer every time. To find out if we are improving in these iterations, we will define a cost function (also called “error function”) that measures how “good” (actually, as “bad”) a certain line is.

                              This function receives the pair of W and as parameters b and returns an error value based on how well the line fits the data. In our example we can use as a cost function the mean squared error[14]. With the mean squared error we get the average of the “errors” based on the distance between the real values and the estimated one on each iteration of the algorithm.

                              Later, I will go into more detail with the cost function and its alternatives, but for this introductory example the mean squared error helps us to move forward step by step.

                              Now it is time to program everything that I have explained with TensorFlow. To do this, first we will create three variables with the following sentences:

                              W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
                              b = tf.Variable(tf.zeros([1]))
                              y = W * x_data + b
                              For now, we can move forward knowing only that the call to the method Variable is defining a variable that resides in the internal graph data structure of TensorFlow, of which I have spoken above. We will return with more information about the method parameters later, but for now I think that it’s better to move forward to facilitate this first approach.

                              Now, with these variables defined, we can express the cost function that we discussed earlier, based on the distance between each point and the calculated point with the function y= W * x + b. After that, we can calculate its square, and average the sum. In TensorFlow this cost function is expressed as follows:

                              loss = tf.reduce_mean(tf.square(y - y_data))
                              As we see, this expression calculates the average of the squared distances between the y_data point that we know, and the point y calculated from the input x_data.

                              At this point, the reader might already suspects that the line that best fits our data is the one that obtains the lesser error value. Therefore, if we minimize the error function, we will find the best model for our data.

                              Without going into too much detail at the moment, this is what the optimization algorithm that minimizes functions known as gradient descent[15] achieves. At a theoretical level gradient descent is an algorithm that given a function defined by a set of parameters, it starts with an initial set of parameter values and iteratively moves toward a set of values that minimize the function. This iterative minimization is achieved taking steps in the negative direction of the function gradient[16]. It’s conventional to square the distance to ensure that it is positive and to make the error function differentiable in order to compute the gradient.

                              The algorithm begins with the initial values of a set of parameters (in our case W and b), and then the algorithm is iteratively adjusting the value of those variables in a way that, in the end of the process, the values of the variables minimize the cost function.

                              To use this algorithm in TensorFlow, we just have to execute the following two statements:

                              optimizer = tf.train.GradientDescentOptimizer(0.5)
                              train = optimizer.minimize(loss)
                              Right now, this is enough to have the idea that TensorFlow has created the relevant data in its internal data structure, and it has also implemented in this structure an optimizer that may be invoked by train, which it is a gradient descent algorithm to the cost function defined. Later on, we will discuss the function parameter called learning rate (in our example with value 0.5).

                              Running the algorithm

                              As we have seen before, at this point in the code the calls specified to the library TensorFlow have only added information to its internal graph, and the runtime of TensorFlow has not yet run any of the algorithms. Therefore, like the example of the previous chapter, we must create a session, call the run method and passing train as parameter. Also, because in the code we have specified variables, we must initialize them previously with the following calls:

                              init = tf.initialize_all_variables()
                               
                               sess = tf.Session()
                               sess.run(init)
                               Now we can start the iterative process that will allow us to find the values of W and b, defining the model line that best fits the points of entry. The training process continues until the model achieves a desired level of accuracy on the training data. In our particular example, if we assume that with only 8 iterations is sufficient, the code could be:

                               for step in xrange(8):
                                  sess.run(train)
                                  print step, sess.run(W), sess.run(b)
                                  The result of running this code show that the values of W and b are close to the value that we know beforehand. In my case, the result of the print is:

                                  (array([ 0.09150752], dtype=float32), array([ 0.30007562], dtype=float32))
                                  And, if we graphically display the result with the following code:

                                  plt.plot(x_data, y_data, 'ro')
                                  plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
                                  plt.legend()
                                  plt.show()
                                  We can see graphically the line defined by parameters W = 0.0854 and b = 0.299 achieved with only 8 iterations:

                                  image016

                                  Note that we have only executed eight iterations to simplify the explanation, but if we run more, the value of parameters get closer to the expected values. We can use the following sentence to print the values of W and b:

                                  print(step, sess.run(W), sess.run(b))
                                  In our case the print outputs are:

                                  (0, array([-0.04841119], dtype=float32), array([ 0.29720169], dtype=float32))
                                  (1, array([-0.00449257], dtype=float32), array([ 0.29804006], dtype=float32))
                                  (2, array([ 0.02618564], dtype=float32), array([ 0.29869056], dtype=float32))
                                  (3, array([ 0.04761609], dtype=float32), array([ 0.29914495], dtype=float32))
                                  (4, array([ 0.06258646], dtype=float32), array([ 0.29946238], dtype=float32))
                                  (5, array([ 0.07304412], dtype=float32), array([ 0.29968411], dtype=float32))
                                  (6, array([ 0.08034936], dtype=float32), array([ 0.29983902], dtype=float32))
                                  (7, array([ 0.08545248], dtype=float32), array([ 0.29994723], dtype=float32))
                                  You can observe that the algorithm begins with the initial values of W= -0.0484 and b=0.2972 (in our case) and then the algorithm is iteratively adjusting in a way that the values of the variables minimize the cost function.

                                  You can also check that the cost function is decreasing with

                                  print(step, sess.run(loss))
                                  In this case the print output is:

                                  (0, 0.015878126)
                                  (1, 0.0079048825)
                                  (2, 0.0041520335)
                                  (3, 0.0023856456)
                                  (4, 0.0015542418)
                                  (5, 0.001162916)
                                  (6, 0.00097872759)
                                  (7, 0.00089203351)
                                  I suggest that reader visualizes the plot at each iteration, allowing us to visually observe how the algorithm is adjusting the parameter values. In our case the 8 snapshots are:

                                  image018
                                  As the reader can see, at each iteration of the algorithm the line fits better to the data. How does the gradient descent algorithm get closer to the values of the parameters that minimize the cost function?

                                  Since our error function consists of two parameters (W and b) we can visualize it as a two-dimensional surface. Each point in this two-dimensional space represents a line. The height of the function at each point is the error value for that line. In this surface some lines yield smaller error values than others. When TensorFlow runs gradient descent search, it will start from some location on this surface (in our example the point W= -0.04841119 and b=0.29720169) and move downhill to find the line with the lowest error.

                                  To run gradient descent on this error function, TensorFlow computes its gradient. The gradient will act like a compass and always point us downhill. To compute it, TensorFlow will differentiate the error function, that in our case means that it will need to compute a partial derivative for W and b that indicates the direction to move in for each iteration.

                                  The learning rate parameter mentioned before, controls how large of a step TensorFlow will take downhill during each iteration. If we introduce a parameter too large of a step, we may step over the minimum. However, if we indicate to TensorFlow to take small steps, it will require much iteration to arrive at the minimum. So using a good learning rate is crucial. There are different techniques to adapt the value of the learning rate parameter, however it is beyond the scope of this introductory book. A good way to ensure that gradient descent algorithm is working fine is to make sure that the error decreases at each iteration.

                                  Remember that in order to facilitate the reader to test the code described in this chapter, you can download it from Github[17] of the book with the name of regression.py. Here you will find all together for easy tracking:

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

                                                                                                                          [contents index]
