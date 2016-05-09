1. TENSORFLOW BASICS

In this chapter I will present very briefly how a TensorFlow’s code and their programming model is. At the end of this chapter, it is expected that the reader can install the TensorFlow package on their personal computer.

An Open Source Package

Machine Learning has been investigated by the academy for decades, but it is only in recent years that its penetration has also increased in corporations. This happened thanks to the large volume of data it already had and the unprecedented computing capacity available nowadays.

In this scenario, there is no doubt that Google, under the holding of Alphabet, is one of the largest corporations where Machine Learning technology plays a key role in all of its virtual initiatives and products.

Last October, when Alphabet announced its quarterly Google’s results, with considerable increases in sales and profits, CEO Sundar Pichai said clearly: “Machine learning is a core, transformative way by which we’re rethinking everything we’re doing”.

Technologically speaking, we are facing a change of era in which Google is not the only big player. Other technology companies such as Microsoft, Facebook, Amazon and Apple, among many other corporations are also increasing their investment in these areas.

In this context, a few months ago Google released its TensorFlow engine under an open source license (Apache 2.0). TensorFlow can be used by developers and researchers who want to incorporate Machine Learning in their projects and products, in the same way that Google is doing internally with different commercial products like Gmail, Google Photos, Search, voice recognition, etc.

TensorFlow was originally developed by the Google Brain Team, with the purpose of conducting Machine Learning and deep neural networks research, but the system is general enough to be applied in a wide variety of other Machine Learning problems.

Since I am an engineer and I am speaking to engineers, the book will look under the hood to see how the algorithms are represented by a data flow graph. TensorFlow can be seen as a library for numerical computation using data flow graphs. The nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors), which interconnect the nodes.

TensorFlow is constructed around the basic idea of building and manipulating a computational graph, representing symbolically the numerical operations to be performed. This allows TensorFlow to take advantage of both CPUs and GPUs right now from Linux 64-bit platforms such as Mac OS X, as well as mobile platforms such as Android or iOS.

Another strength of this new package is its visual TensorBoard module that allows a lot of information about how the algorithm is running to be monitored and displayed. Being able to measure and display the behavior of algorithms is extremely important in the process of creating better models. I have a feeling that currently many models are refined through a little blind process, through trial and error, with the obvious waste of resources and, above all, time.

TensorFlow Serving

Recently Google launched TensorFlow Serving[3], that helps developers to take their TensorFlow machine learning models (and, even so, can be extended to serve other types of models) into production. TensorFlow Serving is an open source serving system (written in C++) now available on GitHub under the Apache 2.0 license.

What is the difference between TensorFlow and TensorFlow Serving?  While in TensorFlow it is easier for the developers to build machine learning algorithms and train them for certain types of data inputs, TensorFlow Serving specializes in making these models usable in production environments. The idea is that developers train their models using TensorFlow and then they use TensorFlow Serving’s APIs to react to input from a client.

This allows developers to experiment with different models on a large scale that change over time, based on real-world data, and maintain a stable architecture and API in place.

The typical pipeline is that a training data is fed to the learner, which outputs a model, which after being validated is ready to be deployed to the TensorFlow serving system. It is quite common to launch and iterate on our model over time, as new data becomes available, or as you improve the model. In fact, in the google post [4] they mention that at Google, many pipelines are running continuously, producing new model versions as new data becomes available.

TensorFlowServingDevelopers use to communicate with TensorFlow Serving a front-end implementation based on gRPC, a high performance, open source RPC framework from Google.

If you are interested in learning more about TensorFlow Serving, I suggest you start by by reading the Serving architecture overview [5] section, set up your environment and start to do a basic tutorial[6] .

TensorFlow Installation

It is time to get your hands dirty. From now on, I recommend that you interleave the reading with the practice on your computer.

TensorFlow has a Python API (plus a C / C ++) that requires the installation of Python 2.7 (I assume that any engineer who reads this book knows how to do it).

In general, when you are working in Python, you should use the virtual environment virtualenv. Virtualenv is a tool to keep Python dependencies required in different projects, in different parts of the same computer. If we use virtualenv to install TensorFlow, this will not overwrite existing versions of Python packages from other projects required by TensorFlow.

First, you should install pip and virtualenv if they are not already installed, like the follow script shows:

# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev python-virtualenv 

# Mac OS X 
$ sudo easy_install pip
$ sudo pip install --upgrade virtualenv
environment virtualenv in the ~/tensorflow directory:

$ virtualenv --system-site-packages ~/tensorflow
The next step is to activate the virtualenv. This can be done as follows:

$ source ~/tensorflow/bin/activate #  with bash 
$ source ~/tensorflow/bin/activate.csh #  with csh
(tensorflow)$
The name of the virtual environment in which we are working will appear at the beginning of each command line from now on. Once the virtualenv is activated, you can use pip to install TensorFlow inside it:

# Ubuntu/Linux 64-bit, CPU only:
(tensorflow)$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl 

# Mac OS X, CPU only:
(tensorflow)$ sudo easy_install --upgrade six
(tensorflow)$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
I recommend that you visit the official documentation indicated here, to be sure that you are installing the latest available version.

If the platform where you are running your code has a GPU, the package to use will be different. I recommend that you visit the official documentation to see if your GPU meets the specifications required to support Tensorflow. Installing additional software is required to run Tensorflow GPU and all the information can be found at Download and Setup TensorFlow[7] web page. For more information on the use of GPUs, I suggest reading chapter 6.

Finally, when you’ve finished, you should disable the virtual environment as follows:

(tensorflow)$ deactivate
Given the introductory nature of this book, we suggest thatthe reader visits the mentioned official documentation page to find more information about other ways to install Tensorflow.

My first code in TensorFlow

As I mentioned at the beginning, we will move in this exploration of the planet TensorFlow with little theory and lots of practice. Let’s start!

From now on, it is best to use any text editor to write python code and save it with extension “.py” (eg test.py). To run the code, it will be enough with the command python test.py.

To get a first impression of what a TensorFlow’s program is, I suggest doing a simple multiplication program; the code looks like this:

import tensorflow as tf
  
   a = tf.placeholder("float")
    b = tf.placeholder("float")
      
       y = tf.mul(a, b)
         
          sess = tf.Session()
            
             print sess.run(y, feed_dict={a: 3, b: 3})
              

              In this code, after importing the Python module tensorflow, we define “symbolic” variables, called placeholder in order to manipulate them during the program execution. Then, we move these variables as a parameter in the call to the function multiply that TensorFlow offers. tf.mul is one of the many mathematical operations that TensorFlow offers to manipulate the tensors. In this moment, tensors can be considered dynamically-sized, multidimensional data arrays.

              The main ones are shown in the following table:

              Operation Description
              tf.add    sum
              tf.sub    substraction
              tf.mul    multiplication
              tf.div    division
              tf.mod    module
              tf.abs    return the absolute value
              tf.neg    return negative value
              tf.sign   return the sign
              tf.inv    returns the inverse
              tf.square calculates the square
              tf.round  returns the nearest integer
              tf.sqrt   calculates the square root
              tf.pow    calculates the power
              tf.exp    calculates the exponential
              tf.log    calculates the logarithm
              tf.maximum    returns the maximum
              tf.minimum    returns the minimum
              tf.cos    calculates the cosine
              tf.sin    calculates the sine
               

               TensorFlow also offers the programmer a number of functions to perform mathematical operations on matrices. Some are listed below:

               Operation    Description
               tf.diag  returns a diagonal tensor with a given diagonal values
               tf.transpose returns the transposes of the argument
               tf.matmul    returns a tensor product of multiplying two tensors listed as arguments
               tf.matrix_determinant    returns the determinant of the square matrix specified as an argument
               tf.matrix_inverse    returns the inverse of the square matrix specified as an argument
               The next step, one of the most important, is to create a session to evaluate the specified symbolic expression. Indeed, until now nothing has yet been executed in this TensorFlowcode. Let me emphasize that TensorFlow is both, an interface to express Machine Learning’s algorithms and an implementation to run them, and this is a good example.

               Programs interact with Tensorflow libraries by creating a session with Session(); it is only from the creation of this session when we can call the run() method, and that is when it really starts to run the specified code. In this particular example, the values of the variables are introduced into the run() method with a feed_dict argument. That’s when the associated code solves the expression and exits from the display a 9 as a result of multiplication.

               With this simple example, I tried to introduce the idea that the normal way to program in TensorFlow is to specify the whole problem first, and eventually create a session to allow the running of the associated computation.

               Sometimes however, we are interested in having more flexibility in order to structure the code, inserting operations to build the graph with operations running part of it. It happens when we are, for example, using interactive environments of Python such as IPython [8]. For this purpose, TesorFlow offers the tf.InteractiveSession() class.

               The motivation for this programming model is beyond the reach of this book. However, to continue with the next chapter, we only need to know that all information is saved internally in a graph structure that contains all the information operations and data .

               This graph describes mathematical computations. The nodes typically implement mathematical operations, but they can also represent points of data entry, output results, or read/write persistent variables. The edges describe the relationships between nodes with their inputs and outputs and at the same time carry tensors, the basic data structure of TensorFlow.

               The representation of the information as a graph allows TensorFlow to know the dependencies between transactions and assigns operations to devices asynchronously, and in parallel, when these operations already have their associated tensors (indicated in the edges input) available.

               Parallelism is therefore one of the factors that enables us to speed up the execution of some computationally expensive algorithms, but also because TensorFlow has already efficiently implemented a set of complex operations. In addition, most of these operations have associated kernels which are implementations of operations designed for specific devices such as GPUs. The following table summarizes the most important operations/kernels[9]:

               Operations groups    Operations
               Maths    Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal
               Array    Concat, Slice, Split, Constant, Rank, Shape, Shuffle
               Matrix   MatMul, MatrixInverse, MatrixDeterminant
               Neuronal Network SoftMax, Sigmoid, ReLU, Convolution2D, MaxPool
               Checkpointing    Save, Restore
               Queues and syncronizations   Enqueue, Dequeue, MutexAcquire, MutexRelease
               Flow control Merge, Switch, Enter, Leave, NextIteration
               Display panel Tensorboard

               To make it more comprehensive, TensorFlow includes functions to debug and optimize programs in a visualization tool called TensorBoard. TensorBoard can view different types of statistics about the parameters and details of any part of the graph computing graphically.

               The data displayed with TensorBoard module is generated during the execution of TensorFlow and stored in trace files whose data is obtained from the summary operations. In the documentation page[10] of TensorFlow, you can find detailed explanation of the Python API.

               The way we can invoke it is very simple: a service with Tensorflow commands from the command line, which will include as an argument the file that contains the trace.

               (tensorflow)$ tensorboard --logdir=&lt;trace file&gt;
               You simply need to access the local socket 6006 from the browser[11] with http://localhost:6006/ .

               The visualization tool called TensorBoard is beyond the reach of this book. For more details about how Tensorboard works, the reader can visit the section TensorBoard Graph Visualization[12]from the TensorFlow tutorial page.
               [contents link]
