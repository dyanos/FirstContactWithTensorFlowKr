iA PRACTICAL APPROACH

Tell me and I forget. Teach me and I remember. Involve me and I learn.
Benjamin Franklin

One of the common applications of Deep Learning includes pattern recognition. Therefore, in the same way as when you start programming there is sort of a tradition to start printing “Hello World”, in Deep Learning a model for the recognition of handwritten digits is usually constructed[1]. The first example of a neural network that I will provide, will also allow me to introduce this new technology called TensorFlow.

However, I do not intend to write a research book on Machine Learning or Deep Learning, I only want to make this new Machine Learning’s package, TensorFlow, available to everybody, as soon as possible. Therefore I apologise in to my fellow data scientists for certain simplifications that I have allowed myself in order to share this knowledge with the general reader.

The reader will find here the regular structure that I use in my classes; that is inviting you to use your computer’s keyboard while you learn. We call it “learn by doing“, and my experience as a professor at UPC tells me that it is an approach that works very well with engineers who are trying to start a new topic.

For this reason, the book is of a practical nature, and therefore I have reduced the theoretical part as much as possible. However certain mathematical details have been included in the text when they are necessary for the learning process.

I assume that the reader has some basic underestanding of Machine Learning, so I will use some popular algorithms to gradually organize the reader’s training in TensorFlow.

In the first chapter, in addition to an introduction to the scenario in which TensorFlow will have an important role, I take the opportunity to explain the basic structure of a TensorFlow program, and explain briefly the data it maintains internally.

In chapter two, through an example of linear regression, I will present some code basics and, at the same time, how to call various important components in the learning process, such as the cost function or the gradient descent optimization algorithm.

In chapter three, where I present a clustering algorithm, I go into detail to present the basic data structure of TensorFlow called tensor, and the different classes and functions that the TensorFlow package offers to create and manage the tensors.

In chapter four, how to build a neural network with a single layer to recognize handwritten digits is presented in detail. This will allow us to sort all the concepts presented above, as well as see the entire process of creating and testing a model.

The next chapter begins with an explanation based on neural network concepts seen in the previous chapter and introduces how to construct a multilayer neural network to get a better result in the recognition of handwritten digits. What it is known as convolutional neural network will be presented in more detail.

In chapter six we look at a more specific issue, probably not of interest to all readers, harnessing the power of calculation presented by GPUs. As introduced in chapter 1, GPUs play an important role in the training process of neural networks.

The book ends with closing remarks, in which I highlight some conclusions. I would like to emphasize that the examples of code in this book can be downloaded from the github repository of the book[2].

[contents link]
