번역자 : 권갑진 kwonssi@naver.com

 CLOSING
 +마치며
  
  Exploration is the engine that drives innovation. Innovation drives economic growth. So let’s all go exploring.
 -
 +탐험은 혁신으로 이끄는 엔진입니다. 혁신은 경제 성장을 이끕니다. 자 우리 모두 함께 탐험을 떠납시다. 
  Edith Widder
 -
 +에디쓰 위더(TED 2013 "우리가 어떻게 거대 오징어를 발견했는가?" 발표자. 해양학자이며 발명가)
   
   
  Here I have presented an introductory guide explaining how to use TensorFlow, providing a warm-up with this technology that will undoubtedly have a leading role in the looming technological scenario. There are, indeed, other alternatives to TensorFlow, and each one suit the best a particular problem; I want to invite the reader to explore beyond the TensorFlow package.
 +여기에 나는 텐서플로우를 어떻게 사용할 것인지를 설명하는 입문 가이드를 제시했으며, 어렴풋한 기술적인 시나리오 안에서 의심의 여지 없이 하나의 선도적인 역할을 하게 될 이 기술에 대한 워밍업을 제공했다.
 
There is lots of diversity in these packages. Some are more specialized, others less. Some are more difficult to install than others. Some of them are very well documented while others , despite working well, are more difficult to find detailed information about how to use them.
이러한 유형의 패키지들에는 매우 많은 다양성이 있다. 일부는 더 많이 특화되어 있으며, 다른 것들은 덜 특화되어 있다. 일부는 다른 것들보다 설치하기 더 많이 어렵다. 그들 중 일부는 매우 잘 문서화 되어 있지만 다른 것들은 잘 작동함에도 불구하고 어떻게 사용하는지에 대한 자세한 정보를 찾기 어렵다.

An important thing: on the following day that TensorFlow was release by Google, I read in a tweet[49] that during the period 2010-2014 a new Deep learning package was released every 47 days, and in 2015 releases were published every 22 days. It is mpressive, isn’t it? As I advanced in the first chapter of the book, as a starting point for the reader, an extensive list can be found at Awesome Deep Learning[50].
한가지 중요한 사실: 구글에 의해 텐서플로우가 배포된 다음날 나는 트위터[49]에서 2010-2014 기간 중에 새로운 딥러닝 패키지가 매 47일마다 배포되었고, 2015년도에는 매 22일마다 배포판이 나왔다. 이것은 인상적인 일이다, 그렇지 않은가? 내가 책의 첫번째 장에서 독자의 시작 지점으로 미리 말한대로, Awesome Deep Learning[50]에서 광대한 딥러닝 패키지 목록을 찾을 수 있다.


Without any doubt, the landscape of Deep Learning was impacted in November 2015 with the release of Google’s TensorFlow, being now the most popular open source machine learning library on Github by a wide margin[51].
의심의 여지 없이 딥러닝의 지평은 2015년 11월 구글의  텐서플로우 배포로 인해 큰 영향을 받았다. 그것은 현재 기트허브 상에서 큰 차이로  가장 인기있는 오픈소스 머신러닝 라이브러리이다[51].

Remember that the second most-starred Machine Learning project of Github is Scikit-learn[52], the de facto official Python general Machine Learning framework. For these users, TensorFlow can be used through Scikit Flow (skflow)[53], a simplified interface for TensorFlow coming out from Google.
기트허브에서 두번째로 가장 빛나는 머신러닝 프로젝트는 Sckikit-learn이며 그것은 산업표준의 공식 파이썬 범용 머신 러닝 프레임워크이다[52]. 이들 사용자를 위해 텐서플로우는 구글로부터 나온 텐서플로우를 위한 간략화된 인터페이스인 Scikit Flow(skflow)를 통해 사용될 수 있다[53]. 

Practically, Scikit Flow is a high level wrapper for the TensorFlow library, which allows the training and fitting of neural networks using the familiar approach of Scikit-Learn. This library covers a variety of needs from linear models to Deep Learning applications.
실질적으로 Scikit Flow는 텐서플로우 라이브러리릉 위한 고급 수준의 wrapper이다. 그것은 Scikit-Learn의 친근한 접근법을 사용해서  신경망을 훈련시키고 적합시키는 것을 가능하게 한다. 

In my humble opinion, after the release of the distributed version of TensorFlow, TensorFlow Serving and Scikit Flow, TensorFlow will become de facto a mainstream Deep Learning library.
나 나름의 주장으로는 텐서플로우의 분산 버전, 텐서플로우 서빙과 Scikit Flow의 배포 후에는 텐서플로우가 주류 딥러닝 라이브리의 사실상의 표준이 될 것이다.

Deep learning has dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains. What will be its future? According to an excellent review from Yann LeCun, Yoshua Bengio and Geoffrey Hilton in Nature journal, the answer is the Unsupervised Learning [54]. They expect Unsupervised Learning to become far more important in longer term than supervised learning. As they mention, human and animal learning is largely unsupervised: we discover the structure of the world by observing it, not by being told the name of every object.
딥 러닝은 음성 인식, 시각적 물체 인식, 객체 감지와  많은 다른 영역들에서 최신을 극적으로 향상시켰다. Yann LeCun, Yoshua Bengio와 Geoffrey Hilton의 네이쳐 저널에서의 매우 뛰어난 리뷰에 따르면, 그 해답은 비지도 학습이다[54]. 그들은 비지도 학습이 지도 학습에 비해 더 긴 기간 동안 매우 더 중요하게 될 것을 기대하고 있다. 그들이 언급한대로 사람과 동물의 학습은 대체로 비지도 방식이다. 우리는 모든 물체으 이름을 듣는 것이 아니라 세계의 구조를 관찰함으로써 발견한다.

They have a lot of espectations of the future progress of the systems that combine CNN with recurrent neural network (RNN) that use reinforcement learning. RNNs process an input that sequence one element at time, maintaining in their hidden units information about the history of all the past elements of the sequence. For an introduction to RNN implementation in TensorFlow the reader can review the Recurrent Neural Networks [55] section in TensorFlow Tutorial.
그들은 CNN을 RNN(Recurrent Neural Network)과 결합한 시스템들의 미래의 진화에 대해 매우 많은 기대를 갖고 있다. RNN은 한번에 시간에 따라 요소들을 하나씩 배열한 입력을 처리하며, 그들의 숨은 단위에 배열의 모든 과거 요소들의 이력에 대한 정보를 유지한다. 텐서플로우를 사용한 RNN 구현에 대한 소개를 위해 독자는 텐서플로우 튜토리얼의 RNN 섹션을 리뷰할 수 있다[55]. 

Besides, there are many challenges ahead in Deep Learning; the time to training them are driving the need for a new class of supercomputer systems. A lot of research is still necessary in order to integrate the best analytics knowledge with new Big Data technologies and the awesome power of emerging computational systems in order to interpret massive amounts of heterogeneous data at an unprecedented rate.
게다가 딥러닝의 앞에 많은 도전이 있다. 그들을 훈련시키는데 소요되는 시간은 새로운 유형의 슈퍼컴퓨터 시스템의 필요성을 이끌고 있다. 최고의 분석학 지식과 빅데이터 기술, 그리고 비균질의 데이터를  이전에 없던 매우 빠른 속도로 빠르게 대용량의 데이터를 해석하기 위한, 새로 부상하는 컴퓨터 시스템들의 놀라운 힘에 대한 많은 수의 연구는 아직 통합하기 위해 많은 연구가 아직 필요하다.

Scientific progress is typically the result of an interdisciplinary, long and sustained effort by a large community rather than a breakthrough, and deep learning, and machine learning in general, is not an exception. We are entering into an extremely exciting period for interdisciplinary research, where ecosystems like the ones found in Barcelona as UPC and BSC-CNS, with deep knowledge in High Performance Computing and Big Data Technologies, will play a big role in this new scenario.


[contents link]
