## **Optical Flow Estimation Based on Deep Learning Approaches**
# **Predicting the Speed of a Car from a Dashboad Camera Video**
**Abstract**
ABSTRACT

The recent advancement in artificial intelligence (AI) and the rapid increase in the computation ability of modern computers has boosted capacities in solving problems. One of the goals of computer science is to enable computers to see like humans. This is somewhat achieved through image and video processing and analysis under the computer vision branch of computer science [1]. Just as the human visual system has a stimulus for the perception of the shape, distance and movement of objects in the real world and control of locomotion, it is also vital for computers to be able to do the same [5]. Videos come with a great amount of information, analyzing them enable us to handle problems such as motion detection and tracking as well as speed estimation. Speed estimation of moving objects from video cameras is one of the most important topics in the field of computer vision. It is one of the key pieces to look at in transport systems, robotics, military and naval operations, sports and among others. In this study, we explore modern optical flow methods which apply deep learning approaches and prioritizes motion as a key characteristic of classification [2]–[4] and use convolutional neural networks (CNN) to predict, with better accuracy, the speed of a car from a car dashboard camera footage.
 
**How to Setup And Test (Use) The Program**
 
  The entire process of running the program includes:
1.  Make sure the videos and the labels are in the “data/” folder.
2.  Make sure your PC/server meets the system setup requirements below:
- CUDA enable GPU.
- At least 8 Gigabytes of RAM size will not freeze the system.
4.  Make sure python3 and the required libraries which includes the following are installed:
    Pytorch (with cuda).
- openCV
- Numpy
- Matplotlib.pyplot
- keras
- scikit-learn
- flowiz
- PIL.Image
5. Run “python main.py” to convert videos (train.mp4 and test.mp4) to images and to optical flow.
6. Run “python train_model.py” to train on training video data. May take at least 4 hours and at most 24 hours to train.
7. Run “python test_model.py” to predict speed for the test.mp4 video. Note that you can ignore the steps 5 and 6, and run only this step for prediction using our already trained model. Thus, if you like to test our model only.
8. After step f, the output is stored as in the root directory as a video named, “test_output.mp4”

**References**
[1]	J. Brownlee, ‘Understand the Impact of Learning Rate on Neural Network Performance’, Machine Learning Mastery, Jan. 24, 2019. https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/ (accessed May 19, 2020).
[2]	F. D, ‘Batch normalization in Neural Networks’, Medium, Oct. 25, 2017. https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c (accessed May 19, 2020).
[3]	J. Brownlee, ‘Dropout Regularization in Deep Learning Models With Keras’, Machine Learning Mastery, Jun. 19, 2016. https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/ (accessed May 19, 2020).
[4]	A. Burton and J. Radford, Thinking in perspective: critical essays in the study of thought processes. Methuen, 1978.
[5]	B. K. P. Horn and B. G. Schunck, ‘Determining Optical Flow’, p. 19.

