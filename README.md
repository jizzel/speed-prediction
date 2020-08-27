# **VEHICLE SPEED ESTIMATION BASED ON OPTICAL FLOW AND DEEP LEARNING**
### **Predicting the Speed of a Vehicle from a Dashboard Camera Video**
**Abstract**

The recent advancement in artificial intelligence (AI) and the rapid increase in the computation ability of modern computers has boosted capacities in solving problems. One of the goals of computer science is to enable computers see like humans. This is somewhat achieved through image and video processing. Videos come with a great amount of information, analyzing them enables us to handle problems such as motion detection and tracking as well as speed estimation. Speed estimation of moving object from video camera is one of the most important subject to study in computer vision. This is one of the key components to address in transport systems, robotics, military and naval operations, sports and among others. In this study, we explore modern optical flow methods which apply deep learning approaches to prioritize motion as a key characteristic of classification [1]–[3] and use Convolutional Neural Networks (CNN) to predict, with better accuracy, the speed of a car from a car dashboard camera footage.
 
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

##### Sample of our OutPut results
![Video 1](https://github.com/jizzel/speed-prediction/blob/master/OutputSample/1)

**References**
- [1]	F. D, ‘Batch normalization in Neural Networks’, Medium, Oct. 25, 2017. https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c (accessed May 19, 2020).
- [2]	J. Brownlee, ‘Dropout Regularization in Deep Learning Models With Keras’, Machine Learning Mastery, Jun. 19, 2016. https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/ (accessed May 19, 2020).
- [3]	A. Burton and J. Radford, Thinking in perspective: critical essays in the study of thought processes. Methuen, 1978.

Paper link - 
