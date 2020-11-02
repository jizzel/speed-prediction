# **VEHICLE SPEED ESTIMATION BASED ON OPTICAL FLOW AND DEEP LEARNING**
### **Predicting the Speed of a Vehicle from a Dashboard Camera Video**
**Abstract**

Recent advancements in AI have inspired the redevelopment and refinement of existing solutions in computer vision, image, and video analysis technology for better performance. Videos contain a large amount of information and analyzing them enable us to manage challenges in motion detection and tracking, speed estimation and others. The speed estimation of moving objects from a video is one of the most important topics studied in computer vision. It is one of the key components to be addressed in transportation systems, robotics, military, and maritime operations, sports, and other fields. Building intelligent vehicles that have fail-safe features to operate on roads is one of steps towards the development of smart cities. Predicting the speed of a vehicle from its dashboard camera video serves as a significant aid for such vehicles and relevant for other use cases such as in sports. In this study, we investigate modern methods of optical flow estimation that employ deep learning approaches and use Convolutional Neural Networks (CNN) to experiment and better predict the speed of a car from its dashboard camera video with enhanced accuracy in real-time capabilities. Our results show predictions with very minimal error (MSE: 0.3942) for both day and night scenes.
 
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
![Video 1](https://github.com/jizzel/speed-prediction/blob/master/OutputSample/1.gif)
![Video 1](https://github.com/jizzel/speed-prediction/blob/master/OutputSample/2.gif)
![Video 1](https://github.com/jizzel/speed-prediction/blob/master/OutputSample/3.gif)
![Video 1](https://github.com/jizzel/speed-prediction/blob/master/OutputSample/4.gif)

**References**
- [1]	F. D, ‘Batch normalization in Neural Networks’, Medium, Oct. 25, 2017. https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c (accessed May 19, 2020).
- [2]	J. Brownlee, ‘Dropout Regularization in Deep Learning Models With Keras’, Machine Learning Mastery, Jun. 19, 2016. https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/ (accessed May 19, 2020).
- [3]	A. Burton and J. Radford, Thinking in perspective: critical essays in the study of thought processes. Methuen, 1978.

Paper link - Pending...
