### **Enhancement of Vehicle Speed Estimation Based on Optical Flow and Deep Learning Approaches**
**Abstract**

In recent times, the subject of Intelligent vehicles has become very popular. The inspiration of this popularity cannot be discussed without mentioning computer vision and deep learning. This is because many vehicles, today, have all kinds of high-quality cameras installed on them to capture videos for different benefits. Object tracking and speed estimation are important tasks in video processing. Several methods for speed estimation have been proposed. This paper deals with speed estimation of a car from a video. We propose a method for estimating the speed of a car, with better accuracy, from a video captured by the car’s dashboard camera. The method uses two networks, one for estimating the displacement of the car, and the other for learning the speed labels. We perform experiments applying several image-processing techniques and using a lightweight and efficient optical flow estimation based deep learning approaches to achieve this goal. The proposed model is trained on the comma.ai speed challenge dataset and the results are evaluated and compared to other submissions on this challenge.
 
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


Paper link - Pending...
