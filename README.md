## **Optical Flow Estimation Based on Deep Learning Approaches** 
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
7. Run “python test_model.py” to predict speed for the test.mp4 video. Note that you can ignore the step “d” and step “e” and run only this step for prediction using our already trained model. Thus, if you like to test our model only.
8. 47
9. After step f, the output is stored as in the root directory as a video named, “test_output.mp4”
