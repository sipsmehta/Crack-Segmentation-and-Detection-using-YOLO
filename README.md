# Crack-Detection-using-YOLO-V7
## Dataset Link: https://drive.google.com/drive/folders/1mY9MXXkM1eetYthLkGzexFyM0kspLaoc?usp=sharing
### The dataset consists of 1399 images
For Training 1330 images are used
For Testing 33 images are used
For Validation 33 images are used
## Description of YOLO V7: 
### The YOLO model is based on a deep convolutional neural network (CNN) that takes an input image and outputs a set of bounding boxes, each representing a detected object along with a confidence score that reflects how confident the model is in the accuracy of the detection. The YOLO algorithm processes the entire image at once, unlike many other object detection models that use a sliding window approach. This approach makes YOLO very fast and efficient for real-time object detection applications.
### YOLO V7 consists of a backbone CNN, followed by a neck and head. The backbone CNN is a series of convolutional layers that extract features from the input image. The neck is a set of additional convolutional layers that help the model to integrate contextual information from different scales. Finally, the head is a set of convolutional layers that predict the bounding boxes and class probabilities.
## Link for Colab Notebook:
https://colab.research.google.com/drive/1IMdgwh9X3emRhwiMPITAa342FEjKeqMX?usp=sharing
## To see the already runned notebook visit to Defect Detection using YOLO Notebook.
### Hyperparameters used: 
hyperparameters: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.3, cls_pw=1.0, obj=0.7, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.2, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.0, paste_in=0.15, loss_ota=1
Model Summary: 415 layers, 37234314 parameters used
## Output
###              Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100% 5/5 [00:02<00:00,  2.45it/s]
                 all          33          90       0.724       0.333       0.321       0.144
               Crack          33          17       0.849       0.412        0.46       0.171
       Efflorescence          33          38       0.371       0.395       0.281        0.11
               Rebar          33          21       0.523       0.476       0.465       0.175
                Rust          33           5           1           0           0           0
             Scaling          33           2           1           0           0           0
               Spall          33           7       0.599       0.714       0.718       0.407
5 epochs completed in 1.392 hours.


### ![image](https://user-images.githubusercontent.com/69897673/219551011-76f1e272-1997-4ad0-9c84-e06adf29a38b.png)
### ![image](https://user-images.githubusercontent.com/69897673/219551043-e54571f5-6a51-44ca-a1af-c3c756159bdc.png)

