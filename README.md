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
# Instance Segmentation using YOLO V8
## Link For Notebbok: https://colab.research.google.com/drive/1EIt6GyWeNy2qPT0170c-jFxPFiHcovrZ?usp=sharing
## YOLO (You Only Look Once) is a popular object detection algorithm that can also be used for instance segmentation. Instance segmentation involves not only detecting objects in an image, but also segmenting them into individual instances.

To perform instance segmentation using YOLO, you can follow these general steps:

Train a YOLO model on a dataset that includes annotations for both object detection and segmentation. The annotations should include bounding boxes for the objects as well as segmentation masks for each instance.

Modify the YOLO architecture to output both the bounding boxes and segmentation masks for each detected object. This can be done by adding additional output channels to the network.

During inference, run the YOLO model on an input image and use the output segmentation masks to segment each object into its individual instance. This can be done by applying a segmentation algorithm, such as watershed or mean shift, to the segmentation masks.

Finally, post-process the segmented instances to refine the results and remove any false positives or duplicates.

### albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       1/10      4.63G      1.504      1.604      2.584      1.558          7        640: 100% 233/233 [01:49<00:00,  2.13it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:04<00:00,  1.46it/s]
                   all        200        249      0.674      0.631      0.619      0.298      0.574       0.53       0.43      0.141

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       2/10      5.91G      1.274      1.106      1.333      1.374          7        640: 100% 233/233 [01:40<00:00,  2.33it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:04<00:00,  1.66it/s]
                   all        200        249      0.724      0.642       0.67      0.362      0.599      0.574      0.502      0.147

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       3/10      5.91G      1.294      1.095      1.296      1.398          6        640: 100% 233/233 [01:40<00:00,  2.33it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:04<00:00,  1.68it/s]
                   all        200        249      0.697      0.655      0.658      0.344      0.535      0.563      0.432       0.12

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       4/10      5.91G      1.339      1.129       1.33      1.443          6        640: 100% 233/233 [01:38<00:00,  2.35it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:04<00:00,  1.49it/s]
                   all        200        249      0.778       0.69      0.704      0.386      0.694      0.578      0.488       0.16

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       5/10      5.92G      1.255      1.108       1.25      1.409          5        640: 100% 233/233 [01:38<00:00,  2.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:04<00:00,  1.66it/s]
                   all        200        249      0.728      0.686      0.677      0.402      0.635      0.607      0.525      0.168

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       6/10      5.92G      1.169      1.083      1.167      1.354         11        640: 100% 233/233 [01:38<00:00,  2.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:05<00:00,  1.37it/s]
                   all        200        249      0.763      0.675      0.705      0.446      0.677      0.598      0.563      0.179

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       7/10      5.92G      1.098      1.059       1.08      1.304          7        640: 100% 233/233 [01:38<00:00,  2.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:04<00:00,  1.62it/s]
                   all        200        249       0.77      0.672      0.723       0.47      0.594      0.606      0.532      0.172

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       8/10      5.92G      1.011      1.056          1      1.252          6        640: 100% 233/233 [01:37<00:00,  2.39it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:04<00:00,  1.71it/s]
                   all        200        249      0.808      0.739      0.737      0.485      0.704      0.631      0.576      0.206

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       9/10      5.92G     0.9496      1.029     0.9284      1.216          6        640: 100% 233/233 [01:36<00:00,  2.42it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:04<00:00,  1.54it/s]
                   all        200        249      0.848      0.731      0.805      0.569        0.7      0.666      0.649      0.214

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      10/10      5.92G     0.8747      1.008     0.8592       1.17          8        640: 100% 233/233 [01:36<00:00,  2.41it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:09<00:00,  1.38s/it]
                   all        200        249      0.815      0.795      0.807      0.581      0.723      0.703      0.652      0.228

10 epochs completed in 0.295 hours.
Optimizer stripped from runs/segment/train14/weights/last.pt, 23.8MB
Optimizer stripped from runs/segment/train14/weights/best.pt, 23.8MB

Validating runs/segment/train14/weights/best.pt...
Ultralytics YOLOv8.0.28 🚀 Python-3.8.10 torch-1.13.1+cu116 CUDA:0 (Tesla T4, 15110MiB)
YOLOv8s-seg summary (fused): 195 layers, 11779987 parameters, 0 gradients, 42.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 7/7 [00:09<00:00,  1.34s/it]
                   all        200        249      0.815      0.795      0.806      0.581      0.791      0.653      0.652      0.228
