# Instance Segmentation using YOLO V8
## Link For Notebbok: https://colab.research.google.com/drive/1EIt6GyWeNy2qPT0170c-jFxPFiHcovrZ?usp=sharing
## Link for dataset: https://drive.google.com/file/d/1Vrn_K7yWRrvQqKNMGVm71_oc7VbqlISr/view?usp=sharing
### Instance segmentation involves not only detecting objects in an image, but also segmenting them into individual instances.

### To perform instance segmentation using YOLO, you can follow these general steps:

### Train a YOLO model on a dataset that includes annotations for both object detection and segmentation. The annotations should include bounding boxes for the objects as well as segmentation masks for each instance.

### Modify the YOLO architecture to output both the bounding boxes and segmentation masks for each detected object. This can be done by adding additional output channels to the network.

### During inference, run the YOLO model on an input image and use the output segmentation masks to segment each object into its individual instance. This can be done by applying a segmentation algorithm, such as watershed or mean shift, to the segmentation masks.

### Finally, post-process the segmented instances to refine the results and remove any false positives or duplicates.

### albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       1/10      4.66G      2.262      5.927      5.814       2.43          6        640: 100% 21/21 [00:17<00:00,  1.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.60it/s]
                   all         73        102      0.173       0.28      0.152     0.0734      0.128      0.179      0.115     0.0341

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       2/10      5.94G      1.756      3.729      2.817       2.06          5        640: 100% 21/21 [00:08<00:00,  2.41it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:02<00:00,  1.34it/s]
                   all         73        102      0.362       0.44      0.311      0.132      0.321      0.332      0.243     0.0768

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       3/10      5.94G      1.533       3.05      2.135      1.876          5        640: 100% 21/21 [00:08<00:00,  2.43it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:02<00:00,  1.32it/s]
                   all         73        102      0.522       0.45      0.395      0.183      0.413      0.341      0.299      0.109

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       4/10      5.94G      1.427      2.735      1.859       1.77          8        640: 100% 21/21 [00:08<00:00,  2.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.53it/s]
                   all         73        102      0.552      0.357      0.383      0.198      0.581      0.365      0.383      0.158

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       5/10      5.94G      1.497      2.561      1.836      1.845          3        640: 100% 21/21 [00:08<00:00,  2.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.69it/s]
                   all         73        102       0.67      0.536      0.584      0.322      0.659      0.524      0.525      0.193

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       6/10      5.94G      1.482      2.462      1.632      1.788          8        640: 100% 21/21 [00:09<00:00,  2.30it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.79it/s]
                   all         73        102      0.504      0.478      0.447      0.216      0.411      0.393      0.358      0.152

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       7/10      5.94G      1.475       2.51       1.51      1.763          3        640: 100% 21/21 [00:09<00:00,  2.15it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.75it/s]
                   all         73        102      0.551      0.493      0.472      0.228      0.485      0.414      0.401      0.172

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       8/10      5.94G      1.349      2.264       1.34      1.656          3        640: 100% 21/21 [00:09<00:00,  2.11it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.79it/s]
                   all         73        102      0.497      0.473      0.418      0.206      0.476      0.464      0.354      0.153

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       9/10      5.94G      1.245      2.111      1.253      1.612          3        640: 100% 21/21 [00:09<00:00,  2.12it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.74it/s]
                   all         73        102      0.543      0.611      0.583      0.315       0.56      0.639      0.578      0.258

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      10/10      5.94G      1.216      2.165      1.291      1.591          6        640: 100% 21/21 [00:09<00:00,  2.15it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:03<00:00,  1.31s/it]
                   all         73        102      0.649      0.629      0.646      0.356      0.707      0.665       0.69      0.302
                 
                 YOLOv8s-seg summary (fused): 195 layers, 11780374 parameters, 0 gradients, 42.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 3/3 [00:03<00:00,  1.21s/it]
                   all         73        102      0.665      0.627      0.647      0.357      0.715      0.665      0.691      0.302
                   Cracks-and-spalling         73         44      0.698      0.788      0.777      0.442      0.768      0.864      0.866       0.44
                  object         73         58      0.633      0.466      0.517      0.273      0.662      0.466      0.516      0.165
                
### ![image](https://user-images.githubusercontent.com/69897673/220198782-676ae15a-813f-46d8-91b7-4d10a9ed52ca.png)

### ![image](https://user-images.githubusercontent.com/69897673/220198805-0cb164e1-8d44-4ee1-a144-71d12607c230.png)

### ![image](https://user-images.githubusercontent.com/69897673/220198828-abee7867-1619-4c31-bcca-75305fdc5fd2.png)



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
