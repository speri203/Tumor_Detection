
# Brain Tumor Detection and Segmentation

---

### Overview

Dataset: [https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)

Within this project I try to classify and segment (at the pixel level) tumors found within brain magnetic resonance imaging (MRI) scans. This particular problem can assist doctors in diagnosing tumors when it may be difficult to identify otherwise. 

This is an example of narrow AI that is trained to perform a particular task on particular image types. In recent years, researchers have come up with models and algorithms that can outperform humans when given enough data, time, and the right features. This project consists of two models. The first model is implemented using ResNet50 and is responsible for detecting tumors within MRI scans. The second model uses the UNet architecture to take the detection from the first model and perform a pixel level segmentation the tumor. This stacked model is able to achieve high accuracy on both tasks.

### Results

---

I will briefly go over the results before going into more depth about the dataset and training process.

1. ResNet50 Tumor Detection
    - Firstly, I will present a confusion matrix representing the detection

        ![https://github.com/speri203/Tumor_Detection/blob/master/images/confusion_matrix.png](https://github.com/speri203/Tumor_Detection/blob/master/images/confusion_matrix.png)

        The confusion matrix presents the overall correct and incorrect detection on the test dataset. It should be noted that the model has never seen the test dataset, so it has no opportunity to "memorize" the detection.

    - Secondly, I will present the Precision, Recall, and F1 score of the detection process

        ![https://github.com/speri203/Tumor_Detection/blob/master/images/Screen%20Shot%202020-10-01%20at%209.11.14%20PM.png](https://github.com/speri203/Tumor_Detection/blob/master/images/Screen%20Shot%202020-10-01%20at%209.11.14%20PM.png)

        - Precision is defined as the model predicting tumor and the label (ground truth) is a tumor. In other words, when the model predicted true how often was it correct.
        - Recall is defined as when the label (ground truth) is a tumor and the model detects it as a tumor.  In other words, when actually it was true how often did the model get it correct.
        - F1-score is a function of precision and recall and is used when we seek a balance between precision and recall metrics.
        - **The overall accuracy (using F-1 score) is 92%**
2. UNet segmentation accuracy
    - The metrics to evaluate mask accuracy is mIoU or mean intersection over union. The formula to calculate this metric is as follows:

        ![https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png](https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png)

        Image credit: [https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

    - The metric takes the overlapping pixels and divides it by the total number number of pixels in the ground truth  and the prediction.
    - **The accuracy of the segmentation is ~96%**

        ![https://github.com/speri203/Tumor_Detection/blob/master/images/mask_predictions.png](https://github.com/speri203/Tumor_Detection/blob/master/images/mask_predictions.png)

        - In the above image the second column (ground truth) is the actual tumor location and segmentation. The third column (prediction) is the models prediction as to the location and segmentation.
        - The blue segmentation is the ground truth overlaid on top of the MRI scan and the red segmentation is the prediction overlaid on the MRI.

### Dataset/Training

---

The dataset consists of 3929 MRI images and their tumor masks if a tumor exists within the scan. of these 3929 images there are a total of 1373 that actually contain tumors or about 35%.

The dataset is split up into multiple folders containing a pairs of MRI scans and their masks.

data.csv features:

- patient_id: a string value of patient ids
- image_path: a string value of the folder this particular MRI is stored (file path)
- mask_path: a string value of the folder this particular mask is stored (file path)
- mask: binary value (0, 1) representing if a tumor is present within the image.

For the ResNet50 model the dataset was split up into 75% for training, 12.5% for testing and 2.5% for validation. The model was trained using Google Colab using a Nvidia Tesla K80 GPU. The ResNet50 convolutional neural network (CNN) was downloaded pretrained on ImageNet and the weights were frozen. This was so the concept of **transfer learning** so the ResNet50 model didn't have to be trained. A custom neural network was created with two hidden layers with 256 neurons each and using the relu activation function. A dropout of 0.3 was added to remove codependence between neurons. The final layer has two neurons and uses the softmax activation. 

For the UNet model only the images with tumors were considered (1373). The training/test/validation split is 75%, 7.5%, and 7.5% respectively. The tversky loss function was used and the code for these functions were found at: [https://github.com/nabsabraham/focal-tversky-unet](https://github.com/nabsabraham/focal-tversky-unet) 

The UNet architecture implemented is as follows:

![https://miro.medium.com/max/2824/1*f7YOaE4TWubwaFF7Z1fzNw.png](https://miro.medium.com/max/2824/1*f7YOaE4TWubwaFF7Z1fzNw.png)

Image link: [https://miro.medium.com/max/2824/1*f7YOaE4TWubwaFF7Z1fzNw.png](https://miro.medium.com/max/2824/1*f7YOaE4TWubwaFF7Z1fzNw.png)

Both of the models were trained for 50 epochs and this was sufficient to achieve the accuracies listed above. 

### Key Points

---

The following python packages were used.

```python
pandas, seaborn, matplotlib, pyplot, sklearn, OpenCV, Tensorflow 2.0 
```

Concepts Utilized.

```python
Transfer Learning, Machine Learning, Deep Learning, Convolutional Neural Networks,
Respective Key performance indicators (KPIs), Training/Testing/Validation splitting,
Image Augmentation, and Residual Networks (ResNet)
```
