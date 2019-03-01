---

---

# FOTS: Fast Oriented Text Spotting with a Unified Network

Xuebo Liu, Ding Liang, Shi Yan, Dagui Chen, Yu Qiao, and Junjie Yan

SenseTime Group Ltd. Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences.

## Abstract

* Most existing methods treat text detection and recognition as separate tasks.
* end-to-end trainable FOTS network for simultaneous detection and recognition, **sharing computation and visual inforamtion among the two complemntary tasks.**
* **RoIRotate** is introduced to share convolutional features between detection and recognition.
* state-of-the-art methods. ICDAR 2015, ICDAR 2018 MLT and ICDAR 2013



## Introduction



* The most common way in scene text readings is to divide it into text detection and text recognition, which are handled as two separate tasks.
* In text detection, usually a convolutional neural network is used to extract feature maps from a scene image, and then different decoders are used to decode the regions.
* Text recognition, a network for **sequential prediction is conducted on top of text regions**, one by one. It **leads to heavy time cost** especially for images with a number of text regions.
* **It ignores the correlation in visual cues shared in detection and recognition.**
* **The key to connec detection and recognition is the ROIRotate**, which gets proper features from feature maps according to the oriented detection bounding boxes.

<img src="./images/FOTS/architecture.png" width="1000px" height="200px">

* The fully convolutional network based oriendted text detection branch is built on top of the feature map to predict the detection boxes.
* The RoIRotate operator extracts text proposal features corresponding to the detection results from the feature map. 
* The text proposal features are then fed into RNN encoder and CTC decoder for text recognition.





