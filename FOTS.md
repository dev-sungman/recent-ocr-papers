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



* The fully convolutional network based oriendted text detection branch is built on top of the feature map to predict the detection boxes.
* The RoIRotate operator extracts text proposal features corresponding to the detection results from the feature map. 
* The text proposal features are then fed into RNN encoder and CTC decoder for text recognition.
* FOTS is the first end-to-end trainable framework for oriented text detection and recognition.



## Methodology

* Overall Architecture

  <img src="./images/FOTS/architecture.png" width="1000px" height="200px">

  * The **backbone of the shared network** is ResNet-50. Inspired by FPN, concatenate low-level feature maps and high-level semantic feature maps.
  * The **text detection branch** outputs dense per-pixel prediction of text using features produced by shared convolutions.
  * With oriented text region proposals produced by detection branch, the proposed **RoIRotate** converts corresponding shared features into fixed-height representations while keeping the original region aspect ratio.
  * Finally, the **text recognition branch** recognizes words in region proposals. CNN and LSTM are adopted to encode text sequence information, followd by a CTC decoder.
  * **Shared conv net architecture**



<img src="./images/FOTS/shared_conv.png" width="500px" height="150px">

* Text Detection Branch
  * As there are a lot of small text boxes in natural scene images, we *upscale the feautre maps from 1/32 to 1/4 size of the original input image* in shared convolutions.
  * After extracting shared features, one convolution is applied to output dense per-pixel predictions of words. 
  * The *first channel computes the probability of each pixel being a positive sample*. Similar to EAST, pixels in shrunk version of the original text regions are considered positive. For each positive sample, the *following 4 channels predict its distances to top, bottom, left, right sides of the bounding box* that contains this pixel, and the last channel predict the orientation of the related bounding box. 
  * Final detection results are produced by applying **thresholding and NMS** to these positive samples.
  * In our experiments, we observe that many patterns similar to text stroke are hard to classify, such as fences, latices, etc. we adopt **online hard example mining(OHEM)** to better distinguish these patterns.





