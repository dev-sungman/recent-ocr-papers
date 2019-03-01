---
html header: <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
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

* **Overall Architecture**

  <img src="./images/FOTS/architecture.png" width="1000px" height="200px">

  * The **backbone of the shared network** is ResNet-50. Inspired by FPN, concatenate low-level feature maps and high-level semantic feature maps.
  * The **text detection branch** outputs dense per-pixel prediction of text using features produced by shared convolutions.
  * With oriented text region proposals produced by detection branch, the proposed **RoIRotate** converts corresponding shared features into fixed-height representations while keeping the original region aspect ratio.
  * Finally, the **text recognition branch** recognizes words in region proposals. CNN and LSTM are adopted to encode text sequence information, followd by a CTC decoder.
  * **Shared conv net architecture**



<img src="./images/FOTS/shared_conv.png" width="500px" height="150px">

* **Text Detection Branch**

  * As there are a lot of small text boxes in natural scene images, we *upscale the feautre maps from 1/32 to 1/4 size of the original input image* in shared convolutions.

  * After extracting shared features, one convolution is applied to output dense per-pixel predictions of words. 

  * The *first channel computes the probability of each pixel being a positive sample*. Similar to EAST, pixels in shrunk version of the original text regions are considered positive. For each positive sample, the *following 4 channels predict its distances to top, bottom, left, right sides of the bounding box* that contains this pixel, and the last channel predict the orientation of the related bounding box. 

  * Final detection results are produced by applying **thresholding and NMS** to these positive samples.

  * In our experiments, we observe that many patterns similar to text stroke are hard to classify, such as fences, latices, etc. we adopt **online hard example mining(OHEM)** to better distinguish these patterns.

  * The **detection branch loss function is composed of two terms**: text classification and bounding box regression. text classification term can be seen as pixel-wise classification loss for a down-sampled score map. *Only shrunk version of the original text region is considered as the positive area*, while the area between the bounding box and the shrunk version is considered as "NOT CARE", and does not contribute to the loss for the classification.

  * Denote the set of selected positive elements by OHEM in the score map as:
    
    $$
    { L }_{ cls }\quad =\quad \frac { 1 }{ |\Omega | } \sum _{ x\in \Omega  }^{  }{ H({ p }_{ x },{ p }_{ x }^{ * }) }
    $$

    $$
    \qquad\qquad\qquad\qquad\qquad\quad\qquad  = \quad \frac { 1 }{ |\Omega | } \sum _{ x\in \Omega  }^{  }{ (-{ p }_{ x }^{ * }log{ p }_{ x }-(1-{ p }_{ x }^{ * })log(1-{ p }_{ x })) }
    $$
    

    * where |.| is the number of elements in a set, and $$H({ p }_{ x },{ p }_{ x }^{ * }) $$ represents the cross entropy loss between $$p_x$$, the prediction of the score map, and $$p_x^*$$, the binary label that indicates text or non-text.
      

  * As for the regression loss, we adopt the IoU loss and the rotation angle loss, since they are robust to variation in object shape, scale and orientation:
    
    $$
    { L }_{ reg }\quad =\quad \frac { 1 }{ |\Omega | } \sum _{ x\in \Omega  }^{  }{IoU(R_x,R_x^*) + \lambda_\theta(1-cos(\theta_x,\theta_x^*))}
    $$
    

    * Here, $$IoU(R_x,R_x^*)$$ is the IoU loss between the predicted bounding box $$R_x$$, and the ground truth $$R_x^*$$. 
    * Second term is rotation angle loss. We set the hyper-parameter $$\lambda_\theta$$ to 10 in experiments.
      

  * Therefore the full detection loss can be written as:
    
    $$
    L_{detect} = L_{cls} +\lambda_{reg}L_{reg}
    $$
    

    * where a hyper-parameter $$\lambda_{reg}$$ balacnes two losses, which is set to 1 in our experiments.
      

* **ROIRotate**

  * RoIRotate applies transformation on oriented feature regions to obtain axis-aligned feature maps.
    

  <img src="./images/FOTS/roi_rotate.png" width="400px" height="150px">

  

  * Fix the output height and keep the aspect ratio unchanged to deal with the variation in text length.

  * Compared to RoIPooling and RoIAlign, RoIRotate provides a more general operation for extracting features for regions of interest. 

  * We also compare to **RROI pooling** proposed in RRPN. *RRoI pooling transforms the rotated region to fixed size region through max-pooling,* while we use **bilinear interpolation** to compute the values of the output. <u>This operation avoids misalignments between the RoI and the extracted features, and additionally it makes the lengths of the output features variable, which is more suitable for text recognition</u>.

  * This process can be divided into two steps. 

    * First, affine transformation parameters are computed via predicted or ground truth coordinates of text proposals. Then, affine transformations are applied to shared feature maps for each region respectively, and canonical horizontal feature maps of text regions are obtained. The first step can be formulated as:

      
      $$
      t_x = l*cos\theta - t*sin\theta - x
      $$

      $$
      t_y = t*cos\theta + l*sin\theta - y
      $$

      $$
      s = \frac {h_t}{t+b}
      $$

      $$
      w_t = s * (l + r)
      $$

      
      $$
      \\ M\quad =\quad \begin{bmatrix} cos\theta  & -sin\theta  & 0 \\ sin\theta  & cos\theta  & 0 \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} s & 0 & 0 \\ 0 & s & 0 \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} 1 & 0 & t_ x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}
      $$

      $$
      = s\begin{bmatrix} cos\theta  & -sin\theta  & t_xcos\theta - t_ysin\theta \\ sin\theta  & cos\theta  & t_xsin\theta + t_ycos\theta \\ 0 & 0 & \frac1s \end{bmatrix}
      $$

      

    * 

