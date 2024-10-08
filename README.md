# Point-Teacher
This is the official implementation of the paper "Point Teacher: Robust End-to-End Tiny Object Detection with Single Point Supervision". [arxiv](xxx)

## :white_check_mark: Updates
* **`Oct. 8th, 2024`**: Update: **Important!** we release the **Point Teacher HBB version** and **Point Teacher OBB version** model!

## Introduction
Point Teacher is a Robust End-to-End Point-Supervised TOD algorithm that can be integrated into either one-stage or two-stage algorithms.

**Abstract**: Point-supervised object detection (PSOD) has recently gained significant attention due to its cost-effectiveness. However, existing methods often assume that point annotations are located in the center or central region of the object, an assumption that is inadequate for tiny objects that occupy very few pixels. To address this limitation, we propose Point Teacher, a robust end-to-end point-based tiny object detector. In this work, we revisit the point annotation challenge as a label noise problem and introduce a two-stage training paradigm. The first stage is Preliminary Learning phase, where the network learns to regress regions by randomly removing parts of the image, helping it to develop preliminary spatial perception capabilities. The second stage is Denoising Learning phase, where we implement a self-optimizing teacher-student network along with a dynamic multiple instance learning (DMIL) approach. The teacher network generates coarse pseudo-boxes based on preliminary learning, which DMIL then refines for effective student network training. Moreover, we introduce a noise-resistant IoU loss, termed Shaking IoU Loss. Shaking IoU Loss mitigates the network's tendency to overfit noisy boxes by introducing perturbations to the regression targets. Extensive experiments on synthetic point-based datasets (i.e., AI-TOD-v2, SODA-A, and TinyPerson) demonstrate the robustness and effectiveness of our Point Teacher.

<div align=center>
<img src="https://github.com/ZhuHaoranEIS/Point-Teacher/blob/main/Supplement_details/background.png" width="500px">
</div>

## Method
![demo image](https://github.com/ZhuHaoranEIS/Point-Teacher/blob/main/Supplement_details/overall_wo_background.png)

## Installation and Get Started
[![Python](https://img.shields.io/badge/python-3.7%20tested-brightgreen)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.10.0%20tested-brightgreen)](https://pytorch.org/)

Required environments:
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)
* [DN-TOD](https://github.com/ZhuHaoranEIS/DN-TOD)
* [SODA](https://github.com/shaunyuan22/SODA-mmrotate)
