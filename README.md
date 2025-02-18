# Point-Teacher
This is the official implementation of the paper "Tiny Object Detection with Single Point Supervision". [arxiv](https://arxiv.org/abs/2412.05837)

## :white_check_mark: Updates
* **`Oct. 8th, 2024`**: Update: **Important!** we release the **Point Teacher HBB version** and **Point Teacher OBB version** model!
* **`Dec. 2th, 2024`**: **If you have any questions, feel free to ask !!!**
* **`Feb. 18th, 2024`**: Update: we have made all training configuration files publicly available! These files include P2BNet, PLUG, PointOBB, PointRbox, and PointOBB-v2!

## Introduction
Point Teacher is a Robust End-to-end Point-supervised Tiny Object Detection algorithm that can be integrated into various detectors.

**Abstract**: Tiny objects, with their limited spatial resolution, often resemble point-like distributions. As a result, bounding box prediction using point-level supervision emerges as a natural and cost-effective alternative to traditional box-level supervision. However, the small scale and lack of distinctive features of tiny objects make point annotations prone to noise, posing significant hurdles for model robustness. To tackle these challenges, we propose Point Teacher—the first end-to-end point-supervised method for robust tiny object detection in aerial images. To handle label noise from scale ambiguity and location shifts in point annotations, Point Teacher employs the teacher-student architecture and decouples the learning into a two-phase denoising process. In this framework, the teacher network progressively denoises the pseudo boxes derived from noisy point annotations, guiding the student network's learning. Specifically, in the first phase, random masking of image regions facilitates regression learning, enabling the teacher to transform noisy point annotations into coarse pseudo boxes. In the second phase, these coarse pseudo boxes are refined using dynamic multiple instance learning, which adaptively selects the most reliable instance from dynamically constructed proposal bags around the coarse pseudo boxes. Extensive experiments on three tiny object datasets (i.e., AI-TOD-v2, SODA-A, and TinyPerson) validate the proposed method's effectiveness and robustness against point location shifts. Notably, relying solely on point supervision, our Point Teacher already shows comparable performance with box-supervised learning methods.

<div align=center>
<img src="https://github.com/ZhuHaoranEIS/Point-Teacher/blob/main/Supplement_details/background.jpg" width="700px">
</div>

<div align=center>
<img src="https://github.com/ZhuHaoranEIS/Point-Teacher/blob/main/Supplement_details/compare.jpg" width="1000px">
</div>

## Method
![demo image](https://github.com/ZhuHaoranEIS/Point-Teacher/blob/main/Supplement_details/point_teacher_framework.jpg)

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


Install:

Note that this repository is based on the [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMRotate](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/ZhuHaoranEIS/Point-Teacher.git
cd Point-Teacher/HBB_TOD or Point-Teacher/OBB_TOD
pip install -r requirements/build.txt
python setup.py develop
```

## Prepare datasets

- Please refer to [AI-TOD](https://github.com/Chasel-Tsui/mmdet-aitod) for AI-TOD-v2.0 and AI-TOD-v1.0 dataset.
- Please refer to [SODA](https://github.com/shaunyuan22/SODA-mmrotate) for SODA-A and SODA-D dataset.

## Training

All models of Point Teacher are trained with a total batch size of 2 (can be adjusted following [MMDetection](https://github.com/open-mmlab/mmdetection)). 

- To train Point Teacher on AI-TOD-v2.0, run

```shell script
# For center point
python tools/train.py configs/point_teacher/aitodv2_point_teacher_0%.py --gpu-id 0

# For random point
python tools/train.py configs/point_teacher/aitodv2_point_teacher_100%.py --gpu-id 0

```

- To train Point Teacher on SODA-A, run

```shell script
# For center point
python tools/train.py configs/point_teacher/sodaa_fcos_point_teacher_1x.py --gpu-id 0

```

## Inference

- Modify [test.py](https://github.com/ZhuHaoranEIS/Point-Teacher/blob/main/HBB_TOD/tools/test.py)

```/path/to/model_config```: modify it to the path of model config, e.g., ```configs/Point_Teacher/aitodv2_point_teacher_1x.py```

```/path/to/model_checkpoint```: modify it to the path of model checkpoint


- Run
```
python tools/test.py configs/Point_Teacher/aitodv2_point_teacher_1x.py /path/to/model_checkpoint --eval bbox
```

## Main results
![results](https://github.com/ZhuHaoranEIS/Point-Teacher/blob/main/Supplement_details/table1.jpg)

![results](https://github.com/ZhuHaoranEIS/Point-Teacher/blob/main/Supplement_details/table2.jpg)


## Visualization
The images are from the AI-TOD-v2.0 datasets. Note that the <font color=green>green box</font> denotes the True Positive, the <font color=red>red box</font> denotes the False Negative and the <font color=blue>blue box</font> denotes the False Positive predictions.
![demo image](https://github.com/ZhuHaoranEIS/Point-Teacher/blob/main/Supplement_details/detection_results.jpg)

## Citation
If you find this work helpful, please consider citing:
```bibtex
@misc{point_teacher,
      title={Tiny Object Detection with Single Point Supervision}, 
      author={Haoran Zhu and Chang Xu and Ruixiang Zhang and Fang Xu and Wen Yang and Haijian Zhang and Gui-Song Xia},
      year={2024},
      eprint={2412.05837},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05837}, 
}
```
