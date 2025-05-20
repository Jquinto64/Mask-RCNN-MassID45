<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

<a href="https://opensource.facebook.com/support-ukraine">
  <img src="https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB" alt="Support Ukraine - Help Provide Humanitarian Aid to Ukraine." />
</a>

Detectron2 is Facebook AI Research's next generation library
that provides state-of-the-art detection and segmentation algorithms.
It is the successor of
[Detectron](https://github.com/facebookresearch/Detectron/)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
It supports a number of computer vision research projects and production applications in Facebook.

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>
<br>

## MassID45 Instructions
1. First download the MS-COCO pretrained R50 checkpoint using
```bash
wget https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl
```

2. Make any desired modifications to the data augmentations and training hyperparameters using `configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py`

3. Modify the  `--dataset_path` argument in `train.sh` to reflect the location of the MassID45 training data. Then run the training script with 
```bash
sbatch train.sh
```
Outputs will be saved in the folder specified by `train.output_dir`.

4.  Replace `--dataset_img_path` and `--dataset_json_path` in `sahi_inference_256_datasets_v2.sh` with the locations of the MassID45 validation or testing data, respectively, then run inference with: 
```bash
sbatch sahi_inference.sh
```
Results will appear in the `runs/predict` folder under the name specified by the `--exp_name` argument.

## Learn More about Detectron2

Explain Like I’m 5: Detectron2            |  Using Machine Learning with Detectron2
:-------------------------:|:-------------------------:
[![Explain Like I’m 5: Detectron2](https://img.youtube.com/vi/1oq1Ye7dFqc/0.jpg)](https://www.youtube.com/watch?v=1oq1Ye7dFqc)  |  [![Using Machine Learning with Detectron2](https://img.youtube.com/vi/eUSgtfK4ivk/0.jpg)](https://www.youtube.com/watch?v=eUSgtfK4ivk)

## What's New
* Includes new capabilities such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend,
  DeepLab, ViTDet, MViTv2 etc.
* Used as a library to support building [research projects](projects/) on top of it.
* Models can be exported to TorchScript format or Caffe2 format for deployment.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Getting Started

See [Getting Started with Detectron2](https://detectron2.readthedocs.io/tutorials/getting_started.html),
and the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn about basic usage.

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
