# R2Gen

This is the implementation of [Exploring Intrinsic Hierarchical Organization for Medical Diagnosis](https://arxiv.org/pdf/2010.16056.pdf) at ISBI.


## Requirements

- `torch==1.7.1`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`



## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

NOTE: The `IU X-Ray` dataset is of small size, and thus the variance of the results is large.


## Train

Run `python main_train_exam.py` to train a model on the IU X-Ray data.


## Test

Run `python main_test.py` to test a model on the IU X-Ray data.


Follow [CheXpert](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt/chexpert) or [CheXbert](https://github.com/stanfordmlgroup/CheXbert) to extract the labels and then run `python compute_ce.py`. Note that there are several steps that might accumulate the errors for the computation, e.g., the labelling error and the label conversion. We refer the readers to those new metrics, e.g., [RadGraph](https://github.com/jbdel/rrg_emnlp) and [RadCliQ](https://github.com/rajpurkarlab/CXR-Report-Metric).

## Visualization

Run `python main_plot.py` to visualize the attention maps on the MIMIC-CXR data.
