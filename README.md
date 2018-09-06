# Understanding Deep CNNs via Interpretable Individual Units

This is the program of personal individual project, the academic part of MSc degree in Master of Science of Imperial College London.

## Introduction
The problem we are interested in is to identify the specific semantic meaning of units within a CNN. In this project, we adapt a quantitative interpretability analysing framework named Network Dissection from [NetDissect](https://github.com/CSAILVision/NetDissect) to perform the identification job and verify its result by an ablation test and visual validation. Currently, our program can provide a complete and detailed semantic meaning identification for VGG16 on the datasets of PASCAL Part and COCO, and visualise the activation map of any unit inside it. However, our program is designed to be easily scaled to support other datasets and models.

## Preparation
Before running the program, there are two things, downloading datasets and installing dependencies, should be done.

### Downloading Datasets
We support two datasets, [PASCAL Part](http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html) and [COCO](http://cocodataset.org/#home), to perform the identification. Unfortunately, we do not finish the script for automatically downloading and preparing the dataset. In other words, anyone want to set up the program from scratch should:

1. Download the dataset to be used on the corresponding website
2. Organise the data and annotations as the requirements of the program, i.e. following the directories setting in the `src/config.py`

### Downloading the Pre-trained Models
In this project, we probe the pre-trained models instead of training new ones. The defualt examined model is VGG16, whose well-trained weights is obtained from [machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg). Again, you are required to download it by yourself and put it in the correct directory with the defined name: 

* `pre-models/vgg16/vgg16_params.npy`.

### Installing Dependencies
The used third-party packages are listed as below:

1. opencv-python 3.4.1
2. tensorflow 1.8.0
3. matplotlib
4. skiimage
5. addict

Apart from them, the library, `cocoapi`, for managing the data of COCO should be recompiled to adapt the running system environment by two steps:

1. `cd utils/cocoapi/PythonAPI/`
2. `make`

## Running Instructions
One more thing before really running the identification or other experiments is to map the data and encode the classes. After obtaining the working data mappings, other functionalities are free to be executed. By the way, Several frequently used instructions are encoded in the `makefile`. So for convenience, please check the file to get the commands.

### Mapping Data
The generated mapping files describe the encoding of classes and the information of samples, which are required by the `utils/helper/data_loader.py` to load the data from specific directories of dataset. The mapping operation can be simply done by running command:

* `make data`

### Running the experiments
After mapping data, the experiments can be executed using the commands:

* Identify semantic meaning of untis: `make match`
* Sensitivity analysis on activation threshold: `make activthres`
* Statistical verification: `make verify`
* Visualisation: `make visualise`

### Configurations
Almost all configurations are specified in `src/config.py`, so you should check the global variables defined in the file and modify the variables in response to the configurations you want to change.

## Custom Data & Model
The support for the new dataset or model can not be completed automatically. You need to, except downloading them, also write some codes and set up the related configurations in `src/config.py`.

### Custom Data
The instructions to add custom data include:

1. download the data and set up the configurations
2. remap the data to generate new mapping files
3. add the functions for loading the images and parsing the corresponding annotations in a similar manner defined in `utils/helper/data_loader.py`

### Custom Model
The instructions contain:

1. download the pre-trained weights data, put it in the `pre-models/` and set up the corresponding configurations
2. write the architecture specifications as the same style as `pre-models/vgg16/vgg16_config.py`
3. if the data file is not encoded in a form of `.npy`, you may also need to consider writing your own parameter loading functions.







