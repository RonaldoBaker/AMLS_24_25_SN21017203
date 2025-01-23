# AMLS Assignment Academic Year 2024/25

This GitHub repository contains all the files used to complete the AMLS Assigment for the academic year 2024/25. This assignment demonstrates the application of machine learnign algorithms to the problem of classifying medical images. It includes implementations using deep learning and traditional machine learning algorithms. The datasets used are part of the [MedMNIST collection](https://github.com/MedMNIST/MedMNIST), which must be downloaded and saved in the `Datasets` folder before attempting to replicate this work.

# Table of Contents
1. [Repository Structure](#repository-structure)
2. [Requirements](#requirements)
3. [Setup and Installation](#setup-and-installation)
4. [Dataset](#dataset)
5. [Usage](#usage)

## Repository Structure
- [A/](A/)
  - [acquisitionA.py](A/acquisitionA.py) - `Acquires the data from the MedMNIST API with useful metadata`
  - [cnn_modelA.pth](A/cnn_modelA.oth) - `The saved CNN model for Task A. Is used when Task A is run in 'test' mode`
  - [mainA.py](A/mainA.py) - `Main program that trains and tests all models for Task A`
  - [preprocessingA.py](A/preprocessingA.py) - `Preprocesses the acquired data to be used in 'mainA.py'`
  - [taskAmodels.py](A/taskAmodels.py) - `Contains the CNN model and other helper classes for model training`
- [B/](B/)
  - [acquisitionB.py](B/acquisitionB.py) - `Acquires the data from the MedMNIST API with useful metadata`
  - [cnn_modelB.pth](B/cnn_modelB.pth) - `The saved CNN model for Task B. Is used when Task B is run in 'test' mode`
  - [mainB.py](B/mainB.py) - `Main program that trains and tests all models for Task B`
  - [taskBmodels.py](B/taskBmodels.py) - `Contains the CNN model and other helper classes for model training`
  - [tuning.py](B/tuning.py) - `Optimises all models used in Task B (this can be run from the CL)`
- [Datasets](Datasets/) - `Datasets will need to be manually saved in this folder before running programs`
  - breastmnist.npz
  - bloodmnist.npz
- [env](env/)
  - [environment.yml](environment.yml) - `Contains information to create virtual environment for this project`
  - [requirements.txt](requirements.txt) - `List of required modules that the 'environment.yaml' file will use`
- [main.py](main.py) `The main script to run both Task A and Task B`

## Requirements
The requirements for this project are as listed in [requirements.txt](requirements.txt).

## Setup and Installation
Setup for this project can be carried out in different ways. In this project, a `.yml` file was used to create a conda environment and install the required modules from `requirements.txt`.

1. Clone project to local machine
```
>>> git clone
```

2. Create conda environment from `environment.yml` file:
```
 >>> conda create env -f environment.yml
```
3. **OR** if directly installing requirements from `.txt` file:
```
>>> pip install -r requirements.txt
```

## Dataset


## Usage
The project can be run in various ways. You can choose to run individual tasks or both tasks at once. Each task can be run in either `train` or `test` mode. `Train` mode means the CNN specifically will train **and** test. `Test` mode means the saved model in the respective file will be loaded and only evaluated on the test set.

Default:
```
>>>  python main.py
```
This will run both tasks in `test` mode. This means that specifically the CNNs will not train, but will load the saved models and only evaluate on the test set (for the sake of run time).

To run one project in either `train` or `test` mode:
```
>>> python main.py a=train
```
To run both projects in either `train` or `test` mode:
```
>>> python main.py a=train b=test
```
The program will only expect up to two arguments. The argument must start with `a=` or `b=`, and the mode is specified after the `=` operator. If any inputs are not in the required format, the default mode will be run.

