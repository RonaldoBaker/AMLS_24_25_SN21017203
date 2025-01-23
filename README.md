# AMLS Assignment Academic Year 2024/25

This GitHub repository contains all the files used to complete the AMLS Assigment for the academic year 2024/25. This assignment demonstrates the application of machine learnign algorithms to the problem of classifying medical images. It includes implementations using deep learning and traditional machine learning algorithms. The datasets used are part of the [MedMNIST collection](https://github.com/MedMNIST/MedMNIST), which must be downloaded and saved in the `Datasets` folder before attempting to replicate this work.

# Table of Contents
1. [Repository Structure](#repository-structure)
2. [Setup & Installation](#setup-&-installation)
3. [Dataset](#dataset)
4. [Usage](#usage)

## Repository Structure
- [A/](A/)
  - [acquisitionA.py](acquisitionA.py) - `Acquires the data from the MedMNIST API with useful metadata`
  - cnn_modelA.pth - `The saved CNN model for Task A. Is used when Task A is run in 'test' mode`
  - mainA.py - `Main program that trains and tests all models for Task A`
  - preprocessingA.py - `Preprocesses the acquired data to be used in 'mainA.py'`
  - taskAmodels.py - `Contains the CNN model and other helper classes for model training`
- B
  - acquisitionB.py - `Acquires the data from the MedMNIST API with useful metadata`
  - cnn_modelB.pth - `The saved CNN model for Task B. Is used when Task B is run in 'test' mode`
  - mainB.py - `Main program that trains and tests all models for Task B`
  - taskBmodels.py - `Contains the CNN model and other helper classes for model training`
  - tuning.py - `Optimises all models used in Task B (this can be run from the CL)`
- Datasets - `Datasets will need to be manually saved in this folder before running programs`
  - breastmnist.npz
  - bloodmnist.npz
- env
  - environment.yaml - `Contains information to create virtual environment for this project`
  - requirements.txt - `List of required modules that the 'environment.yaml' file will use`
- main.py `The main script to run both Task A and Task B`

## Setup & Installation

## Dataset

## Usage
