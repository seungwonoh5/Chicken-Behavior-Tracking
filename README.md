# Chicken-Behavior-Tracking
This is a collaborative project between the Departments of Computer Science and Agriculture at University of Maryland, College Park. The goal of this research is to understand the behavior of chickens using object detection and tracking algorithms, such as Mask-RCNN and DeepSort. The different behaviors we are currently tracking include: sitting, standing, feeding, pecking, accessing water and dust bowl, and social pecking. Such behavior analysis would enable us to understand the well being of the chickens. Understanding the behavior of chickens would help identify any medical or other conditions of the chicken in a similar or relatively similar setting. The immediate and the most important one is the similarity of the chickens which makes tracking them a challenging problem.

This repo contains training mask-RCNN and Faster-RCNN object detector with Detectron2 (Pytorch-based object detection library) on a custom chicken video dataset collected from Agriculture department to replace YOLOv3 detector in DeepSORT tracking algorithm to improve chicken tracking performance.

## Installation
Clone this repository by running ```git clone https://github.com/seungwonoh5/Chicken-Behavior-Tracking```

## Dependencies
To run our program, it requires the following:
numpy==1.19.5
pandas==1.1.5
matplotlib==3.2.2
tensorflow==2.4.1
scikit-learn==0.22.2
All packages will be installed automatically when running pip3 install -r requirements.txt to install all the dependencies.

## What's Included
Inside the repo, there are 4 scripts and 1 notebook file.
data.py: this file provides all the data loading and preprocessing functions. (need to modify to use it for your own dataset)
models.py: this file provides all the decoder models in Keras.
utils.py: this file provides all the visualization and misc functions.
main.py: this file serves as the main script for the experiment that trains both the decoder and the baseline model and compare the results.
main.ipynb: an example file that trains and plots the experimental results for visualization on Google Colab.
Getting Started

We have included jupyter notebooks that provide detailed examples of our experiments. You can either run the notebook main.ipynb which goes through the complete process of the experiment and outputs visualizations of the experiment or run main script that imports other script modules in the repo to goes through the whole experiment. You can use individual scripts to reuse part of the program that you need.

python main.py

## Results
We perform extensive experiments on four datasets sequentially streaming and we show that an online setting continuously updating the model as every data block is processed leads to significant improvements over batch learning that the model is fixed after trained on an initial dataset and deployed for prediction. We show that performance of batch learning degrades after a few datasets where as online learning shows stable performance and even improves on future datasets.


## Contact
Author: Seungwon Oh - wonoh90@gmail.com. To ask questions or report issues, please open an issue on the issues tracker. Discussions, suggestions and questions are welcome!
