# Chicken-Behavior-Tracking
This is a collaborative project between the Departments of Computer Science and Agriculture at University of Maryland, College Park. The goal of this research is to understand the behavior of chickens using object detection and tracking algorithms, such as Mask-RCNN and DeepSort. The different behaviors we are currently tracking include: sitting, standing, feeding, pecking, accessing water and dust bowl, and social pecking. Such behavior analysis would enable us to understand the well being of the chickens. Understanding the behavior of chickens would help identify any medical or other conditions of the chicken in a similar or relatively similar setting. The immediate and the most important one is the similarity of the chickens which makes tracking them a challenging problem.

This repo contains training mask-RCNN and Faster-RCNN object detector with Detectron2 (Pytorch-based object detection library) on a custom chicken video dataset collected from Agriculture department to replace YOLOv3 detector in DeepSORT tracking algorithm to improve chicken tracking performance.
