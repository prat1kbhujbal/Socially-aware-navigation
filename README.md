# Socially-aware-navigation

## Overview
Human pose estimation using deep learning method to predict trajectory-based intent from historical human poses using Kalman filter and utilize these predictions for socially aware navigation.

## Dataset
[**MPII Human Pose Dataset**](http://human-pose.mpi-inf.mpg.de/)

## Pipeline
<p align="middle">
  <img src="Assets/pipeline.png"/>
</p>

## Model Architecture
<p align="middle">
  <img src="Assets/Architecture_Horizontal.png"/>
</p>

## Results
### Preprocessed image with filtered annotations
<p align="middle">
  <img src="Results/filtered.png"/>
</p>

### Model output in simulation
<p align="center">
<img src="./Results/sim_output.gif"/>
</p>

### Navigation
<p align="center">
<img src="./Results/nav_output.png" width= 500/>
</p>

## Refrences
> [1] A plugin for simulation of human pedestrians in ROS Gazebo.
> https://github.com/robotics-upo/gazebo_sfm_plugin.
> Accessed: 2022-12-01
