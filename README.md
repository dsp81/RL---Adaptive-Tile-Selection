# Efficient Poverty Mapping from High Resolution Remote Sensing Images

This project presents an efficient pipeline for estimating poverty indicators across geospatial regions using high-resolution satellite imagery. By leveraging a hybrid model combining object detection and reinforcement learning for adaptive tile selection, our approach significantly reduces computational overhead while maintaining accuracy in poverty estimation.

## Objective

To develop a lightweight and scalable poverty mapping system that:
- Uses low resolution satellite imagery as input
- Selects the most informative tiles for which we should acquire high resolution imagery using object-detection and RL
- Predicts poverty metrics for each region

## Motivation

Our goal is to reduce costs for the poverty mapping process by only acquire High Resolution images for some tiles (regions) which carry the most information while retaining similar accuracy to the approach in which we use High Resolution images for the whole region.

---

## Approach Overview

Our pipeline integrates low-cost remote sensing with intelligent high-resolution image selection to predict poverty levels efficiently and accurately.

1. Input Sources
Low-Resolution Imagery (Sentinel-2, freely available at 10m resolution)

Ground Truth Labels from LSMS survey data (cluster-level poverty index)

High-Resolution Imagery (We made this dataset using Google-Earth) *More details in the files.

2. Stage 1: Object Detection Feature Extraction
Use YOLOv8, pretrained on xView dataset, to detect objects in 1000×1000px high-res tiles.

Extract 10-dimensional feature vectors (counts of parent object classes like buildings, vehicles, etc.)

Aggregate tile-wise object counts into a village-level feature vector.

These vectors act as input features for downstream poverty prediction.

3. Stage 2: Adaptive Tile Selection using RL
The high-resolution image is divided into T tiles, which are further broken down into S sub-tiles.

A two-step MDP is designed:

Step 1: Policy network takes low-res tile as input and outputs a selection mask over sub-tiles.

Step 2: Object detection is run only on selected sub-tiles to save cost.

The policy is trained using REINFORCE (Policy Gradient) to maximize prediction accuracy while minimizing image acquisition cost.

4. Poverty Prediction
We used a XGBoost regressor on the aggregated object counts.

Predict cluster-level poverty index 

Here's the KAGGLE Notebook implementation : https://www.kaggle.com/code/digvijaysinghparihar/yolo-rl/edit
