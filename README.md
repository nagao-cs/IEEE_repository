# Reliability Metrics for N-Version Object Detection

This repository contains the dataset and evaluation tools used in the paper "Reliability Metrics for N-Version Object Detection".

## Overview

This research proposes new metrics for evaluating the reliability of N-Version object detection systems.
The proposed metrics are:

- Cov_OD: Error coverage rate in object detection
- Cer_OD: Prediction certainty in object detection

## Repository Structure

```
.
├── dataset/          # Evaluation dataset
│   ├── center/      # Detection results from center camera
│   ├── left_1/      # Detection results from left camera 1
│   ├── right_1/     # Detection results from right camera 1
│   └── ...
├── src/             # Evaluation source code
│   ├── ODMetrics.py # Implementation of proposed metrics
│   ├── bestConb.py  # Camera combination optimization
│   └── ...
└── README.md
```

## Usage

### Calculate All Metrics

To calculate all metrics (Cov, Cer, Cov_PD, Cer_OD) for 1-7 version detection system:

```bash
python ./src/ODMetrics.py
```

### Find Optimal Camera Combinations

To find the top 10 best camera combinations for each metric:

```bash
python ./src/bestConb.py
```