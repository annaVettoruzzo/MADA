**MADA Meta-Learning Automatic Data Augmentation**
Code for the paper "Efficient Few-Shot Human Activity Recognition via Meta-Learning
and Data Augmentation" currently under review.

The proposed approach augments time series data with a range of transformations, each assigned a 
learnable weight, enabling the model to prioritize the most effective augmentations and discard the
irrelevant ones. Throughout the meta-training phase, the model learns to identify an optimal
weighted combination of these transformations, significantly improving the model’s adaptabil-
ity and generalization to new situations with scarce labeled data. During the meta-test phase,
this knowledge enables the model to efficiently learn from and adapt to a very limited set of
labeled samples from completely new subjects undertaking entirely new activities.

# Data
 * ADL - Activities of Daily Living Recognition with Wrist-worn Accelerometer
 * DSA - Daily and Sports Activities dataset
 * PAMAP - Physical Activity Monitoring dataset
 * VPA - Vicon Physical Action dataset
 * WISDM - Smartphone and Smartwatch Activity and Biometrics dataset
 * REALDISP - Activity Recognition dataset
Data can be downloaded from the [ UCI Machine Learning
Repository]([https://www.google.com](https://archive.ics.uci.edu/).
The code for preprocessing the data is in /data/*dataset_name*/preprocessing.ipynb.


# Usage
For training the models, use the train.py file. For testing run the test.py file.

