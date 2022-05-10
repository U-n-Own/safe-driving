# Safe driving
+ This repo is used to maintain my thesis project about Safe Driving, training a CNN model on data using Federated Learning 
+ Currently instead of using a Federated Learning environment, i'm simulating it using a single GPU.


## Testing results for Centralized:
+ Simple CNN no FedAvg Algorithm:  5 to 10 epochs reaching 0.93 to 0.97 accuracy (May overfit or too complex model)
+ Federate Learning simulated: 10 epochs reaching 0.90 accuracy


## Testing results for Federated Learning vs. Centralized Learning 
+ In 5 round reached 0.80 accuracy and in 10 rounds reached 0.90
+ From the plot we can say that suffers overfitting
+ This image is from old version that contains some problems, need to be rerun to get the appropriate plot
<img src="https://github.com/U-n-Own/safe-driving/blob/main/src/plots/orizontalPlot.png">

## Testing result for Images in dataset vs. Held Aside test set 
<img src="https://github.com/U-n-Own/safe-driving/blob/main/src/plots/federated_learning_plot_after_rework.png?raw=true">

## About Federated Learning 
+ Main property we want are : 
  - Non-IIDness : in an ideal fed. learning experiment you want a lot of data that are Independent and Not Identically Distributed


