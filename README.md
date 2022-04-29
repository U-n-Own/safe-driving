# Safe driving
+ This repo is used to maintain my thesis project about Safe Driving, training a CNN model on data using Federated Learning 


# First tests
+ Currently instead of using a Federated Learning environment, i'm simulating it using a single GPU.


# Testing results for Centralized:
+ Simple CNN no FedAvg Algorithm:  5 to 10 epochs reaching 0.93 to 0.97 accuracy (May overfit or too complex model)


# Testing results for Federated
+ In 5 round reached 0.80 accuracy and in 10 rounds reached 0.90
+ From the plot we can say that suffers overfitting

<img src="https://github.com/U-n-Own/safe-driving/blob/main/src/plots/orizontalPlot.png">

# About Federated Learning 
+ Main property we want are : 
  - Non-IIDness : in an ideal fed. learning experiment you want a lot of data that are Independent and Not Identically Distributed


