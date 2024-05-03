# Training-MNIST-under-8k
Training MNIST under 8k Params with 99.5% Val accuracy.

## Problem Statement

Attempt to consistently reach a validation accuracy of **99.4% over the last few epochs within a training span of 15 epochs**, using MNIST data. Ensure the model maintains a parameter count of under **10,000**, with an even preferable ceiling below **8,000** parameters.

#### Model Parameters
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
            Conv2d-4            [-1, 9, 24, 24]             648
              ReLU-5            [-1, 9, 24, 24]               0
       BatchNorm2d-6            [-1, 9, 24, 24]              18
            Conv2d-7           [-1, 10, 22, 22]             810
              ReLU-8           [-1, 10, 22, 22]               0
       BatchNorm2d-9           [-1, 10, 22, 22]              20
        MaxPool2d-10           [-1, 10, 11, 11]               0
           Conv2d-11             [-1, 12, 9, 9]           1,080
             ReLU-12             [-1, 12, 9, 9]               0
      BatchNorm2d-13             [-1, 12, 9, 9]              24
           Conv2d-14             [-1, 16, 7, 7]           1,728
             ReLU-15             [-1, 16, 7, 7]               0
      BatchNorm2d-16             [-1, 16, 7, 7]              32
           Conv2d-17             [-1, 20, 5, 5]           2,880
             ReLU-18             [-1, 20, 5, 5]               0
      BatchNorm2d-19             [-1, 20, 5, 5]              40
           Conv2d-20             [-1, 10, 5, 5]             200
        AvgPool2d-21             [-1, 10, 1, 1]               0
================================================================
Total params: 7,568
Trainable params: 7,568
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.42
Params size (MB): 0.03
Estimated Total Size (MB): 0.45
----------------------------------------------------------------
```
#### Training Log
```
Epoch 1
Train: Loss=0.1790 Batch_id=234 Accuracy=82.90: 100%|████████████████████████████████████████████████████████| 235/235 [00:15<00:00, 15.15it/s]
Test set: Average loss: 0.1200, Accuracy: 9611/10000 (96.11%)

Epoch 2
Train: Loss=0.0813 Batch_id=234 Accuracy=96.45: 100%|████████████████████████████████████████████████████████| 235/235 [00:15<00:00, 14.75it/s]
Test set: Average loss: 0.0536, Accuracy: 9838/10000 (98.38%)

Epoch 3
Train: Loss=0.0595 Batch_id=234 Accuracy=97.41: 100%|████████████████████████████████████████████████████████| 235/235 [00:16<00:00, 14.61it/s]
Test set: Average loss: 0.0421, Accuracy: 9875/10000 (98.75%)

Epoch 4
Train: Loss=0.0939 Batch_id=234 Accuracy=97.69: 100%|████████████████████████████████████████████████████████| 235/235 [00:15<00:00, 14.78it/s]
Test set: Average loss: 0.0297, Accuracy: 9912/10000 (99.12%)

Epoch 5
Train: Loss=0.0977 Batch_id=234 Accuracy=98.05: 100%|████████████████████████████████████████████████████████| 235/235 [00:15<00:00, 14.69it/s]
Test set: Average loss: 0.0302, Accuracy: 9903/10000 (99.03%)

Epoch 6
Train: Loss=0.0364 Batch_id=234 Accuracy=98.15: 100%|████████████████████████████████████████████████████████| 235/235 [00:15<00:00, 14.77it/s]
Test set: Average loss: 0.0292, Accuracy: 9906/10000 (99.06%)

Epoch 7
Train: Loss=0.0407 Batch_id=234 Accuracy=98.49: 100%|████████████████████████████████████████████████████████| 235/235 [00:18<00:00, 12.38it/s]
Test set: Average loss: 0.0203, Accuracy: 9940/10000 (99.40%)

Epoch 8
Train: Loss=0.0100 Batch_id=234 Accuracy=98.57: 100%|████████████████████████████████████████████████████████| 235/235 [00:18<00:00, 12.42it/s]
Test set: Average loss: 0.0193, Accuracy: 9944/10000 (99.44%)

Epoch 9
Train: Loss=0.0355 Batch_id=234 Accuracy=98.57: 100%|████████████████████████████████████████████████████████| 235/235 [00:18<00:00, 12.46it/s]
Test set: Average loss: 0.0190, Accuracy: 9944/10000 (99.44%)

Epoch 10
Train: Loss=0.0248 Batch_id=234 Accuracy=98.66: 100%|████████████████████████████████████████████████████████| 235/235 [00:15<00:00, 14.77it/s]
Test set: Average loss: 0.0198, Accuracy: 9941/10000 (99.41%)

Epoch 11
Train: Loss=0.0293 Batch_id=234 Accuracy=98.64: 100%|████████████████████████████████████████████████████████| 235/235 [00:19<00:00, 12.32it/s]
Test set: Average loss: 0.0194, Accuracy: 9951/10000 (99.51%)

Epoch 12
Train: Loss=0.0476 Batch_id=234 Accuracy=98.67: 100%|████████████████████████████████████████████████████████| 235/235 [00:18<00:00, 12.39it/s]
Test set: Average loss: 0.0199, Accuracy: 9944/10000 (99.44%)

Epoch 13
Train: Loss=0.0543 Batch_id=234 Accuracy=98.63: 100%|████████████████████████████████████████████████████████| 235/235 [00:18<00:00, 12.39it/s]
Test set: Average loss: 0.0189, Accuracy: 9947/10000 (99.47%)

Epoch 14
Train: Loss=0.0386 Batch_id=234 Accuracy=98.70: 100%|████████████████████████████████████████████████████████| 235/235 [00:15<00:00, 14.79it/s]
Test set: Average loss: 0.0189, Accuracy: 9947/10000 (99.47%)

Epoch 15
Train: Loss=0.0233 Batch_id=234 Accuracy=98.63: 100%|████████████████████████████████████████████████████████| 235/235 [00:19<00:00, 12.29it/s]
Test set: Average loss: 0.0190, Accuracy: 9948/10000 (99.48%)
```
