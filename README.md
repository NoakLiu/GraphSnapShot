This project implements the Static Pre-Sampling and Dynamic Re-Sampling for Efficient Graph Learning Storage and Retrieval.

![model construction](./assets/SSDReS.png)

Result Analysis

setting 1:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 1
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_1hop/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_1hop/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_1hop/Training_Time_versus_Alpha_Changes.png)

setting 2:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 3
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_3hop/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_3hop/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_3hop/Training_Time_versus_Alpha_Changes.png)

   

setting 3:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 4
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_4hop/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_4hop/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_4hop/Training_Time_versus_Alpha_Changes.png)

   