Test FBL Results
```
Loaded graph: Graph(num_nodes={'author': 1134649, 'field_of_study': 59965, 'institution': 8740, 'paper': 736389},
      num_edges={('author', 'affiliated_with', 'institution'): 1043998, ('author', 'writes', 'paper'): 7145660, ('field_of_study', 'rev_has_topic', 'paper'): 7505078, ('institution', 'rev_affiliated_with', 'author'): 1043998, ('paper', 'cites', 'paper'): 10832542, ('paper', 'has_topic', 'field_of_study'): 7505078, ('paper', 'rev_writes', 'author'): 7145660},
      metagraph=[('author', 'institution', 'affiliated_with'), ('author', 'paper', 'writes'), ('institution', 'author', 'rev_affiliated_with'), ('paper', 'paper', 'cites'), ('paper', 'field_of_study', 'has_topic'), ('paper', 'author', 'rev_writes'), ('field_of_study', 'paper', 'rev_has_topic')])
Number of embedding parameters: 154029312
Number of model parameters: 337460
start training...
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:39<00:00, 3956.58it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:45<00:00, 7006.96it/s]
Run: 01, Epoch: 01, Loss: 2.3288, Train: 62.54%, Valid: 48.78%, Test: 48.17%
Epoch 01: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:47<00:00, 3754.08it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:44<00:00, 7041.53it/s]
Run: 01, Epoch: 02, Loss: 1.5384, Train: 76.17%, Valid: 49.18%, Test: 48.00%
Epoch 02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:45<00:00, 3809.68it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:44<00:00, 7046.30it/s]
Run: 01, Epoch: 03, Loss: 1.1387, Train: 84.54%, Valid: 47.89%, Test: 46.71%
Run 01:
Highest Train: 84.54
Highest Valid: 49.18
  Final Train: 76.17
   Final Test: 48.00
start training...
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:35<00:00, 4040.00it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:45<00:00, 6999.27it/s]
Run: 02, Epoch: 01, Loss: 2.3123, Train: 62.17%, Valid: 47.67%, Test: 47.18%
Epoch 01: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:50<00:00, 3691.48it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:45<00:00, 7012.46it/s]
Run: 02, Epoch: 02, Loss: 1.5326, Train: 76.43%, Valid: 48.67%, Test: 47.24%
Epoch 02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:50<00:00, 3697.09it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:45<00:00, 6990.98it/s]
Run: 02, Epoch: 03, Loss: 1.1333, Train: 84.55%, Valid: 47.74%, Test: 46.32%
Run 02:
Highest Train: 84.55
Highest Valid: 48.67
  Final Train: 76.43
   Final Test: 47.24
start training...
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:45<00:00, 3792.72it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:44<00:00, 7028.86it/s]
Run: 03, Epoch: 01, Loss: 2.3205, Train: 62.82%, Valid: 49.50%, Test: 48.70%
Epoch 01: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:42<00:00, 3865.23it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:44<00:00, 7019.46it/s]
Run: 03, Epoch: 02, Loss: 1.5273, Train: 76.85%, Valid: 48.14%, Test: 47.27%
Epoch 02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:43<00:00, 3843.75it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:45<00:00, 6965.29it/s]
Run: 03, Epoch: 03, Loss: 1.1234, Train: 84.90%, Valid: 47.71%, Test: 46.59%
Run 03:
Highest Train: 84.90
Highest Valid: 49.50
  Final Train: 62.82
   Final Test: 48.70
start training...
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:48<00:00, 3728.16it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:49<00:00, 6741.26it/s]
Run: 04, Epoch: 01, Loss: 2.2841, Train: 63.53%, Valid: 48.53%, Test: 47.81%
Epoch 01: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:49<00:00, 3707.63it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:49<00:00, 6729.94it/s]
Run: 04, Epoch: 02, Loss: 1.4975, Train: 77.51%, Valid: 48.10%, Test: 47.15%
Epoch 02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:48<00:00, 3734.01it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:48<00:00, 6756.13it/s]
Run: 04, Epoch: 03, Loss: 1.0835, Train: 85.79%, Valid: 47.38%, Test: 45.79%
Run 04:
Highest Train: 85.79
Highest Valid: 48.53
  Final Train: 63.53
   Final Test: 47.81
start training...
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:52<00:00, 3639.30it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:48<00:00, 6799.30it/s]
Run: 05, Epoch: 01, Loss: 2.3204, Train: 62.81%, Valid: 48.06%, Test: 47.04%
Epoch 01: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:39<00:00, 3943.79it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:45<00:00, 6967.60it/s]
Run: 05, Epoch: 02, Loss: 1.5340, Train: 76.11%, Valid: 49.36%, Test: 48.29%
Epoch 02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:49<00:00, 3706.71it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:49<00:00, 6717.83it/s]
Run: 05, Epoch: 03, Loss: 1.1258, Train: 84.84%, Valid: 47.73%, Test: 46.18%
Run 05:
Highest Train: 84.84
Highest Valid: 49.36
  Final Train: 76.11
   Final Test: 48.29
start training...
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:53<00:00, 3623.17it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:50<00:00, 6692.23it/s]
Run: 06, Epoch: 01, Loss: 2.3205, Train: 62.24%, Valid: 48.81%, Test: 48.67%
Epoch 01: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:53<00:00, 3636.01it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:47<00:00, 6869.42it/s]
Run: 06, Epoch: 02, Loss: 1.5411, Train: 75.97%, Valid: 49.55%, Test: 48.41%
Epoch 02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:52<00:00, 3644.26it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:45<00:00, 6980.84it/s]
Run: 06, Epoch: 03, Loss: 1.1447, Train: 84.15%, Valid: 47.84%, Test: 46.40%
Run 06:
Highest Train: 84.15
Highest Valid: 49.55
  Final Train: 75.97
   Final Test: 48.41
start training...
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:44<00:00, 3832.05it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:46<00:00, 6898.63it/s]
Run: 07, Epoch: 01, Loss: 2.3185, Train: 62.97%, Valid: 49.37%, Test: 48.91%
Epoch 01: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:51<00:00, 3664.36it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:51<00:00, 6603.93it/s]
Run: 07, Epoch: 02, Loss: 1.5338, Train: 76.90%, Valid: 48.72%, Test: 47.70%
Epoch 02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:52<00:00, 3659.62it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:49<00:00, 6746.64it/s]
Run: 07, Epoch: 03, Loss: 1.1307, Train: 84.87%, Valid: 47.77%, Test: 46.47%
Run 07:
Highest Train: 84.87
Highest Valid: 49.37
  Final Train: 62.97
   Final Test: 48.91
start training...
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:49<00:00, 3708.30it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:47<00:00, 6858.63it/s]
Run: 08, Epoch: 01, Loss: 2.3396, Train: 62.58%, Valid: 48.47%, Test: 47.81%
Epoch 01: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:46<00:00, 3785.14it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:47<00:00, 6832.28it/s]
Run: 08, Epoch: 02, Loss: 1.5470, Train: 76.20%, Valid: 48.20%, Test: 47.45%
Epoch 02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:48<00:00, 3740.68it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:51<00:00, 6611.23it/s]
Run: 08, Epoch: 03, Loss: 1.1417, Train: 84.65%, Valid: 46.85%, Test: 45.69%
Run 08:
Highest Train: 84.65
Highest Valid: 48.47
  Final Train: 62.58
   Final Test: 47.81
start training...
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:51<00:00, 3679.06it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:51<00:00, 6617.58it/s]
Run: 09, Epoch: 01, Loss: 2.3307, Train: 63.12%, Valid: 49.42%, Test: 48.96%
Epoch 01: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:48<00:00, 3742.31it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:50<00:00, 6663.03it/s]
Run: 09, Epoch: 02, Loss: 1.5289, Train: 76.87%, Valid: 48.01%, Test: 46.79%
Epoch 02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:47<00:00, 3763.93it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:49<00:00, 6696.23it/s]
Run: 09, Epoch: 03, Loss: 1.1217, Train: 84.93%, Valid: 47.66%, Test: 46.56%
Run 09:
Highest Train: 84.93
Highest Valid: 49.42
  Final Train: 63.12
   Final Test: 48.96
start training...
Epoch 00: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [14:07<00:00, 742.63it/s]
Inference: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [22:00<00:00, 557.60it/s]
Run: 10, Epoch: 01, Loss: 2.3043, Train: 63.28%, Valid: 48.78%, Test: 47.81%
Epoch 01: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [06:27<00:00, 1625.41it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:46<00:00, 6900.92it/s]
Run: 10, Epoch: 02, Loss: 1.5201, Train: 76.60%, Valid: 48.88%, Test: 47.45%
Epoch 02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:49<00:00, 3722.42it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:45<00:00, 6969.32it/s]
Run: 10, Epoch: 03, Loss: 1.1173, Train: 85.00%, Valid: 47.74%, Test: 46.00%
Run 10:
Highest Train: 85.00
Highest Valid: 48.88
  Final Train: 76.60
   Final Test: 47.45
Final performance: 
All runs:
Highest Train: 84.82 ± 0.42
Highest Valid: 49.09 ± 0.42
  Final Train: 69.63 ± 6.99
   Final Test: 48.16 ± 0.60
```

Test FCR Results
```

```

Test FCR-SC Results
```

```

Test OTF Results
```

```

Test OTF-SC Results
```

```