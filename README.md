### Creating a environment to pytorch

https://www.youtube.com/watch?v=eVXl7ajMHAU

https://www.cherryservers.com/blog/how-to-install-pytorch-ubuntu

* Install python
* Activate venv
* Install pytorch

### There are two ways to install pytorch 

1 - Pip install

2 - Anaconda

### Running IRIS app
```shell
python app/main_iris.py
```

### Running CIFAR-10 app
```shell
python app/main_cifar_10.py
```

### Result after running both apps
```shell
python app/main_iris.py
Epoch [1/100], Loss: 1.0230
Epoch [2/100], Loss: 0.9213
Epoch [3/100], Loss: 0.8270
Epoch [4/100], Loss: 0.7343
Epoch [5/100], Loss: 0.6529
Epoch [6/100], Loss: 0.5779
Epoch [7/100], Loss: 0.5121
Epoch [8/100], Loss: 0.4427
Epoch [9/100], Loss: 0.4039
Epoch [10/100], Loss: 0.3653
Epoch [11/100], Loss: 0.3422
Epoch [12/100], Loss: 0.3075
Epoch [13/100], Loss: 0.2966
Epoch [14/100], Loss: 0.2872
Epoch [15/100], Loss: 0.2486
Epoch [16/100], Loss: 0.2462
Epoch [17/100], Loss: 0.2215
Epoch [18/100], Loss: 0.2060
Epoch [19/100], Loss: 0.1921
Epoch [20/100], Loss: 0.1880
Epoch [21/100], Loss: 0.1799
Epoch [22/100], Loss: 0.1645
Epoch [23/100], Loss: 0.1480
Epoch [24/100], Loss: 0.1383
Epoch [25/100], Loss: 0.1303
Epoch [26/100], Loss: 0.1265
Epoch [27/100], Loss: 0.1236
Epoch [28/100], Loss: 0.1207
Epoch [29/100], Loss: 0.1076
Epoch [30/100], Loss: 0.1166
Epoch [31/100], Loss: 0.0962
Epoch [32/100], Loss: 0.0968
Epoch [33/100], Loss: 0.0939
Epoch [34/100], Loss: 0.0870
Epoch [35/100], Loss: 0.0898
Epoch [36/100], Loss: 0.0966
Epoch [37/100], Loss: 0.0835
Epoch [38/100], Loss: 0.0848
Epoch [39/100], Loss: 0.0799
Epoch [40/100], Loss: 0.0947
Epoch [41/100], Loss: 0.0801
Epoch [42/100], Loss: 0.0716
Epoch [43/100], Loss: 0.0770
Epoch [44/100], Loss: 0.0757
Epoch [45/100], Loss: 0.0753
Epoch [46/100], Loss: 0.0676
Epoch [47/100], Loss: 0.0654
Epoch [48/100], Loss: 0.0639
Epoch [49/100], Loss: 0.0648
Epoch [50/100], Loss: 0.0627
Epoch [51/100], Loss: 0.0621
Epoch [52/100], Loss: 0.0676
Epoch [53/100], Loss: 0.0607
Epoch [54/100], Loss: 0.0650
Epoch [55/100], Loss: 0.0566
Epoch [56/100], Loss: 0.0573
Epoch [57/100], Loss: 0.0570
Epoch [58/100], Loss: 0.0565
Epoch [59/100], Loss: 0.0555
Epoch [60/100], Loss: 0.0620
Epoch [61/100], Loss: 0.0576
Epoch [62/100], Loss: 0.0592
Epoch [63/100], Loss: 0.0595
Epoch [64/100], Loss: 0.0537
Epoch [65/100], Loss: 0.0645
Epoch [66/100], Loss: 0.0558
Epoch [67/100], Loss: 0.0631
Epoch [68/100], Loss: 0.0529
Epoch [69/100], Loss: 0.0501
Epoch [70/100], Loss: 0.0584
Epoch [71/100], Loss: 0.0491
Epoch [72/100], Loss: 0.0489
Epoch [73/100], Loss: 0.0500
Epoch [74/100], Loss: 0.0574
Epoch [75/100], Loss: 0.0485
Epoch [76/100], Loss: 0.0504
Epoch [77/100], Loss: 0.0625
Epoch [78/100], Loss: 0.0499
Epoch [79/100], Loss: 0.0534
Epoch [80/100], Loss: 0.0485
Epoch [81/100], Loss: 0.0626
Epoch [82/100], Loss: 0.0451
Epoch [83/100], Loss: 0.0489
Epoch [84/100], Loss: 0.0457
Epoch [85/100], Loss: 0.0586
Epoch [86/100], Loss: 0.0475
Epoch [87/100], Loss: 0.0480
Epoch [88/100], Loss: 0.0437
Epoch [89/100], Loss: 0.0480
Epoch [90/100], Loss: 0.0439
Epoch [91/100], Loss: 0.0431
Epoch [92/100], Loss: 0.0469
Epoch [93/100], Loss: 0.0453
Epoch [94/100], Loss: 0.0428
Epoch [95/100], Loss: 0.0432
Epoch [96/100], Loss: 0.0439
Epoch [97/100], Loss: 0.0420
Epoch [98/100], Loss: 0.0437
Epoch [99/100], Loss: 0.0426
Epoch [100/100], Loss: 0.0506
Accuracy: 100.00%
```

```shell
python app/main_cifar_10.py
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar-10-python.tar.gz
100.0%
Extracting data/cifar-10-python.tar.gz to data
Files already downloaded and verified
Epoch [1/10], Loss: 1.4872
Epoch [2/10], Loss: 1.1269
Epoch [3/10], Loss: 0.9836
Epoch [4/10], Loss: 0.9028
Epoch [5/10], Loss: 0.8334
Epoch [6/10], Loss: 0.7905
Epoch [7/10], Loss: 0.7502
Epoch [8/10], Loss: 0.7151
Epoch [9/10], Loss: 0.6923
Epoch [10/10], Loss: 0.6697
Accuracy: 76.33%
```