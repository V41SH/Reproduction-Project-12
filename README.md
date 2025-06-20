# Reproduction-Project-12

Reproduction of "Learning Steerable Filters for Rotation Equivariant CNNs"

Below are the experiments and their related scripts:
* Visualization of circular harmonics
```
harmonic_viz.py
```
* Experiment with SFCNN architecture for Table 1 from the paper
```
train.py
test.py
models/sfcnn.py
```
* Experiment for figure 4 right from the paper
``` 
train_cnn.py 
test_cnn.py 
models/cnn.py
```

* Experiment for rotation equivariant U-net (ISBI)
```
train_memnet.py
eval_memnet.py
models/memnet.py
```

* Z2CNN baseline from "Group Equivariant Convolutional Networks"
```
baseline.py
```