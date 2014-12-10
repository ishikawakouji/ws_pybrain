ws_pybrain
==========

Workspace for PyBrain

handNN
------

handNN.py read CSV file as DataSets of training and testing.

CSV format is bellow:

```
1: header
2-last: input1, input2, ...., inputN, class number
```

handNN needs class number should start ONE not ZERO.

``` python
import handNN
handNN.runNN()
```
