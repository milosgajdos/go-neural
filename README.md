**THIS PROJECT IS IN AN EXPERIMENTAL PHASE. CONSIDER YOURSELF WARNED ;-)**

# go-neural

This folder will [hopefully] contain some implementations of various Neural Networks learning algorithms. You cna find example[s] of particular learning algorithms in `cmd/` subfolders.
Currently the repo only contains a basic implementation of backpropagation algorithm for 3 layers Neural Network classifier with 10 different labels.

## Backpropagation of Feedforward Neural Network

Backpropagation NN example uses a sample of 5000 images from [MNIST](http://yann.lecun.com/exdb/mnist/) database for training and validation. It performs classification of digits from 0-9.

### Usage

The current code has been tested on Go 1.6. You can get it [here](https://storage.googleapis.com/golang/go1.6.2.darwin-amd64.pkg)

```
time ./bprop -data /Users/milosgajdos/data.csv -labeled -lambda 5.0 -iters 80 -classes 10
Current Cost: 6.824417
Current Cost: 4.202914
Current Cost: 3.265800
Current Cost: 3.233592
Current Cost: 0.572877
...
...
...
Current Cost: 0.513280
Current Cost: 0.511507
Current Cost: 0.509390
Result status: IterationLimit

Neural net success: 94.220000

Classification result:
⎡ 0.00653138058035365⎤
⎢ 0.28605827837318315⎥
⎢  0.3162491186408352⎥
⎢0.026172330168689206⎥
⎢  1.4916033781624203⎥
⎢  0.1623987908121907⎥
⎢ 0.45414039911787174⎥
⎢   0.474234411421992⎥
⎢ 0.24277298719691356⎥
⎣   96.53983892552554⎦


real    1m28.822s
user    1m32.359s
sys     0m6.385s
```

You can see that the network whos 94% accuracy for the given parameters. We have classified a chosen input data to 10 different classes out of which the last one shows the highest probability.
