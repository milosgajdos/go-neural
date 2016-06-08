# go-neural

[![GoDoc](https://godoc.org/github.com/milosgajdos83/go-neural?status.svg)](https://godoc.org/github.com/milosgajdos83/go-neural)
[![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://travis-ci.org/milosgajdos83/go-neural.svg?branch=master)](https://travis-ci.org/milosgajdos83/go-neural)
[![Go Report Card](https://goreportcard.com/badge/milosgajdos83/go-neural)](https://goreportcard.com/report/github.com/milosgajdos83/go-neural)
[![codecov](https://codecov.io/gh/milosgajdos83/go-neural/branch/master/graph/badge.svg)](https://codecov.io/gh/milosgajdos83/go-neural)

This folder will [hopefully] contain some implementations of various Neural Networks learning algorithms. You cna find example[s] of particular learning algorithms in `cmd/` subfolders.
Currently the repo only contains a basic implementation of backpropagation algorithm for 3 layers Neural Network classifier with 10 different labels.

## Backpropagation of Feedforward Neural Network

Backpropagation NN example uses a sample of 5000 images from [MNIST](http://yann.lecun.com/exdb/mnist/) database for training and validation. It performs classification of digits from 0-9.

### Usage

The current code has been tested on Go 1.6. You can get it [here](https://storage.googleapis.com/golang/go1.6.2.darwin-amd64.pkg)

```
make
time ./_build/bprop -labeled -data ./testdata/data.csv -lambda 5.0 -iters 80 -classes 10
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

Neural net accuracy: 94.220000

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

You can see that the network has 94% accuracy for the given parameters on the training set. We have classified a chosen input data to 10 different classes out of which the last one shows the highest probability.
