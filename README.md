# go-neural

[![GoDoc](https://godoc.org/github.com/milosgajdos83/go-neural?status.svg)](https://godoc.org/github.com/milosgajdos83/go-neural)
[![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://travis-ci.org/milosgajdos83/go-neural.svg?branch=master)](https://travis-ci.org/milosgajdos83/go-neural)
[![Go Report Card](https://goreportcard.com/badge/milosgajdos83/go-neural)](https://goreportcard.com/report/github.com/milosgajdos83/go-neural)
[![codecov](https://codecov.io/gh/milosgajdos83/go-neural/branch/master/graph/badge.svg)](https://codecov.io/gh/milosgajdos83/go-neural)

This project contains implementations of various Neural Networks learning algorithms. You will find simple example[s] of neural networks in `cmd/` subfolders. Currently the provides only a simple implementation of [backpropagation algorithm](https://en.wikipedia.org/wiki/Backpropagation) for a simple 3 layers [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) classifier. In the future I might be adding more interesting and advanced learning algorithms for different kinds of neural networks.

The current code has been tested on Go 1.6.

## Get started

Get the source code:

```
$ go get -u github.com/milosgajdos83/go-neural
```
 
 Once you have the source code you can start building simple programs using the provided `go-neural` packages. For example, if you want to create a simple feedforward neural network you can do so using the following code:
 
 ```go
 package main

import (
	"fmt"
	"os"

	"github.com/milosgajdos83/go-neural/neural"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
)

func main() {
	nf := &neural.NeuronFunc{ForwFn: matrix.SigmoidMx, BackFn: matrix.SigmoidGradMx}
	hiddenLayers := []int{4, 5}
	arch := &neural.NetworkArch{Input: 10, Hidden: hiddenLayers, Output: 10}
	config := &neural.Config{Kind: neural.FEEDFWD, Arch: arch, ActFunc: nf}
	net, err := neural.NewNetwork(config)
	if err != nil {
		fmt.Printf("Error creating network: %s\n", err)
		os.Exit(1)
	}
	fmt.Printf("Create new %s neural network: %s\n", net.Kind(), net.ID())
}
```

You can explore the project's packages in [godoc](https://godoc.org/github.com/milosgajdos83/go-neural)

## Backpropagation

There is an example implementation of backrpropagation learning program available in `cmd/bprop` directory. The example program performs classification of digits from in the `0-9` range and then reports the resulting classification success rate. The network accuracty is validated against the training data set. In real life example you should ideally use a separate validation data set! Lastly, the program also reports classification result for the first data sample.

You can build the backpropagation example program by running `bprop` make task:

```
$ make bprop
```

This will place the resulting binary into `_build` project's subdirectory if the build succeeds. To test the program, there is an example data set available in the `testdata` subdirectory which contains a sample of 5000 images from [MNIST](http://yann.lecun.com/exdb/mnist/) database for training and validation.

You can see the example run below:

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

The neural network has `~94%` accuracy when validated on the training set.
You can also see a chosen data sample classification to particular class - the last one being the most probable with `~96%` probability rate.
