# go-neural

[![GoDoc](https://godoc.org/github.com/milosgajdos83/go-neural?status.svg)](https://godoc.org/github.com/milosgajdos83/go-neural)
[![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://travis-ci.org/milosgajdos83/go-neural.svg?branch=master)](https://travis-ci.org/milosgajdos83/go-neural)
[![Go Report Card](https://goreportcard.com/badge/milosgajdos83/go-neural)](https://goreportcard.com/report/github.com/milosgajdos83/go-neural)
[![codecov](https://codecov.io/gh/milosgajdos83/go-neural/branch/master/graph/badge.svg)](https://codecov.io/gh/milosgajdos83/go-neural)

This project will contain basic implementations of various Neural Networks learning algorithms. You will find here simple examples in the projects `cmd/` subfolders. Currently there is only an example implementation of [backpropagation algorithm](https://en.wikipedia.org/wiki/Backpropagation) for a 3 layers [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) multivariate classifier. In the future I will be hopefully adding more interesting and more advanced learning algorithms for different kinds of neural networks.

The code in this project has been developed and tested with the following version of Go:

```
$ go version
go version go1.6.2 darwin/amd64
```

## Get started

Get the source code:

```
$ go get -u github.com/milosgajdos83/go-neural
```
 
 Once you have successfully downloaded the package you can start building simple neural network programs using the packages provided by the project. For example, if you want to create a simple feedforward neural network you can do so using the following code:
 
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

You can explore the project's packages and API in [godoc](https://godoc.org/github.com/milosgajdos83/go-neural).

## Manifest

`go-neural` lets you define your neural network architecture via what's called a `manifest` file which you can parse in your programs using a simple manifest parser package. An example manifest file looks like below:

```yaml
kind: feedfwd
task: class
layers:
  input:
    size: 400
  hidden:
    size: [25]
    activation: relu
  output:
    size: 10
    activation: softmax
training:
  kind: backprop
  params: "lambda=1.0"
optimize:
  method: bfgs
  iterations: 80
```

The above manifest defines 3 layers neural network which uses [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function for all of its hidden layers and [softmax](https://en.wikipedia.org/wiki/Softmax_function) for its output layer. You can also notice that you can specify some parameters that are passed to the backpropagation algorithm as well as some optmization parameters.

## Backpropagation

Project provides an example program available in `cmd/bprop` directory which performs classification of digits from the `0-9` range and then reports the resulting classification success rate. The network accuracy is validated against the training data set. In real life example you should use a separate validation data set! Lastly, the example program also reports classification result for the first data sample.

You can build the example program by running `bprop` make task:

```
$ make bprop
```

This will place the resulting binary into `_build` project's subdirectory if the build succeeds. To test the program, there is an example data set available in the `testdata` subdirectory which contains a sample of 5000 images from [MNIST](http://yann.lecun.com/exdb/mnist/) database for training and validation.

You can see various example runs with different parameters specified via manifest file below. You can find the example manifest files in `cmd/bprop` folder of the project source tree.

### ReLU -> Softmax

```
$ time ./_build/bprop -labeled -data ./testdata/data.csv -manifest cmd/bprop/example.yml
Current Cost: 3.415054
Current Cost: 3.081002
Current Cost: 2.725169
...
...
Current Cost: 0.041690
Current Cost: 0.039663
Current Cost: 0.037856
Current Cost: 0.037296
Result status: IterationLimit

Neural net accuracy: 99.960000

Classification result:
⎡ 1.943663671946687e-11⎤
⎢  0.012190159604108151⎥
⎢ 8.608094616279243e-05⎥
⎢1.0168917476209273e-12⎥
⎢ 1.348762594753421e-07⎥
⎢1.7017240294954928e-08⎥
⎢3.4414528109461814e-07⎥
⎢3.8031639047418544e-07⎥
⎢ 0.0002904105962281858⎥
⎣     99.98743247247788⎦


real	1m40.244s
user	1m38.561s
sys	0m6.071s
```

You can see that the neural network accuracy on the training data set is staggering `99.96%` and that network classifies the first sample to the correct class with `99.98%` probability. Pretty awesome ☺️

### Sigmoid -> Softmax
```
$ time ./_build/bprop -labeled -data ./testdata/data.csv -manifest cmd/bprop/example2.yml
Current Cost: 3.341391
Current Cost: 3.250739
Current Cost: 3.126012
Current Cost: 2.905813
Current Cost: 2.562841
...
...
Current Cost: 0.243739
Current Cost: 0.234082
Current Cost: 0.227347
Current Cost: 0.225749
Result status: IterationLimit

Neural net accuracy: 96.820000

Classification result:
⎡0.00021616794862335984⎤
⎢  0.033884368797300564⎥
⎢  0.003877769382182497⎥
⎢2.5445434981607396e-05⎥
⎢  0.028102698328725285⎥
⎢ 0.0010378827363716836⎥
⎢   0.09126650795952473⎥
⎢ 0.0006346632491998938⎥
⎢  0.015888682987386542⎥
⎣     99.82506581317571⎦


real	1m18.432s
user	1m21.314s
sys	0m5.474s
```

You can see that the neural network accuracy on the training data set is staggering `96.82%` and that network classifies the first sample to the correct class with `99.82%` probability. This is a bit worse than the previous configuration.


### tanh -> tanh
```
$ time ./_build/bprop -labeled -data ./testdata/data.csv -manifest cmd/bprop/example3.yml
Current Cost: 8.301051
Current Cost: 3.982616
Current Cost: 3.464993
Current Cost: 3.095790
Current Cost: 2.874504
...
...
Current Cost: 0.174036
Current Cost: 0.171023
Current Cost: 0.167725
Current Cost: 0.166274
Current Cost: 0.166334
Current Cost: 0.166211
Result status: IterationLimit

Neural net accuracy: 98.500000

Classification result:
⎡ 3.851416213218354e-05⎤
⎢  0.000887320935591917⎥
⎢   0.08430464866100201⎥
⎢1.8784497566246995e-09⎥
⎢  0.006346359320569978⎥
⎢  0.005993924083999875⎥
⎢  0.000956220309679265⎥
⎢  3.70271079686156e-05⎥
⎢   0.02248533121878889⎥
⎣      99.8789506523218⎦


real	3m9.253s
user	3m20.706s
sys	0m12.341s
```

You can see that the neural network accuracy on the training data set is staggering `98.5%` and that network classifies the first sample to the correct class with `99.87%` probability. A bit better than the sigmoid example, but still worse thatn ReLU+Softmax.
