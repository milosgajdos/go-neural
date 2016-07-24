# go-neural

[![GoDoc](https://godoc.org/github.com/milosgajdos83/go-neural?status.svg)](https://godoc.org/github.com/milosgajdos83/go-neural)
[![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://travis-ci.org/milosgajdos83/go-neural.svg?branch=master)](https://travis-ci.org/milosgajdos83/go-neural)
[![Go Report Card](https://goreportcard.com/badge/milosgajdos83/go-neural)](https://goreportcard.com/report/github.com/milosgajdos83/go-neural)
[![codecov](https://codecov.io/gh/milosgajdos83/go-neural/branch/master/graph/badge.svg)](https://codecov.io/gh/milosgajdos83/go-neural)

This project provides a basic implementation of Feedforward Neural Network classifier. You can find an example implementation of [backpropagation algorithm](https://en.wikipedia.org/wiki/Backpropagation) for a 3 layers [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) multivariate classifier in the project's `cmd/` subfolder. In the future I will be hopefully adding more interesting and more advanced examples.

The code in this project has been developed and tested with the following version of Go:

```
$ go version
go version go1.6.3 darwin/amd64
```

## Get started

Get the source code:

```
$ go get -u github.com/milosgajdos83/go-neural
```

Once you have successfully downloaded all the packages you can start building simple neural networks using the packages provided by the project. For example, if you want to create a simple feedforward neural network you can do so using the following code:

 ```go
package main

import (
	"fmt"
	"os"

	"github.com/milosgajdos83/go-neural/neural"
	"github.com/milosgajdos83/go-neural/pkg/config"
)

func main() {
	netConfig := &config.NetConfig{
		Kind: "feedfwd",
		Arch: &config.NetArch{
			Input: &config.LayerConfig{
				Kind: "input",
				Size: 100,
			},
			Hidden: []*config.LayerConfig{
				&config.LayerConfig{
					Kind: "hidden",
					Size: 25,
					NeurFn: &config.NeuronConfig{
						Activation: "sigmoid",
					},
				},
			},
			Output: &config.LayerConfig{
				Kind: "output",
				Size: 500,
				NeurFn: &config.NeuronConfig{
					Activation: "softmax",
				},
			},
		},
	}
	net, err := neural.NewNetwork(netConfig)
	if err != nil {
		fmt.Printf("Error creating network: %s\n", err)
		os.Exit(1)
	}
	fmt.Printf("Created new neural network: %v\n", net)
}
```

You can explore the project's packages and API in [godoc](https://godoc.org/github.com/milosgajdos83/go-neural).

## Manifest

`go-neural` lets you define neural network architecture via what's called a `manifest` file which you can parse in your programs using the project's manifest parser package. An example manifest file looks like below:

```yaml
kind: feedfwd                 # network type: only feedforward networks
task: class                   # network task: only classification tasks
network:                      # network architecture: layers and activations
  input:                      # INPUT layer
    size: 400                 # 400 inputs
  hidden:                     # HIDDEN layer
    size: [25]                # Array of all hidden layers
    activation: relu          # ReLU activation function
  output:                     # OUTPUT layer
    size: 10                  # 10 outputs - this implies 10 classes
    activation: softmax       # softmax activation function
training:                     # network training
  kind: backprop              # type of training: backpropagation only
  cost: xentropy              # cost function: cross entropy (loglikelhood available too)
  params:                     # training parameters
    lambda: 1.0               # lambda is a regularizer
  optimize:                   # optimization parameters
    method: bfgs              # BFGS optimization algorithm
    iterations: 80            # 80 BFGS iterations
```

The above manifest defines 3 layers neural network which uses [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function for all of its hidden layers and [softmax](https://en.wikipedia.org/wiki/Softmax_function) for its output layer. You can also specify some advanced optmization parameters. You can explore all available parameters in `config` package.

## Backpropagation

The project provides an example program available in `cmd/bprop` directory which performs a classification of MNIST digits and then reports the classification success rate. The network accuracy is validated against the training data set for brevity. In real life example you must use a separate validation data set! Lastly, the example program also reports classification result for the first data sample.

You can build the example program by running `bprop` make task:

```
$ make bprop
```

This will place the resulting binary into `_build` project's subdirectory if the build succeeds. To test the program, there is an example data set available in the `testdata` subdirectory which contains a sample of 5000 images from [MNIST](http://yann.lecun.com/exdb/mnist/) database for training and validation.

You can see various example runs with different parameters specified via manifest file below. You can find various examples of manifest files in `cmd/bprop` folder of the project source tree.

### ReLU -> Softmax -> Cross Entropy

```
$ time ./_build/bprop -labeled -data ./testdata/data.csv -manifest cmd/bprop/example.yml
Current Cost: 3.421197
Current Cost: 3.087151
Current Cost: 2.731485
...
...
Current Cost: 0.088055
Current Cost: 0.086561
Current Cost: 0.085719
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

You can see that the neural network classification accuracy **on the training data set** is `99.96%` and that network classifies the first sample to the correct class with `99.98%` probability. This is clearly an example of overfitting.

### ReLU -> Softmax -> Log Likelihood

```
time ./_build/bprop -labeled -data ./testdata/data.csv -manifest cmd/bprop/example4.yml
Current Cost: 2.455806
Current Cost: 2.157898
Current Cost: 1.858962
...
...
Current Cost: 0.070446
Current Cost: 0.069825
Current Cost: 0.069216
Result status: IterationLimit

Neural net accuracy: 99.960000

Classification result:
⎡ 3.046878477304935e-10⎤
⎢    0.0315965041728176⎥
⎢2.1424486587649327e-05⎥
⎢ 5.349015780043783e-12⎥
⎢ 5.797172277201534e-07⎥
⎢ 2.132650877255262e-08⎥
⎢ 5.525355134815623e-06⎥
⎢  2.58203420693211e-07⎥
⎢ 0.0004521601957074575⎥
⎣     99.96792352623254⎦


real    1m28.066s
user    1m34.502s
sys     0m6.913s
```

`ReLU -> Softmax -> Log Likelihood` provides much faster convergence than the previous combination of activations and loss functions. Again, you can see that we are overfitting the training data. In real life you must tune your neural network on separate training, validation and test data sets!
