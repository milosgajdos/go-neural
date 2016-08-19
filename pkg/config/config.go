package config

import (
	"fmt"
	"io/ioutil"
	"os"

	"gopkg.in/yaml.v1"
)

// Manifest is a data structure used to decode neural network configuration manifest
type Manifest struct {
	// Kind holds neural network Kind: feedfwd
	Kind string `yaml:"kind"`
	// Task is neural network task: class, [cluster, predict]
	Task string `yaml:"task"`
	// Network provides neural network layer config and topology
	Network struct {
		// Input layer configuration
		Input struct {
			// Size represents number of input neurons
			Size int `yaml:"size"`
		} `yaml:"input"`
		// Hidden layers configuration
		Hidden struct {
			// Size contains sizes of all hidden layers
			Size []int `yaml:"size"`
			// Activation is neuron activation function
			Activation string `yaml:"activation"`
		} `yaml:"hidden,omitempty"`
		// Output layer configuration
		Output struct {
			// Size represents number of input neurons
			Size int `yaml:"size"`
			// Activation is neuron activation function
			Activation string `yaml:"activation"`
		} `yaml:"output"`
	} `yaml:"network"`
	// Training holds neural network training configuration
	Training struct {
		// Kind holds kind of neural network training
		Kind string `yaml:"kind"`
		// Cost allows to specify cost function: xentropy, loglike
		Cost string `yaml:"cost"`
		// Params contains parameters of neural training
		Params struct {
			// Lambda is regualirzation parameter
			Lambda float64 `yaml:"lambda"`
		} `yaml:"params"`
		// Optimize contains configuration for training optimization
		Optimize struct {
			// Method represents type of optimization
			Method string `yaml:"method"`
			// Iterations is a number of major optimization iterations
			Iterations int `yaml:"iterations,omitempty"`
		} `yaml:"optimize,omitempty"`
	} `yaml:"training"`
}

// network maps supported training and optimization parameters to a particular neural network
var network = map[string]map[string][]string{
	"feedfwd": {
		"training": {"backprop"},
		"optim":    {"bfgs"},
	},
}

// NeuronConfig allows to specify neuron configuration
type NeuronConfig struct {
	// Activation is a neuron activation function
	Activation string
}

// LayerConfig allows to specify neural network layer configuration
type LayerConfig struct {
	// Kind is neural network layer kind: input, output, hidden
	Kind string
	// Size represents a number of neurons in the network layer
	Size int
	// NeurFn holds neuron configuration
	NeurFn *NeuronConfig
}

// NetArch specifies neural network architecture
type NetArch struct {
	// Input layer configuration
	Input *LayerConfig
	// Hidden layers configuration. It is a slice as there can be multiple hidden layers
	Hidden []*LayerConfig
	// Output layer configuration
	Output *LayerConfig
}

// NetConfig allows to specify Neural Network parameters
type NetConfig struct {
	// Kind is Neural Network type
	Kind string
	// Arch specifies network architecture
	Arch *NetArch
}

// OptimConfig allows to specify advanced optimization configuration
type OptimConfig struct {
	// Method is an advanced optimization method
	// Currently only bfgs algorithm is supported
	Method string
	// Iterations specifies the number of optimization iterations
	Iterations int
}

// TrainConfig allows to specify neural network training configuration
type TrainConfig struct {
	// Kind is a neural network training type: backprop
	Kind string
	// Cost is a neural network cost function
	Cost string
	// Lambda is regularizer parameter
	Lambda float64
	// Optimize holds training optimization parameters
	Optimize *OptimConfig
}

// Config allows to specify neural network architecture and training configuration
type Config struct {
	// Network holds neural network configuration
	Network *NetConfig
	// Training holds neural network training configuration
	Training *TrainConfig
}

// New returns neural network config struct based on the supplied manifest file.
// It accepts path to a config manifest file as a parameter. It returns error if the supplied
// manifest file can't be open or if it can not be parsed into a valid configration object.
func New(manPath string) (*Config, error) {
	var m Manifest
	// Open manifest file
	f, err := os.Open(manPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	// read the whole manifest file in one shot
	manData, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}
	// unmarshal the manifest data into Manifest struct
	if err := yaml.Unmarshal(manData, &m); err != nil {
		return nil, err
	}
	return ParseManifest(&m)
}

// ParseManifest parses the manifest supplied as a parameter into Config or fails with error
func ParseManifest(m *Manifest) (*Config, error) {
	// check if the network kind is not empty
	if m.Kind == "" {
		return nil, fmt.Errorf("Network kind can not be empty!\n")
	}
	// check if the requested network kind is supported
	if _, ok := network[m.Kind]; !ok {
		return nil, fmt.Errorf("Unsupported network kind: %s\n", m.Kind)
	}
	// parse neural network layer configuration parameters
	netConfig, err := parseNetConfig(m)
	if err != nil {
		return nil, err
	}
	// parse trainig configuration parameters
	trainConfig, err := parseTrainConfig(m)
	if err != nil {
		return nil, err
	}

	// return new network configuration
	return &Config{
		Network:  netConfig,
		Training: trainConfig,
	}, nil
}

func parseNetConfig(m *Manifest) (*NetConfig, error) {
	// INPUT layer configuration
	if m.Network.Input.Size <= 0 {
		return nil, fmt.Errorf("Incorrect input layer size: %d\n", m.Network.Input.Size)
	}
	inputLayer := &LayerConfig{Kind: "input", Size: m.Network.Input.Size}
	// HIDDEN network layer configuration
	var hiddenLayers []*LayerConfig
	if len(m.Network.Hidden.Size) != 0 {
		hiddenLayers = make([]*LayerConfig, len(m.Network.Hidden.Size))
		for i, size := range m.Network.Hidden.Size {
			if size <= 0 {
				return nil, fmt.Errorf("Incorrect hidden layer size: %d\n", size)
			}
			hiddenLayers[i] = &LayerConfig{
				Kind: "hidden",
				Size: size,
				NeurFn: &NeuronConfig{
					Activation: m.Network.Hidden.Activation,
				},
			}
		}
	}
	// OUTPUT layer configuration
	if m.Network.Output.Size <= 0 {
		return nil, fmt.Errorf("Incorrect output layer size: %d\n", m.Network.Output.Size)
	}
	outputLayer := &LayerConfig{
		Kind: "output",
		Size: m.Network.Output.Size,
		NeurFn: &NeuronConfig{
			Activation: m.Network.Output.Activation,
		},
	}

	return &NetConfig{
		Kind: m.Kind,
		Arch: &NetArch{
			Input:  inputLayer,
			Hidden: hiddenLayers,
			Output: outputLayer,
		},
	}, nil
}

func parseOptimConfig(m *Manifest) (*OptimConfig, error) {
	// optimize Method can't be empty
	if m.Training.Optimize.Method == "" {
		return nil, fmt.Errorf("Optimize method can not be empty!\n")
	}
	// check if the optimization method is supported
	var validOptim bool
	for _, optimizeMethod := range network[m.Kind]["optim"] {
		if optimizeMethod == m.Training.Optimize.Method {
			validOptim = true
			break
		}
	}
	if !validOptim {
		return nil, fmt.Errorf("Unsupported optimization method: %s\n",
			m.Training.Optimize.Method)
	}
	// check number of iterations
	var iters int
	if m.Training.Optimize.Iterations <= 0 {
		iters = 20
	} else {
		iters = m.Training.Optimize.Iterations
	}

	return &OptimConfig{
		Method:     m.Training.Optimize.Method,
		Iterations: iters,
	}, nil
}

func parseTrainConfig(m *Manifest) (*TrainConfig, error) {
	// training kind can't be empty
	if m.Training.Kind == "" {
		return nil, fmt.Errorf("Training kind can not be empty!\n")
	}
	// check if the requested training algorithm is supported
	var validTraining bool
	for _, trainingKind := range network[m.Kind]["training"] {
		if trainingKind == m.Training.Kind {
			validTraining = true
			break
		}
	}
	if !validTraining {
		return nil, fmt.Errorf("Unsupported training requested: %s\n", m.Training.Kind)
	}

	// check training cost function
	if m.Training.Cost == "" {
		return nil, fmt.Errorf("Cost function can not be empty!\n")
	}

	// check lambda parameter
	if m.Training.Params.Lambda < 0 {
		return nil, fmt.Errorf("Incorrect reg parameter: %f\n", m.Training.Params.Lambda)
	}

	// parse optimization config
	optimize, err := parseOptimConfig(m)
	if err != nil {
		return nil, err
	}

	// return train config
	return &TrainConfig{
		Kind:     m.Training.Kind,
		Cost:     m.Training.Cost,
		Lambda:   m.Training.Params.Lambda,
		Optimize: optimize,
	}, nil
}
