package config

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/milosgajdos83/go-neural/pkg/matrix"
	"gopkg.in/yaml.v1"
)

// Manifest is a data structure used to decode neural network configuration manifest
type Manifest struct {
	// Kind holds neural network Kind
	Kind string `yaml:"kind"`
	// Task is neural network task
	Task string `yaml:"task"`
	// Layers hold neural network layer config
	Layers struct {
		// Input layer configuration
		Input struct {
			// Size represents number of input neurons
			Size int `yaml:"size"`
		} `yaml:"input"`
		// Hidden layers configuration
		Hidden struct {
			// Size contains sizes of hidden layers
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
	} `yaml:"layers"`
	// Training holds neural network training configuration
	Training struct {
		// Kind holds kind of neural network training
		Kind string `yaml:"kind"`
		// Params contains parameters of neural training
		Params string `yaml:"params,omitempty"`
	} `yaml:"training"`
	// Optimize contains configuration for training optimization
	Optimize struct {
		// Method represents type of optimization
		Method string `yaml:"method"`
		// Iterations is a number of major optimization iterations
		Iterations int `yaml:"iterations,omitempty"`
	} `yaml:"optimize,omitempty"`
}

// supported allows to query which neural network types are supported for a particular network
// It also allows to query which optimization methods ara available for training particular network
var supported = map[string]map[string][]string{
	"feedfwd": map[string][]string{
		"training": []string{"backprop"},
		"optim":    []string{"bfgs"},
	},
}

// neurFunc maps activation function names to the activation implementations and to their gradients
var neurFunc = map[string]map[string]ActivationFn{
	"sigmoid": map[string]ActivationFn{
		"act":  matrix.SigmoidMx,
		"grad": matrix.SigmoidGradMx,
	},
	"softmax": map[string]ActivationFn{
		"act":  matrix.ExpMx,
		"grad": matrix.SigmoidGradMx,
	},
	"tanh": map[string]ActivationFn{
		"act":  matrix.TanhMx,
		"grad": matrix.TanhGradMx,
	},
	"relu": map[string]ActivationFn{
		"act":  matrix.ReluMx,
		"grad": matrix.ReluGradMx,
	},
}

// ActivationFn defines a neuron activation function
type ActivationFn func(int, int, float64) float64

// NeuronConfig holds neuron configuration
type NeuronConfig struct {
	meta string
	// ActFn is neuron activation function
	ActFn ActivationFn
	// ActGradFn is gradient of activation function
	// used in backpropagation algorithm
	ActGradFn ActivationFn
}

// Meta returns meta information about Neural Functions
func (n NeuronConfig) Meta() string {
	return n.meta
}

// LayerConfig contains layer configuration
type LayerConfig struct {
	// Kind is neural net layer kind
	Kind string
	// Size contains sizes of hidden layers
	Size int
	// NeurFn holds neuron activation functions
	NeurFn *NeuronConfig
}

// NetworkArch represents Neural Network architecture
type NetArch struct {
	// Input lauer configuration
	Input *LayerConfig
	// Hidden layer configuration
	Hidden []*LayerConfig
	// Output layer configuration
	Output *LayerConfig
}

// OptimConfig holds optimization configuration
type OptimConfig struct {
	// Method specifies optimization method
	Method string
	// Iterations specifies number of iterations
	Iterations int
}

// TrainConfig specifies neural net training configuration
type TrainConfig struct {
	// Kind defines a kind of neural net training
	Kind string
	// Params specifies training parameters
	Params string
	// Optimize specifies training optimization parameters
	Optimize *OptimConfig
}

// NetConfig specifies neural network configuration
type NetConfig struct {
	// Kind is Neural Network type
	Kind string
	// Arch holds neural network architecture
	Arch *NetArch
	// Training holds neural network training configuration
	Training *TrainConfig
}

// NewNetConfig returns neural network configuration or fails with error
func NewNetConfig(manPath string) (*NetConfig, error) {
	var m Manifest
	// Open manifest file
	f, err := os.Open(manPath)
	if err != nil {
		return nil, fmt.Errorf("Could not open manifest file: %s\n", err)
	}
	defer f.Close()
	// read the whole manifest file in one shot
	mData, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("Could not read manifest file: %s\n", err)
	}
	// unmarshal the manifest file
	if err := yaml.Unmarshal(mData, &m); err != nil {
		return nil, fmt.Errorf("Could not decode manifest file: %s\n", err)
	}
	return Parse(&m)
}

// Parse parses the manifest into NetConfig or fails with error
func Parse(m *Manifest) (*NetConfig, error) {
	// initialize dummy config
	c := &NetConfig{}
	c.Arch = &NetArch{}
	c.Training = &TrainConfig{}

	// check if the network kind is supported
	if m.Kind == "" {
		return nil, errors.New("Network kind parameter cant be empty!")
	}
	if _, ok := supported[m.Kind]; !ok {
		return nil, fmt.Errorf("Unsupported Neural Network type: %s\n", m.Kind)
	}
	c.Kind = m.Kind
	// parse neural network layers params
	if err := parseLayers(m, c); err != nil {
		return nil, err
	}
	// parse trainig parameters
	if err := parseTraining(m, c); err != nil {
		return nil, err
	}
	// parse optimization params
	if err := parseOptimize(m, c); err != nil {
		return nil, err
	}

	return c, nil
}

func parseLayers(m *Manifest, c *NetConfig) error {
	// INPUT layer configuration
	if m.Layers.Input.Size <= 0 {
		return fmt.Errorf("Incorrect size of input layer: %d\n", m.Layers.Input.Size)
	}
	// set the input size
	inSize := m.Layers.Input.Size
	inputLayer := &LayerConfig{Size: inSize, NeurFn: nil}
	c.Arch.Input = inputLayer
	c.Arch.Input.Kind = "input"
	// check feedfwd network architecture
	if m.Kind == "feedfwd" {
		// HIDDEN network layer configuration
		if len(m.Layers.Hidden.Size) != 0 {
			c.Arch.Hidden = make([]*LayerConfig, len(m.Layers.Hidden.Size))
			for i, size := range m.Layers.Hidden.Size {
				c.Arch.Hidden[i] = &LayerConfig{}
				c.Arch.Hidden[i].Kind = "hidden"
				c.Arch.Hidden[i].Size = size
				activation, ok := neurFunc[m.Layers.Hidden.Activation]
				if !ok {
					return fmt.Errorf("Unknown activation function: %s\n",
						m.Layers.Hidden.Activation)
				}
				c.Arch.Hidden[i].NeurFn = &NeuronConfig{
					meta:      m.Layers.Hidden.Activation,
					ActFn:     activation["act"],
					ActGradFn: activation["grad"],
				}
			}
		}
	}
	// OUTPUT layer configuration
	if m.Layers.Output.Size <= 0 {
		return fmt.Errorf("Incorrect output layer size: %d\n", m.Layers.Output.Size)
	}
	outSize := m.Layers.Output.Size
	outputLayer := &LayerConfig{Size: outSize}
	c.Arch.Output = outputLayer
	c.Arch.Output.Kind = "output"
	// check if the requested activation is supported
	activation, ok := neurFunc[m.Layers.Output.Activation]
	if !ok {
		return fmt.Errorf("Unknown activation function: %s\n",
			m.Layers.Output.Activation)
	}
	c.Arch.Output.NeurFn = &NeuronConfig{
		meta:      m.Layers.Output.Activation,
		ActFn:     activation["act"],
		ActGradFn: activation["grad"],
	}
	if m.Layers.Output.Activation == "tanh" {
		c.Arch.Output.NeurFn.ActFn = matrix.TanhOutMx
	}
	return nil
}

func parseTraining(m *Manifest, c *NetConfig) error {
	trainAlgs, ok := supported[m.Kind]["training"]
	if !ok {
		return fmt.Errorf("No training available for %s neural net.\n", m.Kind)
	}
	for i := range trainAlgs {
		if trainAlgs[i] == m.Training.Kind {
			c.Training.Kind = m.Training.Kind
			break
		}
	}
	if c.Training.Kind == "" {
		return fmt.Errorf("Unsupported training algorithm for %s network: %s\n",
			m.Kind, m.Training.Kind)
	}
	c.Training.Params = m.Training.Params
	return nil
}

func parseOptimize(m *Manifest, c *NetConfig) error {
	optimMethods, ok := supported[m.Kind]["optim"]
	if !ok {
		return fmt.Errorf("Unsupported optimization method: %s\n", m.Optimize.Method)
	}
	c.Training.Optimize = &OptimConfig{}
	for i := range optimMethods {
		if optimMethods[i] == m.Optimize.Method {
			c.Training.Optimize.Method = m.Optimize.Method
			break
		}
	}
	if c.Training.Optimize.Method == "" {
		return fmt.Errorf("Unsupported optimization method for %s network: %s\n",
			m.Kind, m.Optimize.Method)
	}
	if m.Optimize.Iterations <= 0 {
		c.Training.Optimize.Iterations = 20
	} else {
		c.Training.Optimize.Iterations = m.Optimize.Iterations
	}
	return nil
}
