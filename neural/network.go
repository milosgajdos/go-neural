package neural

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/pkg/helpers"
)

const (
	// FEEDFWD is a feed forwardf Neural Network
	FEEDFWD NetworkKind = iota + 1
)

// NetworkKind defines a type of neural network
type NetworkKind uint

// String implements Stringer interface for pretty printing
func (nk NetworkKind) String() string {
	switch nk {
	case FEEDFWD:
		return "FEEDFWD"
	default:
		return "UNKNOWN"
	}
}

// NetworkArch allows to specify network architecture to be created
type NetworkArch struct {
	// Input layer size
	Input int
	// Hidden layers' sizes
	Hidden []int
	// Output layer size
	Output int
}

// Network represents a certain kind of Neural Network.
// It has an id and can have arbitrary number of layers.
type Network struct {
	id     string
	kind   NetworkKind
	layers []*Layer
}

// NewNetwork creates new neural network and returns it
// It accepts two parameters: netKind which represents what kind of neural network layer you want
// to create and layerSizes which specify number of neurons in each layer.
// It fails with error if either the unsupported network kind has been requested or
// if any of the network layers could not be created
func NewNetwork(netKind NetworkKind, netArch *NetworkArch) (*Network, error) {
	// if network kind is unknown return error
	if netKind.String() == "UNKNOWN" {
		return nil, fmt.Errorf("Unsupported Neural Network kind: %s\n", netKind)
	}
	// you must supply network architecture
	if netArch == nil {
		return nil, fmt.Errorf("Invalid network architecture supplied: %v\n", netArch)
	}
	net := &Network{}
	net.id = helpers.PseudoRandString(10)
	net.kind = netKind
	// Initialize INPUT layer: Input and Output layers are the same
	inLayer, err := NewLayer(INPUT, net, netArch.Input, netArch.Input)
	if err != nil {
		return nil, err
	}
	net.layers = append(net.layers, inLayer)
	// layer input size set to INPUT as that's the first layer in to first HIDDEN layer
	layerInSize := netArch.Input
	// create HIDDEN layers
	for _, hiddenSize := range netArch.Hidden {
		layer, err := NewLayer(HIDDEN, net, layerInSize, hiddenSize)
		if err != nil {
			return nil, err
		}
		net.layers = append(net.layers, layer)
		// layerInSize is set to output of the previous layer
		layerInSize = hiddenSize
	}
	// Create OUTPUT layer
	outLayer, err := NewLayer(OUTPUT, net, layerInSize, netArch.Output)
	if err != nil {
		return nil, err
	}
	net.layers = append(net.layers, outLayer)
	// return network
	return net, nil
}

// ID returns neural network id
func (n Network) ID() string {
	return n.id
}

// Kind returns kind of neural network
func (n Network) Kind() NetworkKind {
	return n.kind
}

// Layers returns network layers in slice sorted from INPUT to OUTPUT layer
func (n Network) Layers() []*Layer {
	return n.layers
}

// ForwardProp calculates the result of forward propagation for agiven input matrix.
// It recursively activates all layers in the network and returns the output in a matrix
// It fails with error if requested end layer index is out of all network layers
// It panics if any of the network layer outputs duroing propagation fails to be computed.
func (n *Network) ForwardProp(inMx mat64.Matrix, toLayer int) (mat64.Matrix, error) {
	if inMx == nil {
		return nil, fmt.Errorf("Can't forward propage input: %v\n", inMx)
	}
	// get all the layers
	layers := n.Layers()
	// layer must exist
	if toLayer <= 0 || toLayer > len(layers)-1 {
		return nil, fmt.Errorf("Cant propagate beyond network layers: %d\n", len(layers))
	}
	// calculate the propagation
	out, _, _ := n.doForwPropagation(inMx, 0, toLayer)
	// return the output
	return out, nil
}

// doForwPropagation perform the actual forward propagation
func (n *Network) doForwPropagation(inMx mat64.Matrix, start, end int) (mat64.Matrix, int, int) {
	// get all the layers
	layers := n.Layers()
	// we can't go backwards
	if start > end {
		return inMx, start, end
	}
	// pick starting layer
	layer := layers[start]
	out, err := layer.Out(inMx)
	if err != nil {
		panic(err)
	}
	return n.doForwPropagation(out, start+1, end)
}
