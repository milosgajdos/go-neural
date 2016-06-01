package neural

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/pkg/helpers"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
)

const (
	// INPUT is input network layer
	INPUT LayerKind = iota + 1
	// HIDDEN is hidden network layer
	HIDDEN
	// OUTPUT is output network layer
	OUTPUT
)

// LayerKind defines type of neural network layer
// There are three kinds available: INPUT, HIDDEN and OUTPUT
type LayerKind uint

// String implements Stringer interface for nice LayerKind printing
func (l LayerKind) String() string {
	switch l {
	case INPUT:
		return "INPUT"
	case HIDDEN:
		return "HIDDEN"
	case OUTPUT:
		return "OUTPUT"
	default:
		return "UNKNOWN"
	}
}

// ActivationFn defines a neuron activation function
type ActivationFn func(int, int, float64) float64

// NeuronFunc provides activation functions for forward and back propagation
type NeuronFunc struct {
	ForwFn ActivationFn
	BackFn ActivationFn
}

// Layer represents a Neural Network layer.
// Layer holds weights matrix and layer output deltas matrices internally
type Layer struct {
	// id is Layer unique identifier within network
	id string
	// kind is layer kind: input, hidden or output
	kind LayerKind
	// net is a neural network this layer is part off
	net *Network
	// weights matrix holds layer neuron weights per row
	weights *mat64.Dense
	// deltas matrix holds output deltas used for backprop
	deltas *mat64.Dense
	// activFunc contains various neuron activation functions
	activFunc *NeuronFunc
}

// NewLayer creates a new neural netowrk layer and returns it.
// Layer weights are initialized to uniformly distributed random values (-1,1)
// NewLayer fails with error if the supplied neural network does not exist.
func NewLayer(layerKind LayerKind, net *Network, layerIn, layerOut int) (*Layer, error) {
	if layerIn <= 0 || layerOut <= 0 {
		return nil, fmt.Errorf("Invalid layer size requested: %d, %d\n", layerIn, layerOut)
	}
	// Layer must belong to an existing Neural Network
	if net == nil {
		return nil, fmt.Errorf("Invalid neural network: %v\n", net)
	}
	// Layer kind must be valid
	if layerKind.String() == "UNKNOWN" {
		return nil, fmt.Errorf("Invalid layer kind requested: %s", layerKind)
	}
	layer := &Layer{}
	layer.id = helpers.PseudoRandString(10)
	layer.kind = layerKind
	layer.net = net
	// INPUT layer has neither weights matrix nor activation funcs
	if layerKind != INPUT {
		// initialize weights to random values
		var err error
		layer.weights, err = matrix.MakeRandMx(layerOut, layerIn+1, 0.0, 1.0)
		if err != nil {
			return nil, err
		}
		// initializes deltas to zero values
		layer.deltas = mat64.NewDense(layerOut, layerIn+1, nil)
		// TODO: parameterize activation functions
		layer.activFunc = &NeuronFunc{
			ForwFn: matrix.SigmoidMx,
			BackFn: matrix.SigmoidGradMx,
		}
	}
	return layer, nil
}

// ID returns layer id
func (l Layer) ID() string {
	return l.id
}

// Kind returns layer kind
func (l Layer) Kind() LayerKind {
	return l.kind
}

// Weights returns layer's eights matrix
func (l *Layer) Weights() *mat64.Dense {
	return l.weights
}

// SetWeights allows to set layer weights.
// It fails with error if the supplied weights have different dimension
// than the existing layer weights or if the passed in weights matrix is nil
// or if the layer is an INPUT layer: INPUT layer has no weights matrix.
func (l *Layer) SetWeights(w *mat64.Dense) error {
	// INPUT layer has no weights
	if l.kind == INPUT {
		return fmt.Errorf("Can't set weights matrix of %s layer\n", l.kind)
	}
	// we can't set weights to nil
	if w == nil {
		return fmt.Errorf("Network weights can't be nil")
	}
	// weights dimensions must stay the same
	wr, wc := w.Dims()
	lr, lc := l.weights.Dims()
	if wr != lr || wc != lc {
		return fmt.Errorf("Dimension mismatch. Current: %d x %d Supplied: %d x %d\n",
			lr, lc, wr, wc)
	}
	l.weights = w
	return nil
}

// Deltas returns layer's deltas matrix
func (l *Layer) Deltas() *mat64.Dense {
	return l.deltas
}

// Out calculates output of the network layer for the given input
// If the layer is INPUT layer, it returns the supplied input argument.
func (l *Layer) Out(inputMx mat64.Matrix) (*mat64.Dense, error) {
	// if it's INPUT layer, output is input
	if l.kind == INPUT {
		return inputMx.(*mat64.Dense), nil
	}
	// add bias to input
	biasInMx, err := matrix.AddBias(inputMx)
	if err != nil {
		return nil, err
	}
	// calculate activation function inputs
	out := new(mat64.Dense)
	out.Mul(biasInMx, l.weights.T())
	// activate layer neurons
	out.Apply(l.activFunc.ForwFn, out)
	return out, nil
}

// NeuronFunc returns the layer's NeuronFunc
func (l Layer) NeuronFunc() *NeuronFunc {
	return l.activFunc
}

// SetNeuronFunc allows to set the layer's NeuronFunc
func (l *Layer) SetNeuronFunc(nf *NeuronFunc) {
	l.activFunc = nf
}
