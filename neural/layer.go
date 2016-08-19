package neural

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/pkg/config"
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

// ActivFunc defines a neuron activation function
type ActivFunc func(int, int, float64) float64

// activations maps activation function names to their actual implementations
var activations = map[string]map[string]ActivFunc{
	"sigmoid": {
		"act":  matrix.SigmoidMx,
		"grad": matrix.SigmoidGradMx,
	},
	"softmax": {
		"act":  matrix.ExpMx,
		"grad": matrix.SigmoidGradMx,
	},
	"tanh": {
		"act":  matrix.TanhMx,
		"grad": matrix.TanhGradMx,
	},
	"relu": {
		"act":  matrix.ReluMx,
		"grad": matrix.ReluGradMx,
	},
}

// layerKind maps string representations to LayerKind
var layerKind = map[string]LayerKind{
	"input":  INPUT,
	"hidden": HIDDEN,
	"output": OUTPUT,
}

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

// Layer represents a Neural Network layer.
type Layer struct {
	// id is Layer unique identifier within network
	id string
	// kind is layer kind: input, hidden or output
	kind LayerKind
	// weights matrix holds layer neuron weights per row
	weights *mat64.Dense
	// deltas matrix holds output deltas used for backprop
	deltas *mat64.Dense
	// act is neuron activation function
	act ActivFunc
	// actGrad is derivation of neuron activation function
	actGrad ActivFunc
	// meta contains layer metadata: currently only info about OUT ActFn
	meta string
}

// NewLayer creates a new neural network layer and returns it.
// Layer weights are initialized to uniformly distributed random values (-1,1)
// NewLayer fails with error if the neural network supplied as a parameter does not exist.
func NewLayer(c *config.LayerConfig, layerIn int) (*Layer, error) {
	// layer in must be positive integer
	if layerIn <= 0 {
		return nil, fmt.Errorf("Layer input must be positive integer: %d\n", layerIn)
	}
	// layer size must be positive integer
	if c.Size <= 0 {
		return nil, fmt.Errorf("Layer size must be positive integer: %d\n", c.Size)
	}
	// Layer kind must be valid
	if _, ok := layerKind[c.Kind]; !ok {
		return nil, fmt.Errorf("Invalid layer kind requested: %s", c.Kind)
	}
	layer := &Layer{}
	layer.id = helpers.PseudoRandString(10)
	layer.kind = layerKind[c.Kind]
	// INPUT layer has neither weights matrix nor activation funcs
	if layer.kind != INPUT {
		// Set activation function
		activFunc, ok := activations[c.NeurFn.Activation]
		if !ok {
			return nil, fmt.Errorf("Unsupported activation function: %s\n",
				c.NeurFn.Activation)
		}
		// set activation functions
		layer.act = activFunc["act"]
		// if tanh - needs to be rescaled if used in OUTPUT layer
		if c.NeurFn.Activation == "tanh" {
			if layer.kind == OUTPUT {
				layer.act = matrix.TanhOutMx
			}
		}

		layer.actGrad = activFunc["grad"]
		layer.meta = c.NeurFn.Activation
		layerOut := c.Size
		// initialize weights to random values
		var err error
		layer.weights, err = matrix.MakeRandMx(layerOut, layerIn+1, 0.0, 1.0)
		if err != nil {
			return nil, err
		}
		// initializes deltas to zero values
		layer.deltas = mat64.NewDense(layerOut, layerIn+1, nil)
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

// SetWeights allows to set neural network layer weights.
// It fails with error if either the supplied weights have different dimensions
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
	// We must re-allocate deltas too
	deltas := mat64.NewDense(wr, wc, nil)
	l.deltas = deltas
	return nil
}

// Deltas returns layer's output deltas matrix
// Deltas matrix is initialized to zeros and is only non-zero if the back propagation
// algorithm has been run.
func (l *Layer) Deltas() *mat64.Dense {
	return l.deltas
}

// FwdOut calculates forward output of the network layer for given input.
// If the layer is an INPUT layer, it returns the matrix supplied as an argument.
func (l *Layer) FwdOut(inputMx mat64.Matrix) (mat64.Matrix, error) {
	// if input is nil, return error
	if inputMx == nil {
		return nil, fmt.Errorf("Cant calculate output for: %v\n", inputMx)
	}
	// if it's INPUT layer, output is input
	if l.kind == INPUT {
		return inputMx, nil
	}
	// input column dimensions + bias must match the weights column dimensions
	inRows, inCols := inputMx.Dims()
	_, wCols := l.weights.Dims()
	if inCols+1 != wCols {
		return nil, fmt.Errorf("Dimension mismatch. Weight: %d, Input: %d\n", wCols, inCols)
	}
	// add bias to input
	biasInMx := matrix.AddBias(inputMx)
	// calculate activation function inputs
	out := new(mat64.Dense)
	out.Mul(biasInMx, l.weights.T())
	// activate layer neurons
	out.Apply(l.act, out)
	if l.meta == "softmax" {
		rowSums := matrix.RowSums(out)
		for i := 0; i < inRows; i++ {
			rowVec := out.RowView(i)
			rowVec.ScaleVec(1/rowSums[i], rowVec)
			out.SetRow(i, rowVec.RawVector().Data)
		}
	}
	return out, nil
}

// ActFn returns layer activation function
func (l Layer) ActFn() func(int, int, float64) float64 {
	return l.act
}

// ActGrad returns layer gradient activation function
func (l Layer) ActGrad() func(int, int, float64) float64 {
	return l.actGrad
}
