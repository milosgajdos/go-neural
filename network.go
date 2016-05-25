package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

const (
	// Kind of Neural Network
	FEEDFWD NetworkKind = iota + 1
)

const (
	// Kind of network layers
	INPUT LayerKind = iota + 1
	HIDDEN
	OUTPUT
)

// randomString generates r pseudoandom string of specified size
func randomString(size int) string {
	rand.Seed(55)
	alphanum := "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	bytes := make([]byte, size)
	rand.Read(bytes)
	// iterate through all alphanum bytes
	for i, b := range bytes {
		bytes[i] = alphanum[b%byte(len(alphanum))]
	}
	return string(bytes)
}

// addBias adds a bias vector to matrix x and returns the new matrix
func addBias(x *mat64.Dense) *mat64.Dense {
	xRows, xCols := x.Dims()
	// Initiate bias vector with 1.0s
	ones := make([]float64, xRows)
	for i, _ := range ones {
		ones[i] = 1.0
	}
	// create bias vector
	biasVec := mat64.NewVector(xRows, ones)
	xMx := mat64.NewDense(xRows, xCols+1, nil)
	// Add bias to input data
	xMx.Augment(biasVec, x)
	return xMx
}

// ones returns a matrix of rows x cols filled with 1.0
func ones(rows, cols int) *mat64.Dense {
	onesMx := mat64.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			onesMx.Set(i, j, 1.0)
		}
	}
	return onesMx
}

// Kind of a Network Layer
type NetworkKind uint

// String implements Stringer interface
func (lk NetworkKind) String() string {
	switch lk {
	case FEEDFWD:
		return "FEEDFWD"
	default:
		return "UNKNOWN"
	}
}

// Network represents Neural Network
// It has id and can have arbitrary number of layers
type Network struct {
	id     string
	kind   NetworkKind
	layers []*Layer
}

// NewNetwork creates new neural network and returns it
// It fails with errorif the network could not be created
func NewNetwork(netKind NetworkKind, layers []uint) (*Network, error) {
	if len(layers) < 2 {
		return nil, errors.New("Neural network must have at least 2 layers")
	}
	net := &Network{}
	net.id = randomString(10)
	net.kind = netKind
	// layer input size
	var layerIn uint
	var layerKind LayerKind
	// Initialize every neural net layer
	for id, layerOut := range layers {
		// default layer is HIDDEN
		layerKind = HIDDEN
		// input layer size
		if id == 0 {
			layerKind = INPUT
			layerIn = layerOut
		}
		// output layer
		if id == len(layers)-1 {
			layerKind = OUTPUT
		}
		layer, err := NewLayer(uint(id), layerKind, net, uint(layerIn), uint(layerOut))
		if err != nil {
			return nil, err
		}
		net.layers = append(net.layers, layer)
		layerIn = layerOut
	}
	return net, nil
}

// Train runs Neural Network training for given training data X and labels y
// It returns a precision percentage on the training data or error
// TODO: config to specify kind of training etc.
func (n *Network) Train(x *mat64.Dense, y *mat64.Vector,
	labels uint, lambda float64, iters uint) (float64, error) {
	// number of data samples
	dataLen, _ := x.Dims()
	// run forward propagation
	outMx, _ := n.forwardProp(x, 1)
	// each row represents the expected (labeled) result
	labelsMx := mat64.NewDense(dataLen, int(labels), nil)
	for i := 0; i < y.Len(); i++ {
		val := y.At(i, 0)
		labelsMx.Set(i, int(val)-1, 1.0)
	}
	// J = -(sum(sum((Y_k .* log(a3) + (1 - Y_k) .* log(1 - a3)), 2)))/m;
	logOutMx := new(mat64.Dense)
	// log(outMx)
	logOutMx.Apply(LogMx, outMx)
	mulMxA := new(mat64.Dense)
	// y*log(outMx)
	mulMxA.MulElem(labelsMx, logOutMx)
	// 1 - y
	labelsMx.Apply(SubtrMx, labelsMx)
	// 1 - outMx
	outMx.Apply(SubtrMx, outMx)
	// log(1-outMx)
	outMx.Apply(LogMx, outMx)
	mulMxB := new(mat64.Dense)
	// (1 - y) * log(1-outMx)
	mulMxB.MulElem(labelsMx, outMx)
	// y*log(outMx) + (1 - y)*log(1-outMx)
	mulMxB.Add(mulMxA, mulMxB)
	cost := -(mat64.Sum(mulMxB) / float64(dataLen))
	fmt.Printf("Non-Reg Cost: \n%f\n", cost)
	// calculate regularizer
	if lambda > 0 {
		layers := n.Layers()
		// Ignore first layer i.e. input layer
		for _, layer := range layers[1:] {
			r, c := layer.weightMx.Dims()
			// Don't penalize bias
			wViewMx := layer.weightMx.View(0, 1, r, c-1)
			powMx := new(mat64.Dense)
			powMx.Apply(PowerMx, wViewMx)
			cost += (lambda / (2 * float64(dataLen))) / mat64.Sum(powMx)
		}
	}
	return cost, nil
}

func (Network) costFunc(x []float64) float64 {
	return 0.0
}

func (Network) gradFunc(grad, x []float64) {
}

// feedForward recursively progresses Neural Network
func (n *Network) forwardProp(inMx *mat64.Dense, layerIdx int) (*mat64.Dense, int) {
	if layerIdx > len(n.layers)-1 {
		return inMx, layerIdx
	}
	// add bias to data matrix
	biasInMx := addBias(inMx)
	// compute activations
	weightMx := new(mat64.Dense)
	layer := n.layers[layerIdx]
	weightMx.Mul(biasInMx, layer.weightMx.T())
	activFunc := func(i, j int, x float64) float64 {
		return layer.actFuncs.ForwFn(x)
	}
	weightMx.Apply(activFunc, weightMx)
	layerIdx += 1
	return n.forwardProp(weightMx, layerIdx)
}

// Predict classifies the provided data vector to particular label
// It returns the label number or error
func (n *Network) Predict(x *mat64.Vector) (int, error) {
	return 0, nil
}

// Validate runs Neural Net validation through the provided training set and returns
// percentage of successful classifications or error
func (n *Network) Validate(x *mat64.Dense) (float64, error) {
	return 0.0, nil
}

// Layers returns a slice of netowrk layers
func (n Network) Layers() []*Layer {
	return n.layers
}

// Id returns network id
func (n Network) Id() string {
	return n.id
}

// Kind returns network kind
func (n Network) Kind() NetworkKind {
	return n.kind
}

// Kind of a Network Layer
type LayerKind uint

// String implements Stringer interface
func (lk LayerKind) String() string {
	switch lk {
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

// Layer represents a Neural Network layer
// It has id and a list of Neurons. It has a matrix of neuron weights
// Layer can be of three different kinds: input, hidden or output
type Layer struct {
	id       uint
	kind     LayerKind
	net      *Network
	weightMx *mat64.Dense
	actFuncs *NeuronFunc
}

// NewLayer creates new neural netowrk layer and returns it
func NewLayer(id uint, layerKind LayerKind, net *Network, layerIn, layerOut uint) (*Layer, error) {
	// set random seed
	rand.Seed(55)
	layer := &Layer{}
	layer.id = id
	layer.kind = layerKind
	layer.net = net
	// INPUT layer does not have weights matrix nor activation funcs
	// empirically this is supposed to be the best value
	max := 1.0
	min := 0.0
	epsilon := math.Sqrt(6.0) / math.Sqrt(float64(layerIn+layerOut))
	if layerKind != INPUT {
		weights := make([]float64, layerOut*(layerIn+1))
		for i := range weights {
			// we need value between 0 and 1.0
			weights[i] = rand.Float64()*(max-min) + min
			weights[i] = weights[i]*(2*epsilon) - epsilon
		}
		layer.weightMx = mat64.NewDense(int(layerOut), int(layerIn+1), weights)
		// TODO: parameterize activation functions
		layer.actFuncs = &NeuronFunc{
			ForwFn: Sigmoid,
			BackFn: SigmoidGrad,
		}
	}
	return layer, nil
}

// Id returns network id
func (l Layer) Id() uint {
	return l.id
}

// Kind returns network kind
func (l Layer) Kind() LayerKind {
	return l.kind
}

// Weights returns layer's weights matrix
func (l *Layer) Weights() *mat64.Dense {
	return l.weightMx
}

// SetWeights allows to set the layer's weights matrix
func (l *Layer) SetWeights(weightMx *mat64.Dense) error {
	if l.kind == INPUT {
		return errors.New("INPUT layer does not have weights matrix")
	}
	// TODO: should we Copy/Clone rather than plain assign?
	l.weightMx = weightMx
	return nil
}

// ActivationFn represents a Neuron activation function
// It accepts a vector of float numbers and returns a single value
type ActivationFn func(float64) float64

// NeuronFunc provides activation functions for forward and backprop
type NeuronFunc struct {
	ForwFn ActivationFn
	BackFn ActivationFn
}
