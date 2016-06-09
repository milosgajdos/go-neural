package neural

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/pkg/helpers"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
)

const (
	// FEEDFWD is a feed forward Neural Network
	FEEDFWD NetworkKind = iota + 1
)

// supported neural network types
var supported = map[NetworkKind]func(*Config) (*Network, error){
	FEEDFWD: createFeedFwdNetwork,
}

// NetworkKind defines a type of neural network
type NetworkKind uint

// String implements Stringer interface for pretty printing
func (n NetworkKind) String() string {
	switch n {
	case FEEDFWD:
		return "FEEDFWD"
	default:
		return "UNKNOWN"
	}
}

// NetworkArch represents Neural Network architecture
type NetworkArch struct {
	// Input layer size
	Input int
	// Hidden layers' sizes
	Hidden []int
	// Output layer size
	Output int
}

// Config provides Neural Network configuration
type Config struct {
	// Kind is Neural Network type
	Kind NetworkKind
	// Arch specifies Neural Network architecture
	Arch *NetworkArch
	// ActFunc specifies Neural Network activation functions
	ActFunc *NeuronFunc
}

// Network represents Neural Network
type Network struct {
	id     string
	kind   NetworkKind
	layers []*Layer
}

// NewNetwork creates new Neural Network based on the passed in parameters.
// It fails with error if either the unsupported network kind has been requested or
// if any of the neural network layers failed to be created. This can be due to
// incorrect network architecture i.e. mismatched neural layer dimensions.
func NewNetwork(c *Config) (*Network, error) {
	createNet, ok := supported[c.Kind]
	if !ok {
		return nil, fmt.Errorf("Unsupported Neural Network kind: %s\n", c.Kind)
	}
	// return network
	return createNet(c)
}

// createFeedFwdNetwork creates feedforward neural network or fails with error
func createFeedFwdNetwork(c *Config) (*Network, error) {
	// you must supply network architecture
	if c.Arch == nil {
		return nil, fmt.Errorf("Invalid network architecture supplied: %v\n", c.Arch)
	}
	net := &Network{}
	net.id = helpers.PseudoRandString(10)
	net.kind = c.Kind
	// Initialize INPUT layer: Input and Output layers are the same
	inLayer, err := NewLayer(INPUT, net, nil, c.Arch.Input, c.Arch.Input)
	if err != nil {
		return nil, err
	}
	net.layers = append(net.layers, inLayer)
	// layer input size set to INPUT as that's the first layer in to first HIDDEN layer
	layerInSize := c.Arch.Input
	// create HIDDEN layers
	for _, hiddenSize := range c.Arch.Hidden {
		layer, err := NewLayer(HIDDEN, net, c.ActFunc, layerInSize, hiddenSize)
		if err != nil {
			return nil, err
		}
		net.layers = append(net.layers, layer)
		// layerInSize is set to output of the previous layer
		layerInSize = hiddenSize
	}
	// Create OUTPUT layer
	outLayer, err := NewLayer(OUTPUT, net, c.ActFunc, layerInSize, c.Arch.Output)
	if err != nil {
		return nil, err
	}
	net.layers = append(net.layers, outLayer)
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

// ForwardProp performs forward propagation for a given input up to a specified network layer.
// It recursively activates all layers in the network and returns the output in a matrix
// It fails with error if requested end layer index is beyond all available layers or if
// the supplied input data is nil.
func (n *Network) ForwardProp(inMx mat64.Matrix, toLayer int) (mat64.Matrix, error) {
	if inMx == nil {
		return nil, fmt.Errorf("Can't forward propagate input: %v\n", inMx)
	}
	// get all the layers
	layers := n.Layers()
	// layer must exist
	if toLayer < 0 || toLayer > len(layers)-1 {
		return nil, fmt.Errorf("Cant propagate beyond network layers: %d\n", len(layers))
	}
	// calculate the propagation
	return n.doForwardProp(inMx, 0, toLayer)
}

// doForwProp perform the actual forward propagation
func (n *Network) doForwardProp(inMx mat64.Matrix, from, to int) (mat64.Matrix, error) {
	// get all the layers
	layers := n.Layers()
	// pick starting layer
	layer := layers[from]
	// we can't go backwards
	if from == to {
		return layer.Out(inMx)
	}
	out, err := layer.Out(inMx)
	if err != nil {
		return nil, err
	}
	return n.doForwardProp(out, from+1, to)
}

// BackProp performs back propagation of neural network. It traverses neural network recursively
// and updates deltas of each network layer based on the layer error minimizin the network's objective func.
// It fails with error if either the supplied input and delta matrices are nil or f the specified
// from boundary goes beyond the first network layer that can have output errors calculated
func (n *Network) BackProp(inMx, deltaMx mat64.Matrix, fromLayer int) error {
	if inMx == nil {
		return fmt.Errorf("Can't backpropagate input: %v\n", inMx)
	}
	// can't BP empty error
	if deltaMx == nil {
		return fmt.Errorf("Can't backpropagate ouput error: %v\n", deltaMx)
	}
	// get all the layers
	layers := n.Layers()
	// can't backpropagate beyond the first hidden layer
	if fromLayer < 1 || fromLayer > len(layers)-1 {
		return fmt.Errorf("Cant backpropagate beyond first layer: %d\n", len(layers))
	}
	// perform the actual back propagation till the first hidden layer
	n.doBackProp(inMx, deltaMx, fromLayer, 1)
	return nil
}

// doBackProp performs the actual backpropagation
func (n *Network) doBackProp(inMx, deltaMx mat64.Matrix, from, to int) error {
	// get all the layers
	layers := n.Layers()
	// pick deltas layer
	deltasLayer := layers[from]
	bpDeltasMx := deltasLayer.Deltas()
	// If we reach the 1st hidden layer we return
	if from == to {
		outMx, err := n.ForwardProp(inMx, from-1)
		if err != nil {
			return err
		}
		outMxBias := matrix.AddBias(outMx)
		dMx := new(mat64.Dense)
		dMx.Mul(deltaMx.T(), outMxBias)
		// update deltas
		bpDeltasMx.Add(bpDeltasMx, dMx)
		return nil
	}
	// pick weights layer
	weightsLayer := layers[from]
	bpWeightsMx := weightsLayer.Weights()
	// pick errLayer
	weightsErrLayer := layers[from-1]
	weightsErrMx := weightsErrLayer.Weights()
	// forward propagate to from layer
	outMx, err := n.ForwardProp(inMx, from-1)
	if err != nil {
		return err
	}
	// add Bias unit
	biasOutMx := matrix.AddBias(outMx)
	dMx := new(mat64.Dense)
	dMx.Mul(deltaMx.T(), biasOutMx)
	bpDeltasMx.Add(bpDeltasMx, dMx)
	// errTmp holds layer error not accounting for bias
	errTmpMx := new(mat64.Dense)
	errTmpMx.Mul(bpWeightsMx.T(), deltaMx.T())
	r, c := errTmpMx.Dims()
	// avoid bias
	layeErr := errTmpMx.View(1, 0, r-1, c).(*mat64.Dense)
	// pre-activation unit
	actInMx, err := n.ForwardProp(inMx, from-2)
	if err != nil {
		return err
	}
	biasActInMx := matrix.AddBias(actInMx)
	gradMx := new(mat64.Dense)
	gradMx.Mul(biasActInMx, weightsErrMx.T())
	gradMx.Apply(weightsErrLayer.NeuronFunc().BackFn, gradMx)
	gradMx.MulElem(layeErr.T(), gradMx)
	return n.doBackProp(inMx, gradMx, from-1, to)
}

// Classify classifies the provided data vector to a particular label class.
// It returns a matrix that contains probabilities of the input belonging to a particular class
// It returns error if the network forward propagation fails at any point during classification.
func (n *Network) Classify(inMx mat64.Matrix) (mat64.Matrix, error) {
	if inMx == nil {
		return nil, fmt.Errorf("Can't classify %v\n", inMx)
	}
	// do forward propagation
	out, err := n.ForwardProp(inMx, len(n.Layers())-1)
	if err != nil {
		return nil, err
	}
	samples, _ := inMx.Dims()
	_, results := out.Dims()
	// classification matrix
	classMx := mat64.NewDense(samples, results, nil)
	switch o := out.(type) {
	case *mat64.Dense:
		for i := 0; i < samples; i++ {
			row := new(mat64.Dense)
			row.Clone(o.RowView(i))
			sum := mat64.Sum(row)
			row.Scale(100.0/sum, row)
			data := matrix.Mx2Vec(row, true)
			classMx.SetRow(i, data)
		}
	case *mat64.Vector:
		sum := mat64.Sum(o)
		tmp := new(mat64.Dense)
		tmp.Scale(100.0/sum, o)
		data := matrix.Mx2Vec(tmp, true)
		classMx.SetRow(0, data)
	}
	return classMx, nil
}

// Validate runs forward propagation on the validation data set through neural network.
// It returns the percentage of successful classifications or error.
func (n *Network) Validate(valInMx *mat64.Dense, valOut *mat64.Vector) (float64, error) {
	// validation set can't be nil
	if valInMx == nil || valOut == nil {
		return 0.0, fmt.Errorf("Cant validate data set. In: %v, Out: %v\n", valInMx, valOut)
	}
	out, err := n.ForwardProp(valInMx, len(n.Layers())-1)
	if err != nil {
		return 0.0, err
	}
	rows, _ := out.Dims()
	outMx := out.(*mat64.Dense)
	hits := 0.0
	for i := 0; i < rows; i++ {
		row := outMx.RowView(i)
		max := mat64.Max(row)
		for j := 0; j < row.Len(); j++ {
			if row.At(j, 0) == max {
				if j+1 == int(valOut.At(i, 0)) {
					hits++
					break
				}
			}
		}
	}
	success := (hits / float64(valOut.Len())) * 100
	return success, nil
}
