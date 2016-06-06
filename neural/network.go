package neural

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/pkg/helpers"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
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

// ForwardProp performs the forward propagation for agiven input matrix.
// It recursively activates all layers in the network and returns the output in a matrix
// It fails with error if requested end layer index is out of all network layers
func (n *Network) ForwardProp(inMx mat64.Matrix, toLayer int) (mat64.Matrix, error) {
	if inMx == nil {
		return nil, fmt.Errorf("Can't forward propagate input: %v\n", inMx)
	}
	// get all the layers
	layers := n.Layers()
	// layer must exist
	if toLayer <= 0 || toLayer > len(layers)-1 {
		return nil, fmt.Errorf("Cant propagate beyond network layers: %d\n", len(layers))
	}
	// calculate the propagation
	return n.doForwardProp(inMx, 0, toLayer)
}

// doForwProp perform the actual forward propagation
func (n *Network) doForwardProp(inMx mat64.Matrix, from, to int) (mat64.Matrix, error) {
	// get all the layers
	layers := n.Layers()
	// we can't go backwards
	if from > to {
		return inMx, nil
	}
	// pick starting layer
	layer := layers[from]
	out, err := layer.Out(inMx)
	if err != nil {
		return nil, err
	}
	return n.doForwardProp(out, from+1, to)
}

// BackProp performs back propagation of neural network and updates each layer's delta matrix
// It traverses network recursively and calculates errors and updates deltas of each network layer.
// It fails with error if either supplied input and error matrices are nil or from boundary goes
// beyond the first network layer that can have errors calculated
func (n *Network) BackProp(inMx, errMx mat64.Matrix, fromLayer int) error {
	if inMx == nil {
		return fmt.Errorf("Can't backpropagate input: %v\n", inMx)
	}
	// can't BP empty error
	if errMx == nil {
		return fmt.Errorf("Can't backpropagate ouput error: %v\n", errMx)
	}
	// get all the layers
	layers := n.Layers()
	// can't backpropagate beyond the first hidden layer
	if fromLayer < 1 || fromLayer > len(layers)-1 {
		return fmt.Errorf("Cant backpropagate beyond first layer: %d\n", len(layers))
	}
	// perform the actual back propagation till the first hidden layer
	n.doBackProp(inMx, errMx, fromLayer, 1)
	return nil
}

// doBackProp performs the actual backpropagation
func (n *Network) doBackProp(inMx, errMx mat64.Matrix, from, to int) error {
	//fmt.Println("From", from)
	//r, c := inMx.Dims()
	//fmt.Printf("inMx: %d x %d\n", r, c)
	// get all the layers
	layers := n.Layers()
	// pick deltas layer
	deltasLayer := layers[from]
	bpDeltasMx := deltasLayer.Deltas()
	// pick weights layer
	weightsLayer := layers[from]
	bpWeightsMx := weightsLayer.Weights()
	// If we reach the 1st hidden layer we return
	if from == to {
		// add bias
		dMx := matrix.AddBias(inMx)
		dMx.Mul(errMx.T(), dMx)
		bpDeltasMx.Add(bpDeltasMx, dMx)
		return nil
	}
	// weights err
	weightsErrLayer := layers[from-1]
	weightsErrMx := weightsErrLayer.Weights()
	// forward propagate to from layer
	outMx, err := n.ForwardProp(inMx, from-1)
	if err != nil {
		return err
	}
	//r, c = outMx.Dims()
	//fmt.Printf("outMx: %d x %d\n", r, c)
	// add Bias unit
	biasOutMx := matrix.AddBias(outMx)
	//r, c = biasOutMx.Dims()
	//fmt.Printf("biasOutMx: %d x %d\n", r, c)
	dMx := new(mat64.Dense)
	dMx.Mul(errMx.T(), biasOutMx)
	//r, c = dMx.Dims()
	//fmt.Printf("dMx: %d x %d\n", r, c)
	//r, c = bpDeltasMx.Dims()
	//fmt.Printf("bpDeltasMx: %d x %d\n", r, c)
	bpDeltasMx.Add(bpDeltasMx, dMx)
	// errTmp holds layer error not accounting for bias
	errTmpMx := new(mat64.Dense)
	errTmpMx.Mul(bpWeightsMx.T(), errMx.T())
	r, c := errTmpMx.Dims()
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
	gradMx.MulElem(layeErr, gradMx)
	return n.doBackProp(inMx, gradMx, from-1, to)
}
