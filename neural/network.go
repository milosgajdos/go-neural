package neural

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
	"github.com/milosgajdos83/go-neural/pkg/config"
	"github.com/milosgajdos83/go-neural/pkg/helpers"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
)

const (
	// FEEDFWD is a feed forward Neural Network
	FEEDFWD NetworkKind = iota + 1
)

// optim maps optimization algorithm names to their actual implementations
var optim = map[string]optimize.Method{
	"bfgs": &optimize.BFGS{},
}

// kindMap maps strings to NetworkKind
var netKind = map[string]NetworkKind{
	"feedfwd": FEEDFWD,
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

// network maps supported neural network types to their constructors
var network = map[string]func(*config.NetArch) (*Network, error){
	"feedfwd": createFeedFwdNetwork,
}

// Network represents Neural Network
type Network struct {
	id     string
	kind   NetworkKind
	layers []*Layer
}

// NewNetwork creates new Neural Network based on the passed in configuration parameters.
// It fails with error if either the requested network type is not supported or
// if any of the neural network layers failed to be created.
func NewNetwork(c *config.NetConfig) (*Network, error) {
	if c == nil {
		return nil, fmt.Errorf("Invalid network configuration supplied: %v\n", c)
	}
	// check if the requested network is supported and retrieve its constructor
	createNet, ok := network[c.Kind]
	if !ok {
		return nil, fmt.Errorf("Unsupported neural network type: %s\n", c.Kind)
	}
	// return network
	return createNet(c.Arch)
}

// createFeedFwdNetwork creates feedforward neural network or fails with error
func createFeedFwdNetwork(arch *config.NetArch) (*Network, error) {
	// check if the supplied architecture is not nil
	if arch == nil {
		return nil, fmt.Errorf("Incorrect architecture supplied: %v\n", arch)
	}
	// create new network
	net := &Network{}
	net.id = helpers.PseudoRandString(10)
	net.kind = FEEDFWD
	// Create INPUT layer: INPUT layer has no activation function
	layerInSize := arch.Input.Size
	inLayer, err := NewLayer(arch.Input, arch.Input.Size)
	if err != nil {
		return nil, err
	}
	// add neural net layer to network
	if err := net.AddLayer(inLayer); err != nil {
		return nil, err
	}
	// create HIDDEN layers
	for _, layerConfig := range arch.Hidden {
		layer, err := NewLayer(layerConfig, layerInSize)
		if err != nil {
			return nil, err
		}
		// add neural net layer
		if err := net.AddLayer(layer); err != nil {
			return nil, err
		}
		// layerInSize is set to output of the previous layer
		layerInSize = layerConfig.Size
	}
	// Create OUTPUT layer
	outLayer, err := NewLayer(arch.Output, layerInSize)
	if err != nil {
		return nil, err
	}
	// add neural net layer
	if err := net.AddLayer(outLayer); err != nil {
		return nil, err
	}
	return net, nil
}

// AddLayer adds layer to neural network or fails with error
// AddLayer places restrictions on adding new layers:
// 1. INPUT layer  - there must only be one INPUT layer
// 2. HIDDEN layer - new HIDDEN layer is appened after the last HIDDEN layer
// 3. OUTPUT layer - there must only be one OUTPUT layer
// AddLayer fails with error if either 1. or 3. are not satisfied
// TODO: simplify this madness
func (n *Network) AddLayer(layer *Layer) error {
	layerCount := len(n.layers)
	// if no layer exists yet, just append
	if layerCount == 0 {
		n.layers = append(n.layers, layer)
	}
	// if one layer already exists it depends on which one we are adding
	if layerCount == 1 {
		switch n.layers[0].Kind() {
		case INPUT:
			if layer.Kind() == INPUT {
				return fmt.Errorf("Can't create multiple INPUT layers\n")
			}
			n.layers = append(n.layers, layer)
		case OUTPUT:
			if layer.Kind() == OUTPUT {
				return fmt.Errorf("Can't create multiple OUTPUT layers\n")
			}
			n.layers = append(n.layers, layer)
		default:
			n.layers = append(n.layers, layer)
		}
	}
	if layerCount > 1 {
		switch layer.Kind() {
		case INPUT:
			if n.layers[0].Kind() == INPUT {
				return fmt.Errorf("Can't create multiple INPUT layers\n")
			}
			// Prepend - i.e. place INPUT at the first position
			n.layers = append([]*Layer{layer}, n.layers...)
		case OUTPUT:
			if n.layers[layerCount-1].Kind() == OUTPUT {
				return fmt.Errorf("Can't create multiple OUTPUT layers\n")
			}
			// append at the end
			n.layers = append(n.layers, layer)
		case HIDDEN:
			// find last hidden layer and append afterwards
			var lastHidden int
			for i, l := range n.layers {
				if l.Kind() == HIDDEN {
					lastHidden = i
				}
			}
			// expand capacity
			n.layers = append(n.layers, nil)
			copy(n.layers[lastHidden+2:], n.layers[lastHidden+1:])
			n.layers[lastHidden+1] = layer
		}
	}
	return nil
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
		return layer.FwdOut(inMx)
	}
	out, err := layer.FwdOut(inMx)
	if err != nil {
		return nil, err
	}
	return n.doForwardProp(out, from+1, to)
}

// BackProp performs back propagation of neural network. It traverses neural network recursively
// from layer specified via parameter and calculates error deltas for each network layer.
// It fails with error if either the supplied input and delta matrices are nil or if the specified
// from boundary goes beyond the first network layer that can have output errors calculated
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
	return n.doBackProp(inMx, errMx, fromLayer, 1)
}

// doBackProp performs the actual backpropagation
func (n *Network) doBackProp(inMx, errMx mat64.Matrix, from, to int) error {
	// get all the layers
	layers := n.Layers()
	// pick deltas layer
	layer := layers[from]
	deltasMx := layer.Deltas()
	weightsMx := layer.Weights()
	//forward propagate to previous layer
	outMx, err := n.ForwardProp(inMx, from-1)
	if err != nil {
		return err
	}
	outMxBias := matrix.AddBias(outMx)
	// compute deltas update
	dMx := new(mat64.Dense)
	dMx.Mul(errMx.T(), outMxBias)
	// update deltas
	deltasMx.Add(deltasMx, dMx)
	// If we reach the 1st hidden layer we return
	if from == to {
		return nil
	}
	// errTmpMx holds layer error not accounting for bias
	errTmpMx := new(mat64.Dense)
	errTmpMx.Mul(weightsMx.T(), errMx.T())
	r, c := errTmpMx.Dims()
	// avoid bias
	layerErr := errTmpMx.View(1, 0, r-1, c).(*mat64.Dense)
	// pre-activation unit
	actInMx, err := n.ForwardProp(inMx, from-2)
	if err != nil {
		return err
	}
	biasActInMx := matrix.AddBias(actInMx)
	// pick errLayer
	weightsErrLayer := layers[from-1]
	weightsErrMx := weightsErrLayer.Weights()
	// compute gradient matrix
	gradMx := new(mat64.Dense)
	gradMx.Mul(biasActInMx, weightsErrMx.T())
	gradMx.Apply(weightsErrLayer.ActGrad(), gradMx)
	gradMx.MulElem(layerErr.T(), gradMx)
	return n.doBackProp(inMx, gradMx, from-1, to)
}

// costMap maps name of cost to their actual implementations
var trainCost = map[string]Cost{
	"xentropy": CrossEntropy{},
	"loglike":  LogLikelihood{},
}

// ValidateTrainConfig validates training configuration.
// It returns error if any of the supplied configuration parameters are invalid.
func ValidateTrainConfig(c *config.TrainConfig) error {
	// config can't be nil
	if c == nil {
		return fmt.Errorf("Incorrect configuration supplied: %v\n", c)
	}
	// check if the requested training is supported
	if _, ok := trainCost[c.Cost]; !ok {
		return fmt.Errorf("Unsupported training cost: %s\n", c.Cost)
	}
	// Incorrect lambda supplied
	if c.Lambda < 0 {
		return fmt.Errorf("Incorrect regularizer supplied: %f\n", c.Lambda)
	}
	// if the optimization method is not supported
	if _, ok := optim[c.Optimize.Method]; !ok {
		return fmt.Errorf("Unsupported optimization method: %s\n", c.Optimize.Method)
	}
	// incorrect number of iterations supplied
	if c.Optimize.Iterations <= 0 {
		return fmt.Errorf("Incorrect number of iterations: %d\n", c.Optimize.Iterations)
	}
	return nil
}

// Train trains feedforward neural network per configuration passed in as parameter.
// It returns error if either the training configuration is invalid ot the training fails.
func (n *Network) Train(c *config.TrainConfig, inMx *mat64.Dense, labelsVec *mat64.Vector) error {
	// validate the supplied configuration
	if err := ValidateTrainConfig(c); err != nil {
		return err
	}
	// input matrix can't be nil
	if inMx == nil {
		return fmt.Errorf("Incorrect input supplied: %v\n", inMx)
	}
	// output labels can't be nil
	if labelsVec == nil {
		return fmt.Errorf("Incorrect lables supplied: %v\n", labelsVec)
	}
	// costFunc for optimization
	costFunc := func(x []float64) float64 {
		curCost, err := n.getCost(c, x, inMx, labelsVec)
		if err != nil {
			panic(err)
		}
		// TODO: can be nebled via verbose flag
		fmt.Printf("Current Cost: %f\n", curCost)
		return curCost
	}
	// gradfunc for optimization
	gradFunc := func(grad []float64, x []float64) {
		curGrad, err := n.getGradient(c, x, inMx, labelsVec)
		if err != nil {
			panic(err)
		}
		cdata := copy(grad, curGrad)
		if len(curGrad) != cdata {
			panic("Could not calculate gradient!")
		}
	}
	// initialize parameters
	var initWeights []float64
	layers := n.Layers()
	for i := range layers[1:] {
		initWeights = append(initWeights, matrix.Mx2Vec(layers[i+1].Weights(), false)...)
	}
	// optimization problem settings
	p := optimize.Problem{
		Func: costFunc,
		Grad: gradFunc,
	}
	settings := optimize.DefaultSettings()
	settings.Recorder = nil
	settings.FunctionConverge = nil
	settings.MajorIterations = c.Optimize.Iterations
	// run the optimization
	result, err := optimize.Local(p, initWeights, settings, optim[c.Optimize.Method])
	if err != nil {
		return err
	}
	fmt.Printf("Result status: %s\n", result.Status)
	return nil
}

// getCost calculates the cost of the neural network output for given input and expected output.
func (n *Network) getCost(c *config.TrainConfig, weights []float64,
	inMx *mat64.Dense, labelsVec *mat64.Vector) (float64, error) {
	// get all network layers
	layers := n.Layers()
	// if we supply network weights, set the neural network to provided weights
	if weights != nil {
		if err := setNetWeights(layers[1:], weights); err != nil {
			return -1.0, err
		}
	}
	// run forward propagation from INPUT layer
	outMx, err := n.ForwardProp(inMx, len(layers)-1)
	if err != nil {
		return -1.0, err
	}
	// labelsMx is one-of-N matrix for each output label
	// i.e. 3rd label would be: 0 0 1 0 0 etc.
	_, labelCount := outMx.Dims()
	labelsMx, err := matrix.MakeLabelsMx(labelsVec, labelCount)
	if err != nil {
		return -1.0, err
	}
	// calculate cost
	tc, _ := trainCost[c.Cost]
	cost := tc.CostFunc(inMx, outMx, labelsMx)
	// number of data samples
	samples, _ := inMx.Dims()
	reg := 0.0
	// if regularizer is not 0, calculate L2-regularization
	if c.Lambda > 0 {
		// Ignore first layer i.e. input layer
		for _, layer := range layers[1:] {
			r, c := layer.Weights().Dims()
			// Don't penalize bias units
			weightsMx := layer.Weights().View(0, 1, r, c-1)
			sqrMx := new(mat64.Dense)
			sqrMx.Apply(matrix.PowMx(2), weightsMx)
			reg += mat64.Sum(sqrMx)
		}
		reg = (c.Lambda / (2 * float64(samples))) * reg
	}
	return cost + reg, nil
}

// getGradient calculates network gradient for a particular network and configuration
// It returns a gradient slice or fails with error
func (n *Network) getGradient(c *config.TrainConfig, weights []float64,
	inMx *mat64.Dense, labelsVec *mat64.Vector) ([]float64, error) {
	// get all network layers
	layers := n.Layers()
	// if we supply network weights, set the neural network to provided weights
	if weights != nil {
		if err := setNetWeights(layers[1:], weights); err != nil {
			return nil, err
		}
	}
	// run full forward propagation
	outMx, err := n.ForwardProp(inMx, len(layers)-1)
	if err != nil {
		return nil, err
	}
	// labelsMx is one-of-N matrix for each output label
	// i.e. 3rd label would be: 0 0 1 0 0 etc.
	_, labelCount := outMx.Dims()
	labelsMx, err := matrix.MakeLabelsMx(labelsVec, labelCount)
	if err != nil {
		return nil, err
	}
	// number of data samples
	samples, _ := inMx.Dims()
	// iterate through all samples and calculate errors and corrections
	for i := 0; i < samples; i++ {
		// input vector
		inVec := inMx.RowView(i)
		// expected output
		expVec := labelsMx.RowView(i)
		// output from output layer - safe switch type - ForwardProp returns *mat64.Dense
		outVec := (outMx.(*mat64.Dense)).RowView(i)
		// calculate the error = out - y
		tc, _ := trainCost[c.Cost]
		deltaVec := tc.Delta(outVec, expVec)
		// run the backpropagation
		if err := n.BackProp(inVec.T(), deltaVec.T(), len(layers)-1); err != nil {
			return nil, err
		}
	}
	// calculate the gradient and update network weights
	var gradient []float64
	// skip zero layer - INPUT layer has no Deltas
	for i := 1; i < len(layers); i++ {
		layer := layers[i]
		deltas := layer.Deltas()
		deltas.Scale(1/float64(samples), deltas)
		if c.Lambda > 0.0 {
			rows, cols := layer.Weights().Dims()
			regWeights := mat64.NewDense(rows, cols, nil)
			reg := c.Lambda / float64(samples)
			regWeights.Clone(layer.Weights())
			// set the first column to 0
			zeros := make([]float64, rows)
			regWeights.SetCol(0, zeros)
			regWeights.Scale(reg, regWeights)
			// Update particular layer deltas matrix
			regWeights.Add(deltas, regWeights)
			gradVec := matrix.Mx2Vec(regWeights, false)
			gradient = append(gradient, gradVec...)
		}
	}
	return gradient, nil
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

// setNetWeights sets weights of provided network layers to values supplied via weights slice
// The new weights are stored in weights slice which is then rolled into particular layer's
// weights matrix layer by layer. It fails with error if the supplied weights slice
// does not contain enough elements
func setNetWeights(layers []*Layer, weights []float64) error {
	acc := 0
	wLen := len(weights)
	for _, layer := range layers {
		r, c := layer.Weights().Dims()
		if (wLen - acc) < r*c {
			return fmt.Errorf("Insufficient number of weights supplied %d\n", wLen)
		}
		err := matrix.SetMx2Vec(layer.Weights(), weights[acc:(acc+r*c)], false)
		if err != nil {
			return err
		}
		acc += r * c
	}
	return nil
}
