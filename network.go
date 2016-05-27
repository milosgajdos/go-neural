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
func addBias(x mat64.Matrix) *mat64.Dense {
	rows, cols := x.Dims()
	// Initiate bias vector with 1.0s
	ones := make([]float64, rows)
	for i, _ := range ones {
		ones[i] = 1.0
	}
	// create bias vector
	biasVec := mat64.NewVector(rows, ones)
	// create new matrix that will contain bias
	biasMx := mat64.NewDense(rows, cols+1, nil)
	// Add bias to matrix x
	biasMx.Augment(biasVec, x)
	return biasMx
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

func makeLabelsMx(y *mat64.Vector, samples, labels int) *mat64.Dense {
	mx := mat64.NewDense(samples, labels, nil)
	for i := 0; i < y.Len(); i++ {
		val := y.At(i, 0)
		mx.Set(i, int(val)-1, 1.0)
	}
	return mx
}

func makeRandMx(rows, cols uint, min, max float64) *mat64.Dense {
	// set random seed
	rand.Seed(55)
	// empirically this is supposed to be the best value
	epsilon := math.Sqrt(6.0) / math.Sqrt(float64(rows+cols))
	// allocate data slice
	randVals := make([]float64, rows*cols)
	for i := range randVals {
		// we need value between 0 and 1.0
		randVals[i] = rand.Float64()*(max-min) + min
		randVals[i] = randVals[i]*(2*epsilon) - epsilon
	}
	return mat64.NewDense(int(rows), int(cols), randVals)
}

// mx2Vec turns matrix to slice/vector
func mx2Vec(mx *mat64.Dense) []float64 {
	rows, cols := mx.Dims()
	//fmt.Printf("mx2Vec rows: %d, cols: %d\n", rows, cols)
	vector := make([]float64, rows*cols)
	for i := 0; i < cols; i++ {
		colView := mx.ColView(i)
		for j := 0; j < colView.Len(); j++ {
			//fmt.Println(colView.At(j, 0))
			vector[i*rows+j] = colView.At(j, 0)
		}
	}
	return vector
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
		//if id != 0 {
		//	r, c := layer.Weights().Dims()
		//	fmt.Printf("Initializing layer %d Weights, rows: %d, cols: %d\n", id, r, c)
		//	r, c = layer.Deltas().Dims()
		//	fmt.Printf("Initializing layer %d Deltas, rows: %d, cols: %d\n", id, r, c)
		//}
		net.layers = append(net.layers, layer)
		layerIn = layerOut
	}
	//fmt.Printf("Layer count: %d\n", len(net.layers))
	return net, nil
}

// Train runs Neural Network training for given training data X and labels y
// It returns a precision percentage on the training data or error
// TODO: config to specify kind of training etc.
func (n *Network) Train(x *mat64.Dense, y *mat64.Vector,
	labels int, lambda float64, iters uint) (float64, error) {
	// there must be at least one label
	if labels <= 0 {
		return 0.0, fmt.Errorf("Insufficient number of labels specified: %d\n", labels)
	}
	// number of data samples
	samples, _ := x.Dims()
	// calculate the cost of feedforward prop
	cost := n.Cost(x, y, labels, samples)
	fmt.Printf("Non-Reg Cost: \n%f\n", cost)
	// calculate the regularizer
	reg := n.CostReg(lambda, samples)
	gradSize := 0
	layers := n.Layers()
	for i := range layers[1:] {
		r, c := layers[i+1].Weights().Dims()
		gradSize += r * c
	}
	// allocate slice that stores gradient
	gradient := make([]float64, gradSize)
	fmt.Printf("Gradient length: %d\n", len(gradient))
	n.Gradient(gradient, x, y, labels, lambda)
	//fmt.Printf("Gradient: %v\n", gradient)
	sum := 0.0
	for i := range gradient {
		sum += gradient[i]
	}
	fmt.Printf("Gradient sum: %f\n", sum)
	return cost + reg, nil
}

// J = -(sum(sum((Y_k .* log(a3) + (1 - Y_k) .* log(1 - a3)), 2)))/m;
func (n *Network) Cost(x *mat64.Dense, y *mat64.Vector, labels int, samples int) float64 {
	// run forward propagation - start from INPUT layer
	outputMx, _ := n.forwardProp(x, 0)
	// each row represents the expected (label) result
	// i.e. label 3 will turn into vector 0 0 1 0 0 0...
	labelsMx := makeLabelsMx(y, samples, labels)
	// log(outMx)
	logOutputMx := new(mat64.Dense)
	logOutputMx.Apply(LogMx, outputMx)
	// y*log(outMx)
	mulabelsMxA := new(mat64.Dense)
	mulabelsMxA.MulElem(labelsMx, logOutputMx)
	// 1 - y
	labelsMx.Apply(SubtrMx(1.0), labelsMx)
	// 1 - outMx
	outputMx.Apply(SubtrMx(1.0), outputMx)
	// log(1-outMx)
	outputMx.Apply(LogMx, outputMx)
	// (1 - y) * log(1-outMx)
	mulabelsMxB := new(mat64.Dense)
	mulabelsMxB.MulElem(labelsMx, outputMx)
	// y*log(outMx) + (1 - y)*log(1-outMx)
	mulabelsMxB.Add(mulabelsMxA, mulabelsMxB)
	cost := -(mat64.Sum(mulabelsMxB) / float64(samples))
	return cost
}

// CostReg calculates regularizer cost of Network
// (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))
func (n *Network) CostReg(lambda float64, samples int) (cost float64) {
	// calculate regularizer
	if lambda > 0 {
		layers := n.Layers()
		// Ignore first layer i.e. input layer
		for _, layer := range layers[1:] {
			r, c := layer.Weights().Dims()
			// Don't penalize bias
			weightsMx := layer.Weights().View(0, 1, r, c-1)
			sqrMx := new(mat64.Dense)
			sqrMx.Apply(PowMx(2), weightsMx)
			cost += (lambda / (2 * float64(samples))) / mat64.Sum(sqrMx)
		}
	}
	return cost
}

// feedForward recursively progresses Neural Network
// TODO: figure out how to return mat64.Matrix
func (n *Network) forwardProp(inMx mat64.Matrix, layerIdx int) (*mat64.Dense, int) {
	layers := n.Layers()
	if layerIdx > len(layers)-1 {
		return inMx.(*mat64.Dense), layerIdx
	}
	// compute activations
	layer := layers[layerIdx]
	// calcualte the output of the layer
	out := new(mat64.Dense)
	out.Clone(layer.CompOut(inMx.(*mat64.Dense)))
	return n.forwardProp(out, layerIdx+1)
}

// backProp implements Neural Network back propagation and calculates feed forward prop errors
// Each layer updates its deltas/errors on each backward propagation
func (n *Network) backProp(inMx, deltaMx mat64.Matrix,
	layerIdx, outIdx, sampleIdx int) (*mat64.Dense, int) {
	// network layers
	layers := n.Layers()
	// Weights and Deltas from the same layer
	bpWeightsMx := layers[layerIdx].Weights()
	bpDeltasMx := layers[layerIdx].Deltas()
	// Out layer produces output to the w/d layer
	bpOutLayer := layers[outIdx]
	bpOutMx := bpOutLayer.Out()
	// Printouts
	//fmt.Printf("Layer WeightsIdx: %d\n", layerIdx)
	//fmt.Printf("Layer DeltasIdx: %d\n", layerIdx)
	//fmt.Printf("Layer OutIdx: %d\n", outIdx)
	//r, c := bpWeightsMx.Dims()
	//fmt.Printf("weightsMX rows: %d, cols: %d\n", r, c)
	//r, c = bpDeltasMx.Dims()
	//fmt.Printf("deltasMX rows: %d, cols: %d\n", r, c)
	//r, c = bpOutMx.Dims()
	//fmt.Printf("outMX rows: %d, cols: %d\n", r, c)
	// If we reach the first hidden layer, return
	if outIdx == 0 {
		dMx := new(mat64.Dense)
		// inMx is the same as bpOutMx(i)
		biasInMx := addBias(inMx)
		//r, c := biasInMx.Dims()
		//fmt.Printf("Bias inMx rows: %d, cols: %d\n", r, c)
		dMx.Mul(deltaMx.T(), biasInMx)
		bpDeltasMx.Add(bpDeltasMx, dMx)
		return bpDeltasMx, layerIdx
	}
	// Layer activation functions forward and backward
	forwFunc := func(i, j int, x float64) float64 {
		return bpOutLayer.NeuronFunc().ForwFn(x)
	}
	backFunc := func(i, j int, x float64) float64 {
		return bpOutLayer.NeuronFunc().BackFn(x)
	}
	// sigmoid(Out)
	actOut := new(mat64.Dense)
	actOut.Apply(forwFunc, bpOutMx)
	//r, c = actOut.Dims()
	//fmt.Printf("ActOut rows: %d, cols: %d\n", r, c)
	biasActOut := addBias(actOut)
	// Just pick the first row
	actSample := biasActOut.RowView(sampleIdx).T()
	//r, c = actSample.Dims()
	//fmt.Printf("BiasActOut rows: %d, cols: %d\n", r, c)
	//fmt.Println("BiasActOut")
	//fa := mat64.Formatted(actSample, mat64.Prefix(""))
	//fmt.Printf("%v\n\n", fa)
	//r, c = deltaMx.Dims()
	//fmt.Printf("DeltaMX rows: %d, cols: %d\n", r, c)
	//fmt.Println("DeltaMx")
	//fa = mat64.Formatted(deltaMx, mat64.Prefix(""))
	//fmt.Printf("%v\n\n", fa)
	// delta_i'*a_(i-1)
	dMx := new(mat64.Dense)
	dMx.Mul(deltaMx.T(), actSample)
	//r, c = dMx.Dims()
	//fmt.Printf("DMX out rows: %d, cols: %d\n", r, c)
	//fmt.Println("DMX out")
	//fa = mat64.Formatted(dMx, mat64.Prefix(""))
	//fmt.Printf("%v\n\n", fa)
	//r, c = bpDeltasMx.Dims()
	//fmt.Printf("Deltas rows: %d, cols: %d\n", r, c)
	// D = D + delta*Sig(Oi)
	bpDeltasMx.Add(bpDeltasMx, dMx)
	// tmp var
	tmp := new(mat64.Dense)
	tmp.Mul(bpWeightsMx.T(), deltaMx.T())
	//r, c = bpWeightsMx.T().Dims()
	//fmt.Printf("BP WEIGHTS rows: %d, cols: %d\n", r, c)
	//r, c = deltaMx.T().Dims()
	//fmt.Printf("DeltaMX rows: %d, cols: %d\n", r, c)
	r, c := tmp.Dims()
	//fmt.Printf("TMP rows: %d, cols: %d\n", r, c)
	// ignore the bias output
	delta := tmp.View(1, 0, r-1, c).(*mat64.Dense)
	//r, c = delta.T().Dims()
	//fmt.Printf("NEW DELTA rows: %d, cols: %d\n", r, c)
	// compute sigmoid gradient for a particular sample layer output
	outSample := bpOutMx.RowView(sampleIdx).T()
	//r, c = outSample.Dims()
	//fmt.Printf("OUT SAMPLE rows: %d, cols: %d\n", r, c)
	sigGradOut := new(mat64.Dense)
	sigGradOut.Apply(backFunc, outSample)
	//r, c = sigGradOut.Dims()
	//fmt.Printf("SIGMOID GRAD rows: %d, cols: %d\n", r, c)
	sigGradOut.MulElem(delta.T(), sigGradOut)
	// run recursively
	return n.backProp(inMx, sigGradOut, layerIdx-1, outIdx-1, sampleIdx)
}

// Gradient calculates network gradient at point x
func (n *Network) Gradient(gradient []float64, x *mat64.Dense, y *mat64.Vector,
	labels int, lambda float64) []float64 {
	// network layers
	layers := n.Layers()
	layerCount := len(layers)
	// dimensions of input matrix
	samples, _ := x.Dims()
	// make labels matrix
	labelsMx := makeLabelsMx(y, samples, labels)
	// iterate through all samples and calculate errors and corrections
	for i := 0; i < samples; i++ {
		// pick a sample
		inSample := x.RowView(i)
		//fmt.Println("Input Sample")
		//fa := mat64.Formatted(inSample, mat64.Prefix(""))
		//fmt.Printf("%v\n\n", fa)
		// pick the expected output
		expOutput := labelsMx.RowView(i)
		//r, c := expOutput.Dims()
		//fmt.Printf("expOutput rows: %d, cols: %d\n", r, c)
		//fmt.Println("Expected Output")
		//fa := mat64.Formatted(expOutput, mat64.Prefix(""))
		//fmt.Printf("%v\n\n", fa)
		// pick actual output from output layer
		output := layers[layerCount-1].Out().RowView(i)
		//output := layers[layerCount-1].Out().ColView(i)
		//r, c = output.Dims()
		//fmt.Printf("Output rows: %d, cols: %d\n", r, c)
		//fmt.Println("Computed output")
		//fa = mat64.Formatted(output, mat64.Prefix(""))
		//fmt.Printf("%v\n\n", fa)
		// calculate the error = out - y
		output.SubVec(output, expOutput)
		//fmt.Println("Subtracted output")
		//fa = mat64.Formatted(output, mat64.Prefix(""))
		//fmt.Printf("%v\n\n", fa)
		// run the backpropagation
		n.backProp(inSample.T(), output.T(), layerCount-1, layerCount-2, i)
	}
	// zero-th layer is INPUT layer and has no Deltas
	next := 0
	for i := 1; i < len(layers); i++ {
		deltas := layers[i].Deltas()
		deltas.Scale(1/float64(samples), deltas)
		gradMx := n.GradientReg(i, lambda, samples)
		gradMx.Add(deltas, gradMx)
		r, c := gradMx.Dims()
		//fmt.Printf("GradMx rows: %d, cols: %d\n", r, c)
		//fmt.Println("WAT")
		gradVec := mx2Vec(gradMx)
		for j := 0; j < len(gradVec); j++ {
			gradient[next+j] = gradVec[j]
		}
		next += r * c
	}
	return gradient
}

// GradientReg calculates gradient regularizer for a particular layer identified by index idx
func (n *Network) GradientReg(idx int, lambda float64, samples int) *mat64.Dense {
	layers := n.Layers()
	layer := layers[idx]
	r, c := layer.Weights().Dims()
	// initialize weights
	regWeights := mat64.NewDense(r, c, nil)
	if lambda > 0 {
		// calculate the regularizer
		reg := lambda / float64(samples)
		regWeights.Clone(layer.Weights())
		// set the first column to 0
		zeros := make([]float64, r)
		regWeights.SetCol(0, zeros)
		regWeights.Scale(reg, regWeights)
	}
	return regWeights
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
	out      *mat64.Dense
	weights  *mat64.Dense
	deltas   *mat64.Dense
	neurFunc *NeuronFunc
}

// NewLayer creates new neural netowrk layer and returns it
func NewLayer(id uint, layerKind LayerKind, net *Network, layerIn, layerOut uint) (*Layer, error) {
	layer := &Layer{}
	layer.id = id
	layer.kind = layerKind
	layer.net = net
	// INPUT layer does not have weights matrix nor activation funcs
	if layerKind != INPUT {
		// initialize weights to random values
		layer.weights = makeRandMx(layerOut, layerIn+1, 0.0, 1.0)
		// initializes deltas to zero values
		layer.deltas = mat64.NewDense(int(layerOut), int(layerIn)+1, nil)
		// TODO: parameterize activation functions
		layer.neurFunc = &NeuronFunc{
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
	return l.weights
}

// Deltas returns layer's deltas matrix
func (l *Layer) Deltas() *mat64.Dense {
	return l.deltas
}

// Out returns particular layer output
func (l *Layer) Out() *mat64.Dense {
	return l.out
}

// SetWeights allows to set the layer's weights matrix
func (l *Layer) SetWeights(weights *mat64.Dense) error {
	if l.kind == INPUT {
		return errors.New("INPUT layer does not have weights matrix")
	}
	// TODO: should we Copy/Clone rather than plain assign?
	l.weights = weights
	return nil
}

func (l *Layer) SetDeltas(deltas *mat64.Dense) error {
	if l.kind == INPUT {
		return errors.New("INPUT layer does not have deltas matrix")
	}
	// TODO: should we Copy/Clone rather than plain assign?
	l.deltas = deltas
	return nil
}

// Out calculates matrix of output for a particular inputMx
func (l *Layer) CompOut(inputMx *mat64.Dense) *mat64.Dense {
	// if it's INPUT layer, output is input
	if l.kind == INPUT {
		//r, c := inputMx.Dims()
		//fmt.Printf("%s Out rows: %d, cols: %d\n", r, c)
		l.out = inputMx
		return inputMx
	}
	// add bias to input
	biasMx := addBias(inputMx)
	// otherwise apply weights
	out := new(mat64.Dense)
	out.Mul(biasMx, l.weights.T())
	// activate layer neurons
	activFunc := func(i, j int, x float64) float64 {
		return l.neurFunc.ForwFn(x)
	}
	out.Apply(activFunc, out)
	//r, c := out.Dims()
	//fmt.Printf("%s Out rows: %d, cols: %d\n", r, c)
	// store output matrix for this layer
	l.out = out
	return out
}

func (l *Layer) SetNeurFunc(nf *NeuronFunc) {
	l.neurFunc = nf
}

func (l Layer) NeuronFunc() *NeuronFunc {
	return l.neurFunc
}

// ActivationFn represents a Neuron activation function
// It accepts a vector of float numbers and returns a single value
type ActivationFn func(float64) float64

// NeuronFunc provides activation functions for forward and backprop
type NeuronFunc struct {
	ForwFn ActivationFn
	BackFn ActivationFn
}
