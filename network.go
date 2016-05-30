package main

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
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
func ones(rows, cols int) (*mat64.Dense, error) {
	if rows <= 0 || cols <= 0 {
		return nil, errors.New("Rows and columns must be positive integer")
	}
	onesMx := mat64.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			onesMx.Set(i, j, 1.0)
		}
	}
	return onesMx, nil
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
func mx2Vec(m *mat64.Dense, byRow bool) []float64 {
	if byRow {
		return mxByRow(m)
	}
	return mxByCol(m)
}

func mxByRow(mx *mat64.Dense) []float64 {
	rows, cols := mx.Dims()
	vector := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		view := mx.RowView(i)
		for j := 0; j < view.Len(); j++ {
			vector[i*cols+j] = view.At(j, 0)
		}
	}
	return vector
}

func mxByCol(mx *mat64.Dense) []float64 {
	rows, cols := mx.Dims()
	vector := make([]float64, rows*cols)
	for i := 0; i < cols; i++ {
		view := mx.ColView(i)
		for j := 0; j < view.Len(); j++ {
			vector[i*rows+j] = view.At(j, 0)
		}
	}
	return vector
}

// vec2Mx copies vector into matrix
func vec2Mx(vec []float64, mx *mat64.Dense, byRow bool) {
	if byRow {
		vecByRow(vec, mx)
		return
	}
	vecByCol(vec, mx)
}

func vecByCol(vec []float64, mx *mat64.Dense) {
	rows, cols := mx.Dims()
	acc := 0
	for i := 0; i < cols; i++ {
		mx.SetCol(i, vec[acc:(acc+rows)])
		acc += rows
	}
}

func vecByRow(vec []float64, mx *mat64.Dense) {
	rows, cols := mx.Dims()
	acc := 0
	for i := 0; i < rows; i++ {
		mx.SetRow(i, vec[acc:(acc+cols)])
		acc += cols
	}
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
func (n *Network) Train(mx *mat64.Dense, y *mat64.Vector,
	labels int, lambda float64, iters int) (float64, error) {
	// there must be at least one label
	if labels <= 0 {
		return 0.0, fmt.Errorf("Number of labels must be positive integer: %d\n", labels)
	}
	// weightsVec contains neural network parameters rolled into vector
	weightsVec := make([]float64, 0)
	// Roll in the layer weights Matrices into weightsVec
	// TODO: this is temporary - need to find a way to avoid double allocation
	layers := n.Layers()
	for i := range layers[1:] {
		weightsVec = append(weightsVec, mx2Vec(layers[i+1].Weights(), false)...)
	}
	// costFunc
	costFunc := func(x []float64) float64 {
		return n.CostFunc(x, mx, y, labels, lambda)
	}
	// gradFunc
	// allocate slice for gradient
	//gradientVec := make([]float64, len(weightsVec))
	gradFunc := func(grad []float64, x []float64) {
		if len(x) != len(grad) {
			panic("incorrect size of the gradient")
		}
		n.GradFunc(grad, x, mx, y, labels, lambda)
	}
	// optimization problem
	p := optimize.Problem{
		Func: costFunc,
		Grad: gradFunc,
	}
	settings := optimize.DefaultSettings()
	settings.Recorder = nil
	settings.FunctionConverge = nil
	settings.MajorIterations = iters
	result, err := optimize.Local(p, weightsVec, settings, &optimize.BFGS{})
	if err != nil {
		log.Fatal(err)
	}
	//if err = result.Status.Err(); err != nil {
	//	log.Fatal(err)
	//}
	fmt.Printf("result.Status: %v\n", result.Status)
	// calculate the cost of feedforward prop
	//cost := n.CostFunc(weightsVec, x, y, labels, lambda)
	//grad := n.GradFunc(gradientVec, weightsVec, x, y, labels, lambda)
	//fmt.Println("Gradient length", len(grad))
	return 0.0, nil
}

// J = -(sum(sum((Y_k .* log(a3) + (1 - Y_k) .* log(1 - a3)), 2)))/m;
func (n *Network) CostFunc(netWeights []float64, x *mat64.Dense, y *mat64.Vector,
	labels int, lambda float64) float64 {
	// TODO: move to a separate function
	layers := n.Layers()
	acc := 0
	for _, layer := range layers[1:] {
		r, c := layer.Weights().Dims()
		vec2Mx(netWeights[acc:(acc+r*c)], layer.Weights(), false)
		acc += r * c
	}
	// number of data samples
	samples, _ := x.Dims()
	// run forward propagation - start from INPUT layer
	output, _ := n.forwardProp(x, 0)
	outputMx := output.(*mat64.Dense)
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
	// calculate the regularizer
	reg := n.costReg(lambda, samples)
	cost += reg
	fmt.Printf("Cost: %f\n", cost)
	return cost
}

// CostFuncReg calculates regularizer cost of Network
// (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))
func (n *Network) costReg(lambda float64, samples int) (cost float64) {
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
func (n *Network) forwardProp(inMx mat64.Matrix, layerIdx int) (mat64.Matrix, int) {
	layers := n.Layers()
	if layerIdx > len(layers)-1 {
		return inMx, layerIdx
	}
	// compute activations
	layer := layers[layerIdx]
	// calcualte the output of the layer
	out := new(mat64.Dense)
	out.Clone(layer.CompOut(inMx))
	return n.forwardProp(out, layerIdx+1)
}

// GradFunc calculates network gradient at point x
func (n *Network) GradFunc(gradient []float64, netWeights []float64,
	x *mat64.Dense, y *mat64.Vector, labels int, lambda float64) []float64 {
	// network layers and layer count
	layers := n.Layers()
	layerCount := len(layers)
	// Init net layers
	acc := 0
	for _, layer := range layers[1:] {
		r, c := layer.Weights().Dims()
		vec2Mx(netWeights[acc:(acc+r*c)], layer.Weights(), false)
		acc += r * c
	}
	// dimensions of input matrix
	samples, _ := x.Dims()
	// make labels matrix
	labelsMx := makeLabelsMx(y, samples, labels)
	// iterate through all samples and calculate errors and corrections
	for i := 0; i < samples; i++ {
		// pick a sample
		inSample := x.RowView(i)
		// pick the expected output
		expOutput := labelsMx.RowView(i)
		// pick actual output from output layer
		output := layers[layerCount-1].Out().RowView(i)
		// calculate the error = out - y
		output.SubVec(output, expOutput)
		// run the backpropagation
		n.backProp(inSample.T(), output.T(), layerCount-1, layerCount-2, i)
	}
	// zero-th layer is INPUT layer and has no Deltas
	next := 0
	for i := 1; i < layerCount; i++ {
		deltas := layers[i].Deltas()
		deltas.Scale(1/float64(samples), deltas)
		if lambda > 0 {
			gradMx := n.gradientReg(i, lambda, samples)
			gradMx.Add(deltas, gradMx)
			r, c := gradMx.Dims()
			gradVec := mx2Vec(gradMx, false)
			for j := 0; j < len(gradVec); j++ {
				gradient[next+j] = gradVec[j]
			}
			next += r * c
		}
	}
	return gradient
}

// GradFuncReg calculates gradient regularizer for a particular layer identified by index idx
func (n *Network) gradientReg(idx int, lambda float64, samples int) *mat64.Dense {
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

// backProp implements Neural Network back propagation and calculates feed forward prop errors
// Each layer updates its deltas/errors on each backward propagation
func (n *Network) backProp(inMx, deltaMx mat64.Matrix,
	layerIdx, outIdx, sampleIdx int) (*mat64.Dense, int) {
	// network layers
	layers := n.Layers()
	// Weights and Deltas from the same layer
	bpWeightLayer := layers[layerIdx]
	bpWeightsMx := bpWeightLayer.Weights()
	bpDeltasMx := bpWeightLayer.Deltas()
	// Out layer produces output to the w/d layer
	bpOutLayer := layers[outIdx]
	bpOutMx := bpOutLayer.Out()
	bpActInMx := bpOutLayer.ActIn()
	// If we reach the first hidden layer, return
	if outIdx == 0 {
		dMx := new(mat64.Dense)
		// inMx is the same as bpOutMx(i)
		biasInMx := addBias(inMx)
		dMx.Mul(deltaMx.T(), biasInMx)
		bpDeltasMx.Add(bpDeltasMx, dMx)
		return bpDeltasMx, layerIdx
	}
	// add bias to Out matrix
	biasOutMx := addBias(bpOutMx)
	// Just pick the first row
	outSample := biasOutMx.RowView(sampleIdx).T()
	// delta_i'*a_(i-1)
	dMx := new(mat64.Dense)
	dMx.Mul(deltaMx.T(), outSample)
	// D = D + delta*O(i)
	bpDeltasMx.Add(bpDeltasMx, dMx)
	// tmp var
	tmp := new(mat64.Dense)
	tmp.Mul(bpWeightsMx.T(), deltaMx.T())
	// ignore the bias output
	r, c := tmp.Dims()
	delta := tmp.View(1, 0, r-1, c).(*mat64.Dense)
	// compute sigmoid gradient for a particular activation input
	backFunc := func(i, j int, x float64) float64 {
		return bpOutLayer.NeuronFunc().BackFn(x)
	}
	actInSample := bpActInMx.RowView(sampleIdx).T()
	sigGradOut := new(mat64.Dense)
	sigGradOut.Apply(backFunc, actInSample)
	sigGradOut.MulElem(delta.T(), sigGradOut)
	// run recursively
	return n.backProp(inMx, sigGradOut, layerIdx-1, outIdx-1, sampleIdx)
}

// Classify classifies the provided data vector to particular label
// It returns the label number or error
func (n *Network) Classify(x *mat64.Vector) int {
	output, _ := n.forwardProp(x, 0)
	rows, _ := output.Dims()
	max := mat64.Max(output)
	for i := 0; i < rows; i++ {
		if output.At(i, 0) == max {
			return i
		}
	}
	return 0
}

// Validate runs Neural Net validation through the provided training set and returns
// percentage of successful classifications or error
func (n *Network) Validate(x *mat64.Dense, y *mat64.Vector) (float64, error) {
	output, _ := n.forwardProp(x, 0)
	outputMx := output.(*mat64.Dense)
	rows, _ := outputMx.Dims()
	hits := 0.0
	for i := 0; i < rows; i++ {
		row := outputMx.RowView(i)
		max := mat64.Max(row)
		for j := 0; j < row.Len(); j++ {
			if row.At(j, 0) == max {
				if j+1 == int(y.At(i, 0)) {
					hits++
					break
				}
			}
		}
	}
	success := (hits / float64(y.Len())) * 100
	return success, nil
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
	id   uint
	kind LayerKind
	net  *Network
	// TODO: turn these to mat64.Matrix-es
	out      *mat64.Dense
	actIn    *mat64.Dense
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

// ActIn returns activation function input
func (l *Layer) ActIn() *mat64.Dense {
	return l.actIn
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
func (l *Layer) CompOut(inputMx mat64.Matrix) *mat64.Dense {
	// if it's INPUT layer, output is input
	if l.kind == INPUT {
		l.out = inputMx.(*mat64.Dense)
		return inputMx.(*mat64.Dense)
	}
	// add bias to input
	biasInMx := addBias(inputMx)
	// otherwise apply weights
	actIn := new(mat64.Dense)
	actIn.Mul(biasInMx, l.weights.T())
	// store output matrix for this layer
	l.actIn = actIn
	// activate layer neurons
	out := new(mat64.Dense)
	activFunc := func(i, j int, x float64) float64 {
		return l.neurFunc.ForwFn(x)
	}
	out.Apply(activFunc, actIn)
	// store activation matrix for this layer
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
