package neural

import (
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/pkg/config"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
	"github.com/stretchr/testify/assert"
)

var (
	fileName  = "manifest.yml"
	inMx      *mat64.Dense
	labelsVec *mat64.Vector
)

func setup() {
	content := []byte(`kind: feedfwd
task: class
network:
  input:
    size: 4
  hidden:
    size: [5]
    activation: sigmoid
  output:
    size: 5
    activation: softmax
training:
  kind: backprop
  cost: xentropy
  params:
    lambda: 1.0
  optimize:
    method: bfgs
    iterations: 2`)

	tmpPath := filepath.Join(os.TempDir(), fileName)
	if err := ioutil.WriteFile(tmpPath, content, 0666); err != nil {
		log.Fatal(err)
	}

	// Create features matrix
	features := []float64{5.1, 3.5, 1.4, 0.1,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.3,
		4.6, 3.1, 1.5, 0.4,
		5.0, 3.6, 1.4, 0.5}
	inMx = mat64.NewDense(5, 4, features)
	labels := []float64{2.0, 1.0, 3.0, 2.0, 4.0}
	labelsVec = mat64.NewVector(len(labels), labels)
}

func teardown() {
	os.Remove(filepath.Join(os.TempDir(), fileName))
}

func TestMain(m *testing.M) {
	// set up tests
	setup()
	// run the tests
	retCode := m.Run()
	// delete test files
	teardown()
	// call with result of m.Run()
	os.Exit(retCode)
}

func TestNetworkKind(t *testing.T) {
	assert := assert.New(t)
	// create different network kinds
	networkKinds := []struct {
		k   NetworkKind
		out string
	}{
		{FEEDFWD, "FEEDFWD"},
		{NetworkKind(1000), "UNKNOWN"},
	}

	for _, networkKind := range networkKinds {
		assert.Equal(networkKind.k.String(), networkKind.out)
	}
}

func TestNewNetwork(t *testing.T) {
	assert := assert.New(t)
	// basic configuration settings
	tmpPath := path.Join(os.TempDir(), fileName)
	conf, err := config.New(tmpPath)
	assert.NotNil(conf)
	assert.NoError(err)
	// Config.Network
	c := conf.Network
	// create new network
	n, err := NewNetwork(c)
	assert.NotNil(n)
	assert.NoError(err)
	// empty config
	n, err = NewNetwork(nil)
	assert.Nil(n)
	assert.Error(err)
	// unknown network kind
	c.Kind = "foobar"
	n, err = NewNetwork(c)
	assert.Nil(n)
	assert.Error(err)
	c.Kind = "feedfwd"
	// nil architecture
	origArch := c.Arch
	c.Arch = nil
	n, err = NewNetwork(c)
	assert.Nil(n)
	assert.Error(err)
	c.Arch = origArch
	// nil INPUT
	origInput := c.Arch.Input
	c.Arch.Input = nil
	n, err = NewNetwork(c)
	assert.Nil(n)
	assert.Error(err)
	c.Arch.Input = origInput
	// incorrect INPUT layer size
	origInSize := c.Arch.Input.Size
	c.Arch.Input.Size = -100
	n, err = NewNetwork(c)
	assert.Nil(n)
	assert.Error(err)
	c.Arch.Input.Size = origInSize
	// incorrect HIDDEN layer size
	origHidSize := c.Arch.Hidden[0].Size
	c.Arch.Hidden[0].Size = -100
	n, err = NewNetwork(c)
	assert.Nil(n)
	assert.Error(err)
	c.Arch.Hidden[0].Size = origHidSize
	// nil OUTPUT
	origOutput := c.Arch.Output
	c.Arch.Output = nil
	n, err = NewNetwork(c)
	assert.Nil(n)
	assert.Error(err)
	c.Arch.Output = origOutput
	// incorrect OUTPUT layer size
	origOutSize := c.Arch.Output.Size
	c.Arch.Output.Size = -100
	n, err = NewNetwork(c)
	assert.Nil(n)
	assert.Error(err)
	c.Arch.Output.Size = origOutSize
}

func TestAddLayer(t *testing.T) {
	assert := assert.New(t)
	// basic configuration settings
	tmpPath := path.Join(os.TempDir(), fileName)
	conf, err := config.New(tmpPath)
	assert.NotNil(conf)
	assert.NoError(err)
	// Config.Network
	c := conf.Network
	// create new network
	n, err := NewNetwork(c)
	assert.NotNil(n)
	assert.NoError(err)
	// create input layer
	l, err := NewLayer(c.Arch.Input, 10)
	assert.NotNil(l)
	assert.NoError(err)
	// add duplicate input layer
	err = n.AddLayer(l)
	assert.Error(err)
	// create output layer
	l, err = NewLayer(c.Arch.Output, 10)
	assert.NotNil(l)
	assert.NoError(err)
	// add duplicate output layer
	err = n.AddLayer(l)
	assert.Error(err)
	// add another hidden layer
	l, err = NewLayer(c.Arch.Hidden[0], 10)
	assert.NotNil(l)
	assert.NoError(err)
	// add duplicate output layer
	err = n.AddLayer(l)
	assert.NoError(err)
}

func TestID(t *testing.T) {
	assert := assert.New(t)
	// create dummy network
	tmpPath := path.Join(os.TempDir(), fileName)
	c, err := config.New(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	n, err := NewNetwork(c.Network)
	assert.NotNil(n)
	assert.NoError(err)
	assert.Len(n.ID(), 10)
}

func TestKind(t *testing.T) {
	assert := assert.New(t)
	// create dummy network
	tmpPath := path.Join(os.TempDir(), fileName)
	c, err := config.New(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	// create dummy network
	n, err := NewNetwork(c.Network)
	assert.NotNil(n)
	assert.NoError(err)
	assert.Equal(n.Kind(), FEEDFWD)
}

func TestLayers(t *testing.T) {
	assert := assert.New(t)
	// create dummy network
	tmpPath := path.Join(os.TempDir(), fileName)
	c, err := config.New(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	n, err := NewNetwork(c.Network)
	assert.NotNil(n)
	assert.NoError(err)
	layers := n.Layers()
	assert.NotNil(layers)
	assert.True(len(layers) > 0)
	// INPUT layer must be of INPUT kind
	layerKind := layers[0].Kind()
	assert.Equal(layerKind, INPUT)
	// HIDDEN layers
	for _, layer := range layers[1:2] {
		assert.Equal(layer.Kind(), HIDDEN)
	}
	// OUTPUT layer
	layerKind = layers[len(layers)-1].Kind()
	assert.Equal(layerKind, OUTPUT)
}

func TestForwardProp(t *testing.T) {
	assert := assert.New(t)
	// create features matrix
	features := []float64{5.1, 3.5, 1.4, 0.2,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.2,
		4.6, 3.1, 1.5, 0.2,
		5.0, 3.6, 1.4, 0.2}
	inMx := mat64.NewDense(5, 4, features)
	inRows, inCols := inMx.Dims()
	// create test network
	tmpPath := path.Join(os.TempDir(), fileName)
	c, err := config.New(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	// simplify the config
	c.Network.Arch.Input.Size = inCols
	hiddenLayers := []*config.LayerConfig{
		{Kind: "hidden",
			Size: 5,
			NeurFn: &config.NeuronConfig{
				Activation: "sigmoid",
			},
		},
	}
	c.Network.Arch.Hidden = hiddenLayers
	c.Network.Arch.Output.Size = 5
	net, err := NewNetwork(c.Network)
	assert.NotNil(net)
	assert.NoError(err)
	// retrieve layers
	layers := net.Layers()
	assert.NotNil(layers)
	// 0-th end layer returns the input
	out, err := net.ForwardProp(inMx, 0)
	assert.NotNil(out)
	assert.NoError(err)
	assert.Equal(out, inMx)
	// can't propagate beyond last layer
	out, err = net.ForwardProp(inMx, len(layers))
	assert.Nil(out)
	assert.Error(err)
	// Propagate till the last layer
	out, err = net.ForwardProp(inMx, len(layers)-1)
	assert.NotNil(out)
	assert.NoError(err)
	outRows, outCols := out.Dims()
	assert.Equal(outRows, inRows)
	assert.Equal(outCols, c.Network.Arch.Output.Size)
	// Propagate to the hidden layer
	out, err = net.ForwardProp(inMx, len(layers)-2)
	assert.NotNil(out)
	assert.NoError(err)
	outRows, outCols = out.Dims()
	assert.Equal(outRows, inRows)
	assert.Equal(outCols, c.Network.Arch.Hidden[0].Size)
	// can't fwd propagate nil input
	out, err = net.ForwardProp(nil, len(layers)-1)
	assert.Nil(out)
	assert.Error(err)
	// incorrect input dimensions
	tstMx := mat64.NewDense(100, 20, nil)
	assert.NotNil(tstMx)
	out, err = net.ForwardProp(tstMx, len(layers)-1)
	assert.Nil(out)
	assert.Error(err)
}

func TestBackProp(t *testing.T) {
	assert := assert.New(t)
	// create features matrix
	features := []float64{5.1, 3.5, 1.4, 0.2,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.2,
		4.6, 3.1, 1.5, 0.2,
		5.0, 3.6, 1.4, 0.2}
	inMx := mat64.NewDense(5, 4, features)
	_, inCols := inMx.Dims()
	// create test network
	tmpPath := path.Join(os.TempDir(), fileName)
	c, err := config.New(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	// simplify the config
	c.Network.Arch.Input.Size = inCols
	hiddenLayers := []*config.LayerConfig{
		{Kind: "hidden",
			Size: 5,
			NeurFn: &config.NeuronConfig{
				Activation: "sigmoid",
			},
		},
	}
	c.Network.Arch.Hidden = hiddenLayers
	c.Network.Arch.Output.Size = 5
	net, err := NewNetwork(c.Network)
	assert.NotNil(net)
	assert.NoError(err)
	// retrieve layers
	layers := net.Layers()
	assert.NotNil(layers)
	// expected labels
	expVal := []float64{2, 1, 3, 2, 4}
	expVec := mat64.NewVector(len(expVal), expVal)
	// propagate forward to the last layer
	out, err := net.ForwardProp(inMx, len(layers)-1)
	assert.NotNil(out)
	assert.NoError(err)
	errVec := (out.(*mat64.Dense)).RowView(0)
	errVec.SubVec(errVec, expVec)
	// Pick a sample vector and test backprop
	sampleVec := inMx.RowView(0)
	err = net.BackProp(sampleVec.T(), errVec.T(), len(layers)-1)
	assert.NoError(err)
	// nil input matrix throws errors
	err = net.BackProp(nil, nil, 0)
	assert.Error(err)
	// nil error matrix throws error
	err = net.BackProp(sampleVec.T(), nil, len(layers)-1)
	assert.Error(err)
	// number of bp layers beyond network size throws error
	err = net.BackProp(sampleVec.T(), errVec.T(), 100)
	assert.Error(err)
	// negative from boundary throws error
	err = net.BackProp(sampleVec.T(), errVec.T(), 100)
	assert.Error(err)
}

func TestValidateTrainConfig(t *testing.T) {
	assert := assert.New(t)
	// start with correct config
	c := &config.TrainConfig{
		Kind:   "backprop",
		Cost:   "xentropy",
		Lambda: 1.0,
		Optimize: &config.OptimConfig{
			Method:     "bfgs",
			Iterations: 50,
		},
	}
	err := ValidateTrainConfig(c)
	assert.NoError(err)
	// config can't be nil
	err = ValidateTrainConfig(nil)
	assert.Error(err)
	// unsupported cost function
	origCost := c.Cost
	c.Cost = "fooCost"
	err = ValidateTrainConfig(c)
	assert.Error(err)
	c.Cost = origCost
	// wrong lambda
	origLambda := c.Lambda
	c.Lambda = -100
	err = ValidateTrainConfig(c)
	assert.Error(err)
	c.Lambda = origLambda
	// unsupported Optimization method
	origMethod := c.Optimize.Method
	c.Optimize.Method = "foobar"
	err = ValidateTrainConfig(c)
	assert.Error(err)
	c.Optimize.Method = origMethod
	// Wrong number of iterations
	origIters := c.Optimize.Iterations
	c.Optimize.Iterations = -10
	err = ValidateTrainConfig(c)
	assert.Error(err)
	c.Optimize.Iterations = origIters
}

func TestTrain(t *testing.T) {
	assert := assert.New(t)
	// basic configuration settings
	tmpPath := path.Join(os.TempDir(), fileName)
	conf, err := config.New(tmpPath)
	assert.NotNil(conf)
	assert.NoError(err)
	// create new network
	netConf := conf.Network
	n, err := NewNetwork(netConf)
	assert.NotNil(n)
	assert.NoError(err)
	// nil config causes error
	trainConf := conf.Training
	err = n.Train(nil, inMx, labelsVec)
	assert.Error(err)
	// nil input causes error
	err = n.Train(trainConf, nil, labelsVec)
	assert.Error(err)
	// nil labelsVec causes error
	err = n.Train(trainConf, inMx, nil)
	assert.Error(err)
	// calculate cost
	err = n.Train(trainConf, inMx, labelsVec)
	assert.NoError(err)
}

func TestClassify(t *testing.T) {
	assert := assert.New(t)
	// basic configuration settings
	tmpPath := path.Join(os.TempDir(), fileName)
	conf, err := config.New(tmpPath)
	assert.NotNil(conf)
	assert.NoError(err)
	// create new network
	netConf := conf.Network
	n, err := NewNetwork(netConf)
	assert.NotNil(n)
	assert.NoError(err)
	// nil input throws error
	classOut, err := n.Classify(nil)
	assert.Nil(classOut)
	assert.Error(err)
	// classify the features input
	classOut, err = n.Classify(inMx)
	assert.NotNil(n)
	assert.NoError(err)
	inRows, _ := inMx.Dims()
	oRows, oCols := classOut.Dims()
	// every input must be classified
	assert.Equal(oRows, inRows)
	// output vector is a one-of-N classification vector
	assert.Equal(oCols, netConf.Arch.Output.Size)
	// pass a single vector in
	tstIn := inMx.RowView(0).T()
	classOut, err = n.Classify(tstIn)
	assert.NotNil(n)
	assert.NoError(err)
	oRows, oCols = classOut.Dims()
	assert.Equal(oRows, 1)
	assert.Equal(oCols, netConf.Arch.Output.Size)
}

func TestValidate(t *testing.T) {
	assert := assert.New(t)
	// basic configuration settings
	tmpPath := path.Join(os.TempDir(), fileName)
	conf, err := config.New(tmpPath)
	assert.NotNil(conf)
	assert.NoError(err)
	// create new network
	netConf := conf.Network
	n, err := NewNetwork(netConf)
	assert.NotNil(n)
	assert.NoError(err)
	// expected labels
	expVal := []float64{2, 1, 3, 2, 4}
	expVec := mat64.NewVector(len(expVal), expVal)
	// nil input throws error
	success, err := n.Validate(nil, expVec)
	assert.Error(err)
	assert.True(success == 0.0)
	// nil expected value throws error
	success, err = n.Validate(inMx, nil)
	assert.Error(err)
	assert.True(success == 0.0)
	// run validation
	success, err = n.Validate(inMx, expVec)
	assert.NoError(err)
	assert.True(success < 100.0)
}

func TestSetNetWeights(t *testing.T) {
	assert := assert.New(t)
	// basic configuration settings
	tmpPath := path.Join(os.TempDir(), fileName)
	conf, err := config.New(tmpPath)
	assert.NotNil(conf)
	assert.NoError(err)
	// create new network
	netConf := conf.Network
	n, err := NewNetwork(netConf)
	assert.NotNil(n)
	assert.NoError(err)
	// Neural net layers
	layers := n.Layers()
	acc := 0
	for _, layer := range layers[1:] {
		r, c := layer.Weights().Dims()
		acc += r * c
	}
	weights := make([]float64, acc)
	var netWeights []float64
	err = setNetWeights(layers[1:], weights)
	assert.NoError(err)
	for i := range layers[1:] {
		netWeights = append(netWeights, matrix.Mx2Vec(layers[i+1].Weights(), false)...)
	}
	assert.Equal(weights, netWeights)
	// incorrect length of weights
	weights = make([]float64, 5)
	err = setNetWeights(layers[1:], weights)
	assert.Error(err)
}
