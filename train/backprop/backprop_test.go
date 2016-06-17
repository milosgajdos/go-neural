package backprop

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/neural"
	"github.com/milosgajdos83/go-neural/pkg/config"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
	"github.com/stretchr/testify/assert"
)

var (
	fileName  = "manifest.yml"
	net       *neural.Network
	inMx      *mat64.Dense
	labelsVec *mat64.Vector
)

func setup() {
	content := []byte(`kind: feedfwd
task: class
layers:
  input:
    size: 400
  hidden:
    size: [5]
    activation: sigmoid
  output:
    size: 10
    activation: softmax
training:
  kind: backprop
  params: "lambda=1.0"
optimize:
  method: bfgs
  iterations: 69`)
	tmpPath := filepath.Join(os.TempDir(), fileName)
	if err := ioutil.WriteFile(tmpPath, content, 0666); err != nil {
		log.Fatal(err)
	}
	// Create test features matrix
	features := []float64{5.1, 3.5, 1.4, 0.1,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.3,
		4.6, 3.1, 1.5, 0.4,
		5.0, 3.6, 1.4, 0.5}
	inMx = mat64.NewDense(5, 4, features)
	_, inCols := inMx.Dims()
	labels := []float64{2.0, 1.0, 3.0, 2.0, 4.0}
	labelsVec = mat64.NewVector(len(labels), labels)
	// basic configuration settings
	c, err := config.NewNetConfig(tmpPath)
	if err != nil {
		log.Fatal(err)
	}
	// set config to test case data
	c.Arch.Input.Size = inCols
	c.Arch.Hidden[0].Size = 5
	c.Arch.Output.Size = len(labels)
	net, err = neural.NewNetwork(c)
	if err != nil {
		log.Fatal(err)
	}
}

func teardown() {
	os.Remove(filepath.Join(os.TempDir(), fileName))
}

func TestMain(m *testing.M) {
	setup()
	retCode := m.Run()
	os.Exit(retCode)
}

func TestValidateConfig(t *testing.T) {
	assert := assert.New(t)
	// start with correct config
	c := &Config{Weights: nil, Optim: "bfgs", Lambda: 1.0, Labels: 10, Iters: 50}
	err := ValidateConfig(c)
	assert.NoError(err)
	// config can't be nil
	err = ValidateConfig(nil)
	assert.Error(err)
	// unsupported Optimization method
	c.Optim = "foobar"
	err = ValidateConfig(c)
	assert.Error(err)
	c.Optim = "bfgs"
	// wrong number of labels
	c.Labels = 0
	err = ValidateConfig(c)
	assert.Error(err)
	// Wrong number of iterations
	c.Labels, c.Iters = 10, -10
	err = ValidateConfig(c)
	assert.Error(err)
	// incorrect lambda
	c.Lambda, c.Iters = -10.0, 50
	err = ValidateConfig(c)
	assert.Error(err)
}

func TestTrain(t *testing.T) {
	assert := assert.New(t)
	// create test config without any weights
	c := &Config{Weights: nil, Optim: "bfgs", Lambda: 1.0, Labels: 5, Iters: 2}
	err := Train(net, c, inMx, labelsVec)
	assert.NoError(err)
	// nil input causes error
	err = Train(net, c, nil, labelsVec)
	assert.Error(err)
	// nil labelsVec causes error
	err = Train(net, c, inMx, nil)
	assert.Error(err)
	// bogus configuration causes error
	c.Lambda = -100.0
	err = Train(net, c, inMx, labelsVec)
	assert.Error(err)
}

func TestCost(t *testing.T) {
	assert := assert.New(t)
	// create test config without any weights
	c := &Config{Weights: nil, Optim: "bfgs", Lambda: 1.0, Labels: 5, Iters: 50}
	cost, err := Cost(net, c, inMx, labelsVec)
	assert.NoError(err)
	assert.True(cost > 0)
	// allocate weights
	var weights []float64
	layers := net.Layers()
	for i := range layers[1:] {
		weights = append(weights, matrix.Mx2Vec(layers[i+1].Weights(), false)...)
	}
	c.Weights = weights
	cost, err = Cost(net, c, inMx, labelsVec)
	assert.NoError(err)
	assert.True(cost > 0)
	// negative number of labels
	c.Labels = -100
	cost, err = Cost(net, c, inMx, labelsVec)
	assert.True(cost == -1.0)
	assert.Error(err)
	c.Labels = 5
	// Can't calculate cost for nil matrix
	cost, err = Cost(net, c, nil, labelsVec)
	assert.True(cost == -1.0)
	assert.Error(err)
	cost, err = Cost(net, c, inMx, nil)
	assert.True(cost == -1.0)
	assert.Error(err)
	// Incorrect matrix dimensions
	tstMx := mat64.NewDense(100, 100, nil)
	cost, err = Cost(net, c, tstMx, labelsVec)
	assert.True(cost == -1.0)
	assert.Error(err)
}

func TestCostReg(t *testing.T) {
	assert := assert.New(t)
	// if lambda is 0.0, regularizer is 0.0
	reg, err := CostReg(net, 0.0, 1000)
	assert.NoError(err)
	assert.Equal(reg, 0.0)
	// if lambda is > 0.0 regularizer must be positive number
	reg, err = CostReg(net, 10.0, 100)
	assert.True(reg > 0)
	assert.NoError(err)
	// lambda and samples must be positive numbers
	reg, err = CostReg(net, -10.0, 100)
	assert.Error(err)
	assert.True(reg < 0)
	reg, err = CostReg(net, 10.0, -100)
	assert.Error(err)
	assert.True(reg < 0)
}

func TestGrad(t *testing.T) {
	assert := assert.New(t)
	// create test config without any weights
	c := &Config{Weights: nil, Optim: "bfgs", Lambda: 1.0, Labels: 5, Iters: 50}
	grad, err := Grad(net, c, inMx, labelsVec)
	assert.NoError(err)
	assert.NotNil(grad)
	layers := net.Layers()
	var gradLen int
	for _, layer := range layers[1:] {
		r, c := layer.Weights().Dims()
		gradLen += r * c
	}
	assert.Equal(gradLen, len(grad))
	// allocate weights
	var weights []float64
	for i := range layers[1:] {
		weights = append(weights, matrix.Mx2Vec(layers[i+1].Weights(), false)...)
	}
	c.Weights = weights
	grad, err = Grad(net, c, inMx, labelsVec)
	assert.NoError(err)
	assert.NotNil(grad)
	// nil inMx causes error
	grad, err = Grad(net, c, nil, labelsVec)
	assert.Error(err)
	assert.Nil(grad)
	// expected labels are borked
	c.Labels = -100
	grad, err = Grad(net, c, inMx, labelsVec)
	assert.Error(err)
	assert.Nil(grad)
	c.Labels = 5
	// non-sense input matrix
	tstMx := mat64.NewDense(100, 20, nil)
	assert.NotNil(tstMx)
	grad, err = Grad(net, c, tstMx, labelsVec)
	assert.Error(err)
	assert.Nil(grad)
}

func TestGradReg(t *testing.T) {
	assert := assert.New(t)
	// if lambda is 0.0, regularizer is 0.0
	reg, err := GradReg(net, 1, 1.0, 1000)
	assert.NoError(err)
	assert.NotNil(reg)
	layers := net.Layers()
	layRows, layCols := layers[1].Weights().Dims()
	regRows, regCols := reg.Dims()
	assert.Equal(layRows, regRows)
	assert.Equal(layCols, regCols)
	// incorrect index
	reg, err = GradReg(net, 0, 1.0, 1000)
	assert.Error(err)
	assert.Nil(reg)
	// incorrect number of samples
	reg, err = GradReg(net, 1, 1.0, -1000)
	assert.Error(err)
	assert.Nil(reg)
	// zero lambda returns zero matrix
	tstMx := mat64.NewDense(layRows, layCols, nil)
	assert.NotNil(tstMx)
	reg, err = GradReg(net, 1, 0.0, 10)
	assert.NoError(err)
	assert.NotNil(reg)
	assert.Equal(tstMx, reg)
}

func TestSetNetWeights(t *testing.T) {
	assert := assert.New(t)
	// Neural net layers
	layers := net.Layers()
	acc := 0
	for _, layer := range layers[1:] {
		r, c := layer.Weights().Dims()
		acc += r * c
	}
	weights := make([]float64, acc)
	var netWeights []float64
	err := setNetWeights(layers[1:], weights)
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
