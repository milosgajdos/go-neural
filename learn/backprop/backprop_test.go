package backprop

import (
	"log"
	"os"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/neural"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
	"github.com/stretchr/testify/assert"
)

var (
	net       *neural.Network
	inMx      *mat64.Dense
	labelsVec *mat64.Vector
)

func setup() {
	// Create test features matrix
	features := []float64{5.1, 3.5, 1.4, 0.1,
		4.9, 3.0, 1.4, 0.2,
		4.7, 3.2, 1.3, 0.3,
		4.6, 3.1, 1.5, 0.4,
		5.0, 3.6, 1.4, 0.5}
	inMx = mat64.NewDense(5, 4, features)
	labels := []float64{2.0, 1.0, 3.0, 2.0, 4.0}
	labelsVec = mat64.NewVector(len(labels), labels)
	// create test network
	_, inCols := inMx.Dims()
	hiddenLayers := []int{5}
	na := &neural.NetworkArch{Input: inCols, Hidden: hiddenLayers, Output: len(labels)}
	var err error
	net, err = neural.NewNetwork(neural.FEEDFWD, na)
	if err != nil {
		log.Fatal(err)
	}
}

func TestMain(m *testing.M) {
	setup()
	retCode := m.Run()
	os.Exit(retCode)
}

func TestCost(t *testing.T) {
	assert := assert.New(t)
	// create test config without any weights
	c := &Config{Weights: nil, Lambda: 1.0, Labels: 5}
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
	// Can't calculate cost for nil matrix
	cost, err = Cost(net, c, nil, labelsVec)
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
	c := &Config{Weights: nil, Lambda: 1.0, Labels: 5}
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
