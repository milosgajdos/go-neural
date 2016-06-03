package neural

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestNetworkKinds(t *testing.T) {
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
	// unknown network
	n, err := NewNetwork(NetworkKind(10000), new(NetworkArch))
	assert.Nil(n)
	assert.Error(err)
	// nil architecture
	n, err = NewNetwork(FEEDFWD, nil)
	assert.Nil(n)
	assert.Error(err)
	// zero size INPUT layer
	na := &NetworkArch{Input: 0, Hidden: nil, Output: 100}
	n, err = NewNetwork(FEEDFWD, na)
	assert.Nil(n)
	assert.Error(err)
	// negative output layer
	na = &NetworkArch{Input: 10, Hidden: nil, Output: -100}
	n, err = NewNetwork(FEEDFWD, na)
	assert.Nil(n)
	assert.Error(err)
	// correct architecture
	hidden := []int{20, 10}
	na = &NetworkArch{Input: 10, Hidden: hidden, Output: 10}
	n, err = NewNetwork(FEEDFWD, na)
	assert.NotNil(n)
	assert.NoError(err)
}

func TestID(t *testing.T) {
	assert := assert.New(t)
	// create dummy network
	hidden := []int{20, 10}
	na := &NetworkArch{Input: 10, Hidden: hidden, Output: 10}
	n, err := NewNetwork(FEEDFWD, na)
	assert.NotNil(n)
	assert.NoError(err)
	assert.Len(n.ID(), 10)
}

func TestKind(t *testing.T) {
	assert := assert.New(t)
	// create dummy network
	hidden := []int{20, 10}
	na := &NetworkArch{Input: 10, Hidden: hidden, Output: 10}
	n, err := NewNetwork(FEEDFWD, na)
	assert.NotNil(n)
	assert.NoError(err)
	assert.Equal(n.Kind(), FEEDFWD)
}

func TestLayers(t *testing.T) {
	assert := assert.New(t)
	// create dummy network
	hidden := []int{20, 10}
	na := &NetworkArch{Input: 10, Hidden: hidden, Output: 10}
	n, err := NewNetwork(FEEDFWD, na)
	assert.NotNil(n)
	assert.NoError(err)
	layers := n.Layers()
	assert.NotNil(layers)
	assert.Equal(len(layers), 4)
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
	// create test network
	inRows, inCols := inMx.Dims()
	hiddenLayers := []int{5}
	na := &NetworkArch{Input: inCols, Hidden: hiddenLayers, Output: 5}
	net, err := NewNetwork(FEEDFWD, na)
	assert.NotNil(net)
	assert.NoError(err)
	out, _ := net.ForwardProp(inMx, 0)
	assert.NotNil(out)
	outRows, outCols := out.Dims()
	assert.Equal(outRows, inRows)
	assert.Equal(outCols, na.Output)
	// Panics with error
	assert.Panics(func() { net.ForwardProp(nil, 0) })
}
