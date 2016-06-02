package neural

import (
	"testing"

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
