package neural

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/pkg/config"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
	"github.com/stretchr/testify/assert"
)

func TestLayerKind(t *testing.T) {
	assert := assert.New(t)
	layerKinds := []struct {
		k   LayerKind
		out string
	}{
		{INPUT, "INPUT"},
		{HIDDEN, "HIDDEN"},
		{OUTPUT, "OUTPUT"},
		{LayerKind(1000), "UNKNOWN"},
	}

	for _, layerKind := range layerKinds {
		assert.Equal(layerKind.k.String(), layerKind.out)
	}
}

func TestNewLayer(t *testing.T) {
	assert := assert.New(t)

	c := &config.LayerConfig{
		Kind: "input",
		Size: 10,
		NeurFn: &config.NeuronConfig{
			ActFn:     matrix.SigmoidMx,
			ActGradFn: matrix.SigmoidGradMx,
		},
	}
	// init naiive network
	net := new(Network)
	net.id = "testid"

	// invalid layer parameters passed in
	tstLayer, err := NewLayer(net, c, -10)
	assert.Nil(tstLayer)
	assert.Error(err)
	// invalid layer parameters passed in
	c.Size = -10
	tstLayer, err = NewLayer(net, c, 10)
	assert.Nil(tstLayer)
	assert.Error(err)
	c.Size = 10
	// nil network
	tstLayer, err = NewLayer(nil, c, 10)
	assert.Nil(tstLayer)
	assert.Error(err)
	// no network id
	net.id = ""
	tstLayer, err = NewLayer(net, c, 10)
	assert.Nil(tstLayer)
	assert.Error(err)
	net.id = "testid"
	// UNKNOWN layer
	c.Kind = "foobar"
	tstLayer, err = NewLayer(net, c, 10)
	assert.Nil(tstLayer)
	assert.Error(err)
	c.Kind = "input"
	// correct cases
	lKinds := []string{"input", "hidden", "output"}
	for _, lKind := range lKinds {
		c.Kind = lKind
		tstLayer, err := NewLayer(net, c, 10)
		assert.NotNil(tstLayer)
		assert.NoError(err)
	}
}

func TestIDAndKind(t *testing.T) {
	assert := assert.New(t)

	// layer config
	c := &config.LayerConfig{
		Kind: "input",
		Size: 10,
		NeurFn: &config.NeuronConfig{
			ActFn:     matrix.SigmoidMx,
			ActGradFn: matrix.SigmoidGradMx,
		},
	}
	//init naiive network
	net := new(Network)
	net.id = "testid"

	// create test network layer
	lID := ""
	lKinds := []string{"input", "hidden", "output"}
	for _, lKind := range lKinds {
		c.Kind = lKind
		tstLayer, err := NewLayer(net, c, 10)
		assert.NotNil(tstLayer)
		assert.NoError(err)
		// id can't be empty
		assert.True(tstLayer.ID() != "")
		// layers can't have identical Ids
		assert.True(tstLayer.ID() != lID)
		assert.Equal(tstLayer.Kind(), layerKind[lKind])
		lID = tstLayer.ID()
	}
}

func TestSetWeights(t *testing.T) {
	assert := assert.New(t)

	c := &config.LayerConfig{
		Kind: "input",
		Size: 20,
		NeurFn: &config.NeuronConfig{
			ActFn:     matrix.SigmoidMx,
			ActGradFn: matrix.SigmoidGradMx,
		},
	}
	//init naiive network
	net := new(Network)
	net.id = "testid"

	// INPUT layer does not have any weights
	tstLayer, err := NewLayer(net, c, 10)
	assert.NotNil(tstLayer)
	assert.NoError(err)
	weights := mat64.NewDense(100, 200, nil)
	err = tstLayer.SetWeights(weights)
	assert.Error(err)
	// INPUT layer has no weights or deltas
	inW, inD := tstLayer.Weights(), tstLayer.Deltas()
	assert.Nil(inW)
	assert.Nil(inD)

	// HIDDEN layer
	c.Kind = "hidden"
	tstLayer, err = NewLayer(net, c, 10)
	assert.NotNil(tstLayer)
	assert.NoError(err)
	// can't set layers to nil
	err = tstLayer.SetWeights(nil)
	assert.Error(err)

	//OUTPUT layer wrong dimensions
	tstLayer, err = NewLayer(net, c, 10)
	assert.NotNil(tstLayer)
	assert.NoError(err)
	wRows, wCols := 20, 1000
	weights = mat64.NewDense(wRows, wCols, nil)
	err = tstLayer.SetWeights(weights)
	assert.Error(err)

	// OUTPUT layer correct dimensions
	tstLayer, err = NewLayer(net, c, 10)
	assert.NotNil(tstLayer)
	assert.NoError(err)
	wRows, wCols = 20, 11
	weights = mat64.NewDense(wRows, wCols, nil)
	err = tstLayer.SetWeights(weights)
	assert.NoError(err)
	// check the deltas and weights dimensions
	twRows, twCols := tstLayer.Weights().Dims()
	tdRows, tdCols := tstLayer.Deltas().Dims()
	assert.Equal(twRows, wRows)
	assert.Equal(twCols, wCols)
	assert.Equal(tdRows, wRows)
	assert.Equal(tdCols, wCols)
}

func TestOut(t *testing.T) {
	assert := assert.New(t)

	c := &config.LayerConfig{
		Kind: "input",
		Size: 10,
		NeurFn: &config.NeuronConfig{
			ActFn:     matrix.SigmoidMx,
			ActGradFn: matrix.SigmoidGradMx,
		},
	}
	// dummy network for testing
	net := new(Network)
	net.id = "testid"
	// Layer parameters
	layerIn, layerOut := 2, 2
	c.Size = layerOut
	inputLayer, err := NewLayer(net, c, layerIn)
	assert.NotNil(inputLayer)
	assert.NoError(err)

	// Correct dimension matrix
	data := []float64{1.0, 1.0, 2.0, 2.0, 3.0, 3.0}
	corrInMx := mat64.NewDense(layerIn+1, layerOut, data)
	assert.NotNil(corrInMx)

	// nil input yields nil output
	out, err := inputLayer.Out(nil)
	assert.Nil(out)
	assert.Error(err)
	// INPUT layer proxies the input to output
	out, err = inputLayer.Out(corrInMx)
	assert.NotNil(out)
	assert.NoError(err)
	assert.True(mat64.Equal(corrInMx, out))

	// HIDDEN layer test
	c.Kind = "hidden"
	hiddenLayer, err := NewLayer(net, c, layerIn)
	assert.NotNil(hiddenLayer)
	assert.NoError(err)
	// mismatched dimension
	mismData := []float64{3.0, 4.0, 1.0}
	mismInMx := mat64.NewDense(1, 3, mismData)
	out, err = hiddenLayer.Out(mismInMx)
	assert.Nil(out)
	assert.Error(err)
	// correct data dimension must yield the following result
	dataOut := []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	expOut := mat64.NewDense(layerIn+1, layerOut, dataOut)
	// testing weights
	weightsData := []float64{2.0, 3.0, 4.0, 5.0, 6.0, 7.0}
	weights := mat64.NewDense(layerOut, layerIn+1, weightsData)
	err = hiddenLayer.SetWeights(weights)
	assert.NoError(err)
	// compute output
	out, err = hiddenLayer.Out(corrInMx)
	assert.NotNil(out)
	assert.NoError(err)
	assert.True(mat64.EqualApprox(out, expOut, 0.001))
}
