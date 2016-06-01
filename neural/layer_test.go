package neural

import (
	"testing"

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
	// init naiive network
	net := new(Network)
	// define helper types
	type dims struct {
		rows int
		cols int
	}
	type expected struct {
		kind  LayerKind
		wdims dims
		ddims dims
	}
	// run test
	tstLayers := []struct {
		kind LayerKind
		net  *Network
		in   int
		out  int
		exp  expected
	}{
		{HIDDEN, net, 10, 20, expected{HIDDEN, dims{20, 11}, dims{20, 11}}},
		{OUTPUT, net, 100, 200, expected{OUTPUT, dims{200, 101}, dims{200, 101}}},
		{LayerKind(1000), net, 10, 10, expected{}},
		{HIDDEN, net, 10, 20, expected{HIDDEN, dims{20, 11}, dims{20, 11}}},
		{INPUT, net, 10, 30, expected{INPUT, dims{30, 11}, dims{30, 11}}},
		{HIDDEN, nil, 100, 10, expected{}},
		{INPUT, net, -10, 20, expected{}},
	}

	for _, l := range tstLayers {
		tstLayer, err := NewLayer(l.kind, l.net, l.in, l.out)
		if err != nil {
			assert.Nil(tstLayer)
		} else {
			assert.NotNil(tstLayer)
			// check IDs
			id := tstLayer.ID()
			assert.Len(id, 10)
			kind := tstLayer.Kind()
			assert.Equal(kind, l.exp.kind)
			// check weights
			if kind == INPUT {
				assert.Nil(tstLayer.Weights())
				assert.Nil(tstLayer.Deltas())
			} else {
				weights := tstLayer.Weights()
				assert.NotNil(weights)
				r, c := weights.Dims()
				assert.Equal(r, l.exp.wdims.rows)
				assert.Equal(c, l.exp.wdims.cols)
				// check deltas
				deltas := tstLayer.Deltas()
				assert.NotNil(deltas)
				r, c = deltas.Dims()
				assert.Equal(r, l.exp.ddims.rows)
				assert.Equal(c, l.exp.ddims.cols)
			}
		}
	}
}
