package matrix

import (
	"math"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestLogMx(t *testing.T) {
	assert := assert.New(t)

	inData := []float64{1.0, 10, 20}
	inMx := mat64.NewDense(1, len(inData), inData)
	assert.NotNil(inMx)
	// test cases
	testCases := []struct {
		data     []float64
		expected bool
	}{
		{[]float64{0.0, 2.302585, 2.995732}, true},
		{[]float64{0.0, 0.1, 0.3}, false},
	}

	for _, tc := range testCases {
		tstMx := mat64.NewDense(1, len(tc.data), tc.data)
		assert.NotNil(tstMx)
		logMx := new(mat64.Dense)
		logMx.Apply(LogMx, inMx)
		assert.True(tc.expected == mat64.EqualApprox(logMx, tstMx, 0.001))
	}
}

func TestSubtrMx(t *testing.T) {
	assert := assert.New(t)

	inData := []float64{1.0, 2.0, 3.0}
	inMx := mat64.NewDense(1, len(inData), inData)
	assert.NotNil(inMx)
	// test cases
	testCases := []struct {
		data     []float64
		expected bool
	}{
		{[]float64{0.0, -1.0, -2.0}, true},
		{[]float64{0.0, 1.2, 0.1}, false},
	}

	for _, tc := range testCases {
		tstMx := mat64.NewDense(1, len(tc.data), tc.data)
		assert.NotNil(tstMx)
		sub1Mx := new(mat64.Dense)
		sub1Mx.Apply(SubtrMx(1), inMx)
		assert.True(tc.expected == mat64.Equal(sub1Mx, tstMx))
	}
}

func TestAddMx(t *testing.T) {
	assert := assert.New(t)

	inData := []float64{1.0, 2.0, 3.0}
	inMx := mat64.NewDense(1, len(inData), inData)
	assert.NotNil(inMx)
	// test cases
	testCases := []struct {
		data     []float64
		expected bool
	}{
		{[]float64{2.0, 3.0, 4.0}, true},
		{[]float64{0.0, 1.2, 0.1}, false},
	}

	for _, tc := range testCases {
		tstMx := mat64.NewDense(1, len(tc.data), tc.data)
		assert.NotNil(tstMx)
		add1Mx := new(mat64.Dense)
		add1Mx.Apply(AddMx(1), inMx)
		assert.True(tc.expected == mat64.Equal(add1Mx, tstMx))
	}
}

func TestPowMx(t *testing.T) {
	assert := assert.New(t)

	inData := []float64{1.0, 2.0, 3.0}
	inMx := mat64.NewDense(1, len(inData), inData)
	assert.NotNil(inMx)
	// test cases
	testCases := []struct {
		data     []float64
		expected bool
	}{
		{[]float64{1.0, 8.0, 27.0}, true},
		{[]float64{0.0, 1.2, 0.1}, false},
	}

	for _, tc := range testCases {
		tstMx := mat64.NewDense(1, len(tc.data), tc.data)
		assert.NotNil(tstMx)
		pow3Mx := new(mat64.Dense)
		pow3Mx.Apply(PowMx(3.0), inMx)
		assert.True(tc.expected == mat64.Equal(pow3Mx, tstMx))
	}
}

func TestSigmoidMx(t *testing.T) {
	assert := assert.New(t)

	inData := []float64{0.0, math.Pow(10.0, 6.0), -10000000}
	inMx := mat64.NewDense(1, len(inData), inData)
	assert.NotNil(inMx)
	// test cases
	testCases := []struct {
		data     []float64
		expected bool
	}{
		{[]float64{0.5, 1.0, 0.0}, true},
		{[]float64{0.0, 1.2, 0.1}, false},
	}

	for _, tc := range testCases {
		tstMx := mat64.NewDense(1, len(tc.data), tc.data)
		assert.NotNil(tstMx)
		sigMx := new(mat64.Dense)
		sigMx.Apply(SigmoidMx, inMx)
		assert.True(tc.expected == mat64.EqualApprox(sigMx, tstMx, 0.001))
	}
}
