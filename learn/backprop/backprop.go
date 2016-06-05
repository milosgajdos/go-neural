package backprop

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/neural"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
)

// Config allows to supply back propagation learning parameters
type Config struct {
	// Weights contain neural network weights
	// rolled into slice
	Weights []float64
	// Lambda is a cost regularization parameter
	Lambda float64
	// Labels number of classifications labels
	Labels int
}

// Cost calculates cost of the objective function cost for a particular network and parameters
// It returns a single value or fails with error.
// Underneath it implements the following objective function:
// J = -(sum(sum((out_k .* log(out) + (1 - out_k) .* log(1 - out)), 2)))/samples
func Cost(n *neural.Network, c *Config, inMx *mat64.Dense, expOut *mat64.Vector) (float64, error) {
	if inMx == nil || expOut == nil {
		return -1.0, fmt.Errorf("Cant calculate cost for In: %v Out: %v\n", inMx, expOut)
	}
	layers := n.Layers()
	// if we supply network weights, set the neural network to given weights
	if c.Weights != nil {
		if err := setNetWeights(layers[1:], c.Weights); err != nil {
			return -1.0, err
		}
	}
	// number of data samples
	samples, _ := inMx.Dims()
	// run forward propagation from INPUT layer
	netOut, err := n.ForwardProp(inMx, len(layers)-1)
	if err != nil {
		return -1.0, err
	}
	netOutMx := netOut.(*mat64.Dense)
	// labelsMx is one-of-N matrix for each output label
	// i.e. 3rd label would be: 0 0 1 0 0 etc.
	labelsMx, err := matrix.MakeLabelsMx(expOut, c.Labels)
	if err != nil {
		return -1.0, err
	}
	// out_k .* log(out)
	costMxA := new(mat64.Dense)
	costMxA.Apply(matrix.LogMx, netOutMx)
	costMxA.MulElem(labelsMx, costMxA)
	// (1 - out_k) .* log(1 - out)
	costMxB := new(mat64.Dense)
	labelsMx.Apply(matrix.SubtrMx(1.0), labelsMx)
	netOutMx.Apply(matrix.SubtrMx(1.0), netOutMx)
	netOutMx.Apply(matrix.LogMx, netOutMx)
	costMxB.MulElem(labelsMx, netOutMx)
	// Cost matrix
	costMxB.Add(costMxA, costMxB)
	// cost value
	cost := -(mat64.Sum(costMxB) / float64(samples))
	reg, err := CostReg(n, c.Lambda, samples)
	if err != nil {
		return -1.0, err
	}
	return cost + reg, nil
}

// CostReg calculates regularization cost for a particular network and parameters
// It returns a single value. Underneathe it implements the following function:
// (lambda/(2*samples))*(sum(sum(Theta_i(:,2:end).^2)) + ........
func CostReg(n *neural.Network, lambda float64, samples int) (float64, error) {
	// lambda or samples must be positive numbers
	if lambda < 0 || samples <= 0 {
		return -1.0, fmt.Errorf("Lambda and samples must be positive numbers")
	}
	reg := 0.0
	// calculate regularizer
	if lambda > 0 {
		layers := n.Layers()
		// Ignore first layer i.e. input layer
		for _, layer := range layers[1:] {
			r, c := layer.Weights().Dims()
			// Don't penalize bias units
			weightsMx := layer.Weights().View(0, 1, r, c-1)
			sqrMx := new(mat64.Dense)
			sqrMx.Apply(matrix.PowMx(2), weightsMx)
			reg += (lambda / (2 * float64(samples))) / mat64.Sum(sqrMx)
		}
	}
	return reg, nil
}

// Grad calculates network gradient for a particular network and configuration
// It returns a gradient slice or fails with error
func Grad(n *neural.Network, c *Config, inMx *mat64.Dense, expOut *mat64.Vector) ([]float64, error) {
	// get all network layers
	layers := n.Layers()
	// input and expected output must not be nil
	if inMx == nil || expOut == nil {
		return nil, fmt.Errorf("Cant calculate gradient for In: %v Out: %v\n", inMx, expOut)
	}
	// if we supply network weights, set the neural network to given weights
	if c.Weights != nil {
		if err := setNetWeights(layers[1:], c.Weights); err != nil {
			return nil, err
		}
	}
	// number of data samples
	samples, _ := inMx.Dims()
	labelsMx, err := matrix.MakeLabelsMx(expOut, c.Labels)
	if err != nil {
		return nil, err
	}
	// Do the full forward propagation
	out, err := n.ForwardProp(inMx, len(layers)-1)
	if err != nil {
		return nil, err
	}
	// iterate through all samples and calculate errors and corrections
	for i := 0; i < samples; i++ {
		// pick a sample
		inSample := inMx.RowView(i)
		// pick the expected output
		expVec := labelsMx.RowView(i)
		// pick actual output from output layer
		deltaVec := (out.(*mat64.Dense)).RowView(i)
		// calculate the error = out - y
		deltaVec.SubVec(deltaVec, expVec)
		// run the backpropagation
		if err := n.BackProp(inSample.T(), deltaVec.T(), len(layers)-1); err != nil {
			return nil, err
		}
	}
	// zero-th layer is INPUT layer and has no Deltas
	var gradient []float64
	for i := 1; i < len(layers); i++ {
		deltas := layers[i].Deltas()
		deltas.Scale(1/float64(samples), deltas)
		gradReg, err := GradReg(n, i, c.Lambda, samples)
		if err != nil {
			return nil, err
		}
		gradReg.Add(deltas, gradReg)
		gradVec := matrix.Mx2Vec(gradReg, false)
		gradient = append(gradient, gradVec...)
	}
	return gradient, nil
}

// GradReg calculates regularization cost of the gradient for a particular network and config.
func GradReg(n *neural.Network, layerIdx int, lambda float64, samples int) (*mat64.Dense, error) {
	layers := n.Layers()
	// 0-th layer has not weights, hence no gradient
	if layerIdx == 0 || layerIdx > len(layers)-1 {
		return nil, fmt.Errorf("Incorrect layer index supplied: %d\n", layerIdx)
	}
	// incorrect number of samples
	if samples <= 0 {
		return nil, fmt.Errorf("Incorrect number of samples supplied: %d\n", samples)
	}
	layer := layers[layerIdx]
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
	return regWeights, nil
}

// setNetWeights sets weights of all the requsted layers to values supplied via weights slice
func setNetWeights(layers []*neural.Layer, weights []float64) error {
	acc := 0
	wLen := len(weights)
	for _, layer := range layers {
		r, c := layer.Weights().Dims()
		if (wLen - acc) < r*c {
			return fmt.Errorf("Insufficient number of weights supplied %d\n", wLen)
		}
		err := matrix.SetMx2Vec(layer.Weights(), weights[acc:(acc+r*c)], false)
		if err != nil {
			return err
		}
		acc += r * c
	}
	return nil
}
