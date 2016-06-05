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
	if inMx == nil {
		return -1.0, fmt.Errorf("Cant calculate cost for %v matrix\n", inMx)
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
