package neural

import (
	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/pkg/matrix"
)

// Cost is neural network training cost
type Cost interface {
	// CostFunc defines neural network cost function for given input, output and labels.
	// It returns a single number: cost for given input and output
	CostFunc(mat64.Matrix, mat64.Matrix, mat64.Matrix) float64
	// Delta implements function that calculates error in the last network layer
	// It returns the output error matrix
	Delta(mat64.Matrix, mat64.Matrix) mat64.Matrix
}

// CrossEntropy implements Cost interface
type CrossEntropy struct{}

// CostFunc implements cross entropy cost function.
// C = -(sum(sum((out_k .* log(out) + (1 - out_k) .* log(1 - out)), 2)))/samples
func (c CrossEntropy) CostFunc(inMx, outMx, labelsMx mat64.Matrix) float64 {
	// safe switch type as matrix.MakeLabelsMx returns *mat64.Dense
	lMx := labelsMx.(*mat64.Dense)
	oMx := outMx.(*mat64.Dense)
	// out_k .* log(out)
	costMxA := new(mat64.Dense)
	costMxA.Apply(matrix.LogMx, oMx)
	costMxA.MulElem(lMx, costMxA)
	// (1 - out_k) .* log(1 - out)
	costMxB := new(mat64.Dense)
	lMx.Apply(matrix.SubtrMx(1.0), lMx)
	oMx.Apply(matrix.SubtrMx(1.0), oMx)
	oMx.Apply(matrix.LogMx, oMx)
	costMxB.MulElem(labelsMx, oMx)
	// Cost matrix
	costMxB.Add(costMxA, costMxB)
	// calculate the cost
	samples, _ := inMx.Dims()
	cost := -(mat64.Sum(costMxB) / float64(samples))
	return cost
}

// Delta calculates the error of the last layer and returns it
// D = (out_k - out)
func (c CrossEntropy) Delta(outMx, expMx mat64.Matrix) mat64.Matrix {
	deltaMx := new(mat64.Dense)
	deltaMx.Sub(outMx, expMx)
	return deltaMx
}

// LogLikelihood implements Cost interface
type LogLikelihood struct{}

// CostFunc implements log-likelihood cost function.
// C = -sum(sum(out_k.*log(out)))
func (c LogLikelihood) CostFunc(inMx, outMx, labelsMx mat64.Matrix) float64 {
	// safe switch type as matrix.MakeLabelsMx returns *mat64.Dense
	lMx := labelsMx.(*mat64.Dense)
	oMx := outMx.(*mat64.Dense)
	// out_k .* log(out)
	costMx := new(mat64.Dense)
	costMx.Apply(matrix.LogMx, oMx)
	costMx.MulElem(lMx, costMx)
	// calculate the cost
	samples, _ := inMx.Dims()
	cost := (-mat64.Sum(costMx) / float64(samples))
	return cost
}

// Delta calculates the error of the last layer and returns it
// D = (out_k - out)
func (c LogLikelihood) Delta(outMx, expMx mat64.Matrix) mat64.Matrix {
	deltaMx := new(mat64.Dense)
	deltaMx.Sub(outMx, expMx)
	return deltaMx
}
