package matrix

import "math"

// LogMx allows to calculate log of each matrix element
func LogMx(i, j int, x float64) float64 {
	return math.Log(x)
}

// SubtrMx allows to subtract a number from all matrix elements
func SubtrMx(f float64) func(int, int, float64) float64 {
	return func(i, j int, x float64) float64 {
		return f - x
	}
}

// AddMx allows to add an arbitrary number to all matrix elements
func AddMx(f float64) func(int, int, float64) float64 {
	return func(i, j int, x float64) float64 {
		return f + x
	}
}

// PowMx allows to calculate power of matrix elements
func PowMx(f float64) func(int, int, float64) float64 {
	return func(i, j int, x float64) float64 {
		return math.Pow(x, f)
	}
}

// SigmoidMx allows to apply sigmoid func to all matrix elements
func SigmoidMx(i, j int, x float64) float64 {
	return Sigmoid(x)
}

// SigmoidGradMx allows to apply Sigmoidd derivation func to all matrix elements
func SigmoidGradMx(i, j int, x float64) float64 {
	return SigmoidGrad(x)
}

// Sigmoid provides sigmoid activation function
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidGrad provides sigmoid derivation used in backprop algorithm
func SigmoidGrad(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}
