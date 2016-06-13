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

// ExpMx allows to calculate exponential of matrix elements
func ExpMx(i, j int, x float64) float64 {
	return math.Exp(x)
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

// TanhMx allows to apply tanh function to all matrix elements
func TanhMx(i, j int, x float64) float64 {
	return math.Tanh(x)
}

// TanhGradMx provides Tanh derivation used in backpropagation algorithm
func TanhGradMx(i, j int, x float64) float64 {
	return 1.0 - (math.Tanh(x) * math.Tanh(x))
}

// ReluMx allows to apply Relu to all matrix elements
func ReluMx(i, j int, x float64) float64 {
	if x > 0.0 {
		return x
	}
	return 0.0
}

// ReluGradMx provides Relu a "derlivation" used in backpropagation algorithm
func ReluGradMx(i, j int, x float64) float64 {
	if x > 0.0 {
		return 1.0
	}
	return 0.0
}
