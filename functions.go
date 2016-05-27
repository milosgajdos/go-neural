package main

import "math"

// logFunc will helps us calculate log of each matrix element
func LogMx(i, j int, x float64) float64 {
	return math.Log(x)
}

// SubtrMx allows to subtract a number from all elements of matrix
func SubtrMx(f float64) func(int, int, float64) float64 {
	return func(i, j int, x float64) float64 {
		return f - x
	}
}

// PowMx can be used to calculate power of matrix elements
func PowMx(f float64) func(int, int, float64) float64 {
	return func(i, j int, x float64) float64 {
		return math.Pow(x, f)
	}
}

// Sigmoid activation function
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Sigmoid derivation used in backprop algorithm
func SigmoidGrad(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}
