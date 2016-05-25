package main

import "math"

// logFunc will helps us calculate log of each matrix element
func LogMx(i, j int, x float64) float64 {
	return math.Log(x)
}

// subtOneFunc will help us subtract matrix elements from 1.0
func SubtrMx(i, j int, x float64) float64 {
	return 1.0 - x
}

// PowerMx provides can be used to calculate power of matrix elements
func PowerMx(i, j int, x float64) float64 {
	return x * x
}

// Sigmoid activation function
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Sigmoid derivation used in backprop algorithm
func SigmoidGrad(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}
