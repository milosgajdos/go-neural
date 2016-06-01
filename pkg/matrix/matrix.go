package matrix

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

// Ones returns a matrix of rows x cols filled with 1.0
// It returns error if the supplied number of rows or columns are not positive integers
func Ones(rows, cols int) (*mat64.Dense, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("Incorrect dimensions supplied: %d x %dd\n", rows, cols)
	}
	// allocate zero matrix and set every element to 1.0
	onesMx := mat64.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			onesMx.Set(i, j, 1.0)
		}
	}
	return onesMx, nil
}

// AddBias adds a bias unit (either a vector or a single unit) to mat64.Matrix
// and returns the new augmented matrix without modifying the original one
// It returns error if the bias matrix could not be created
func AddBias(m mat64.Matrix) (*mat64.Dense, error) {
	rows, cols := m.Dims()
	// bias is a 1-column matrix that contains 1.0s
	bias, err := Ones(rows, 1)
	if err != nil {
		return nil, err
	}
	// allocate the new augmented bias matrix
	biasMx := mat64.NewDense(rows, cols+1, nil)
	biasMx.Augment(bias, m)
	return biasMx, nil
}

// MakeLabelsMx creates a 1-of-N matrix from the supplied vector of labels
// Labels matrix has the following dimensions: labels.Len() x count
// It does not modify the supplied matrix of labels.
// It returns error if the number of labels is negative integer
func MakeLabelsMx(labels *mat64.Vector, labCount int) (*mat64.Dense, error) {
	if labCount < 0 {
		return nil, fmt.Errorf("Incorrect number of labels: %d\n", labCount)
	}
	// get number of samples
	samples := labels.Len()
	// create labels matrix
	mx := mat64.NewDense(samples, labCount, nil)
	for i := 0; i < samples; i++ {
		val := labels.At(i, 0)
		mx.Set(i, int(val)-1, 1.0)
	}
	return mx, nil
}

// MakeRandMx creates a new matrix with of size rows x cols that is initialized
// to random number uniformly distributed in interval (min, max)
func MakeRandMx(rows, cols int, min, max float64) (*mat64.Dense, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("Incorrect dimensions supplied: %d x %dd\n", rows, cols)
	}
	// set random seed
	rand.Seed(55)
	// empirically this is supposed to be the best value
	epsilon := math.Sqrt(6.0) / math.Sqrt(float64(rows+cols))
	// allocate data slice
	randVals := make([]float64, rows*cols)
	for i := range randVals {
		// we need value between 0 and 1.0
		randVals[i] = rand.Float64()*(max-min) + min
		randVals[i] = randVals[i]*(2*epsilon) - epsilon
	}
	return mat64.NewDense(int(rows), int(cols), randVals), nil
}

// Mx2Vec unrolls all elements of matrix into a slice and returns it.
// Matrix elements can be unrolled either by row or by a column.
func Mx2Vec(m *mat64.Dense, byRow bool) []float64 {
	if byRow {
		return mx2VecByRow(m)
	}
	return mx2VecByCol(m)
}

// mx2VecByRow rolls matrix into a slice by rows
func mx2VecByRow(m *mat64.Dense) []float64 {
	rows, cols := m.Dims()
	vec := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		view := m.RowView(i)
		for j := 0; j < view.Len(); j++ {
			vec[i*cols+j] = view.At(j, 0)
		}
	}
	return vec
}

// mx2VecByCol rolls matrix into a slice by columns
func mx2VecByCol(m *mat64.Dense) []float64 {
	rows, cols := m.Dims()
	vec := make([]float64, rows*cols)
	for i := 0; i < cols; i++ {
		view := m.ColView(i)
		for j := 0; j < view.Len(); j++ {
			vec[i*rows+j] = view.At(j, 0)
		}
	}
	return vec
}

// SetMx2Vec sets all elements of a matrix to values stored in a slice
// passed in as a parameter. It fails with error if number of elements
// of the matrix is bigger than number of elements of the slice.
func SetMx2Vec(vec []float64, mx *mat64.Dense, byRow bool) (err error) {
	r, c := mx.Dims()
	if r*c > len(vec) {
		err = fmt.Errorf("Elements count mismatch: Vec: %d, Matrix: %d\n", len(vec), r*c)
		return
	}
	if byRow {
		setMx2VecByRow(vec, mx)
		return
	}
	setMx2VecByCol(vec, mx)
	return
}

// setMxByRowVec sets elements of mx from vec by rows
func setMx2VecByRow(vec []float64, mx *mat64.Dense) {
	rows, cols := mx.Dims()
	acc := 0
	for i := 0; i < rows; i++ {
		mx.SetRow(i, vec[acc:(acc+cols)])
		acc += cols
	}
}

// setMxByColVec sets elements of mx from vec by columns
func setMx2VecByCol(vec []float64, mx *mat64.Dense) {
	rows, cols := mx.Dims()
	acc := 0
	for i := 0; i < cols; i++ {
		mx.SetCol(i, vec[acc:(acc+rows)])
		acc += rows
	}
}

// RowsMax returns a slice of max values per each matrix row
// It returns nil if passed in matrix is nil or has zero elements
func RowsMax(m *mat64.Dense) []float64 {
	if m == nil {
		return nil
	}
	rows, _ := m.Dims()
	max := make([]float64, rows)
	for i := 0; i < rows; i++ {
		max[i] = mat64.Max(m.RowView(i))
	}
	return max
}

// ColsMax returns a slice of max values per each matrix column
// It returns nil if passed in matrix is nil or has zero elements
func ColsMax(m *mat64.Dense) []float64 {
	if m == nil {
		return nil
	}
	_, cols := m.Dims()
	max := make([]float64, cols)
	for i := 0; i < cols; i++ {
		max[i] = mat64.Max(m.ColView(i))
	}
	return max
}

// RowSums returns a slice of sums of all elemnts in each matrix row
// It returns nil if passed in matrix is nil or has zero elements
func RowSums(m *mat64.Dense) []float64 {
	if m == nil {
		return nil
	}
	rows, _ := m.Dims()
	sum := make([]float64, rows)
	for i := 0; i < rows; i++ {
		sum[i] = mat64.Sum(m.RowView(i))
	}
	return sum
}

// ColSums returns a slice of sums of all elemnts in each matrix column
// It returns nil if passed in matrix is nil or has zero elements
func ColSums(m *mat64.Dense) []float64 {
	if m == nil {
		return nil
	}
	_, cols := m.Dims()
	sum := make([]float64, cols)
	for i := 0; i < cols; i++ {
		sum[i] = mat64.Sum(m.ColView(i))
	}
	return sum
}
