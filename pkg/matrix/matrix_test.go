package matrix

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestOnes(t *testing.T) {
	assert := assert.New(t)

	// Negative rows/cols are not allowed
	assert.Panics(func() { Ones(-4, 3) })

	// all elements must be equal to 1.0
	onesVec := []float64{1.0, 1.0, 1.0, 1.0}
	onesMx := mat64.NewDense(2, 2, onesVec)
	mx := Ones(2, 2)
	assert.NotNil(mx)
	assert.True(mat64.Equal(onesMx, mx))
}

func TestAddBias(t *testing.T) {
	assert := assert.New(t)

	// create new matrix and add bias
	tstMx := mat64.NewDense(2, 2, nil)
	biasMx := AddBias(tstMx)
	assert.NotNil(biasMx)
	assert.False(mat64.Equal(tstMx, biasMx))
	r, c := tstMx.Dims()
	rb, cb := biasMx.Dims()
	// bias matrix has one more column
	assert.True(cb == c+1)
	assert.True(rb == r)
	// check if the bias matrix first col is all 1.0s
	biasCol := biasMx.ColView(0)
	tstVec := mat64.NewDense(2, 1, []float64{1.0, 1.0})
	assert.True(mat64.Equal(tstVec, biasCol))
}

func TestMakeLabelsMx(t *testing.T) {
	assert := assert.New(t)

	// create new labels vector
	labCount := 2
	labels := []float64{1.0, 2.0, 1.0}
	labVec := mat64.NewVector(len(labels), labels)
	labMx, err := MakeLabelsMx(labVec, labCount)
	assert.NotNil(labMx)
	assert.NoError(err)
	r, c := labMx.Dims()
	assert.True(r == len(labels))
	assert.True(c == labCount)

	// this will fail with error
	labCount = -2
	labVec = mat64.NewVector(len(labels), nil)
	labMx, err = MakeLabelsMx(labVec, labCount)
	assert.Nil(labMx)
	assert.Error(err)
}

func TestMakeRandMx(t *testing.T) {
	assert := assert.New(t)

	// create new matrix
	rows, cols := 2, 3
	min, max := 0.0, 1.0
	randMx, err := MakeRandMx(rows, cols, min, max)
	assert.NotNil(randMx)
	assert.NoError(err)
	r, c := randMx.Dims()
	assert.True(r == rows)
	assert.True(c == cols)
	for i := 0; i < c; i++ {
		col := randMx.ColView(i)
		assert.True(max >= mat64.Max(col))
	}

	// Can't create new matrix
	randMx, err = MakeRandMx(rows, -6, min, max)
	assert.Nil(randMx)
	assert.Error(err)
}

func TestMx2Vec(t *testing.T) {
	assert := assert.New(t)

	// expected outputs
	byRow := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	byCol := []float64{1.2, 4.5, 8.9, 3.4, 6.7, 10.0}
	// NewDense creates new matrix by rows
	tstMx := mat64.NewDense(3, 2, byRow)
	// Check if the marix is rolled into slice by row
	rowVec := Mx2Vec(tstMx, true)
	assert.NotNil(rowVec)
	assert.EqualValues(rowVec, byRow)

	// Check if the marix is rolled into slice by col
	colVec := Mx2Vec(tstMx, false)
	assert.NotNil(colVec)
	assert.EqualValues(colVec, byCol)
}

func TestSetMx2Vec(t *testing.T) {
	assert := assert.New(t)

	// expected results
	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	mx := mat64.NewDense(3, 2, nil)
	assert.NotNil(mx)

	// Set matrix by row
	err := SetMx2Vec(mx, data, true)
	rowMx := mat64.NewDense(3, 2, data)
	assert.NoError(err)
	assert.NotNil(rowMx)
	assert.True(mat64.Equal(mx, rowMx))

	// Set matrix by col
	err = SetMx2Vec(mx, data, false)
	colData := []float64{1.2, 6.7, 3.4, 8.9, 4.5, 10.0}
	colMx := mat64.NewDense(3, 2, colData)
	assert.NoError(err)
	assert.NotNil(colMx)
	assert.True(mat64.Equal(mx, colMx))

	// Vector is smaller than number of matrix elements
	shortVec := []float64{1.3, 2.4}
	err = SetMx2Vec(mx, shortVec, true)
	assert.Error(err)
}

func TestRowColMax(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	rowsMax := []float64{3.4, 6.7, 10.0}
	colsMax := []float64{8.9, 10.0}
	mx := mat64.NewDense(3, 2, data)
	assert.NotNil(mx)
	// check rows
	tstRowsMax := RowsMax(mx)
	assert.NotNil(tstRowsMax)
	assert.EqualValues(rowsMax, tstRowsMax)
	// check cols
	tstColsMax := ColsMax(mx)
	assert.NotNil(tstColsMax)
	assert.EqualValues(colsMax, tstColsMax)
	// should get nil back
	tst := RowsMax(nil)
	assert.Nil(tst)
	tst = ColsMax(nil)
	assert.Nil(tst)

}

func TestRowColSums(t *testing.T) {
	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	rowSums := []float64{4.6, 11.2, 18.9}
	colSums := []float64{14.6, 20.1}
	delta := 0.001
	mx := mat64.NewDense(3, 2, data)
	assert.NotNil(t, mx)
	// check rows
	tstRowSums := RowSums(mx)
	assert.NotNil(t, tstRowSums)
	assert.InDeltaSlice(t, rowSums, tstRowSums, delta)
	// check cols
	tstColSums := ColSums(mx)
	assert.NotNil(t, tstColSums)
	assert.InDeltaSlice(t, colSums, tstColSums, delta)
	// should get nil back
	tst := RowSums(nil)
	assert.Nil(t, tst)
	tst = ColSums(nil)
	assert.Nil(t, tst)
}
