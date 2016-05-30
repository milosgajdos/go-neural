package dataset

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

// load data funcs
var loadFuncs = map[string]func(io.Reader) (*mat64.Dense, error){
	".csv": LoadCSV,
}

// Data represents training data set
type DataSet struct {
	mx mat64.Matrix
}

// NewData returns *Data or fails with error if either the path to data set
// supplied as a parameter does not exist or if the data set file is encoded
// in an unsupported file format. File format is inferred from the file extension.
func NewDataSet(path string) (*DataSet, error) {
	// Check if the training data file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, err
	}
	// Check if the supplied file type is supported
	fileType := filepath.Ext(path)
	loadData, ok := loadFuncs[fileType]
	if !ok {
		return nil, fmt.Errorf("Unsupported file type: %s\n", fileType)
	}
	// Open training data file
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	// Load file
	mx, err := loadData(file)
	if err != nil {
		return nil, err
	}
	// Return Data
	return &DataSet{
		mx: mx,
	}, nil
}

// Matrix returns the data set represented as matrix
func (ds DataSet) Data() mat64.Matrix {
	return ds.mx
}

// Scale centers the data set to zero mean values and scales each column.
// It modifies the data stored in the data set. If your data contains also
// labeles in the last column, make sure you extract it before scaling.
func (ds *DataSet) Scale() {
	rows, cols := ds.mx.Dims()
	// mean/stdev store each column mean/stdev values
	col := make([]float64, rows)
	mean := make([]float64, cols)
	stdev := make([]float64, cols)
	for i := 0; i < cols; i++ {
		// copy i-th column to col
		mat64.Col(col, i, ds.mx)
		mean[i], stdev[i] = stat.MeanStdDev(col, nil)
	}
	scale := func(i, j int, x float64) float64 {
		return (x - mean[j]) / stdev[j]
	}
	dataMx := ds.mx.(*mat64.Dense)
	dataMx.Apply(scale, dataMx)
}

// LoadCSV loads training set from the path supplied as a parameter.
// It returns data matrix that contains particular CSV fields in columns.
// It returns error if the supplied data set contains corrrupted data or
// if the data can not be converted to float numbers
func LoadCSV(r io.Reader) (*mat64.Dense, error) {
	// data matrix dimensions: rows x cols
	var rows, cols int
	// mxData contains ALL data read field by field
	mxData := make([]float64, 0)
	// create new CSV reader
	csvReader := csv.NewReader(r)
	// read all data record by record
	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		// allocate the dataRow during the first iteration
		if rows == 0 {
			// initialize cols on first iteration
			cols = len(record)
		}
		// number of columns is not the same as in the read record
		if cols != len(record) {
			// TODO: decide what to do when values are missing
			return nil, fmt.Errorf("Inconsistent number of features: %d\n", len(record))
		}
		// convert strings to floats
		for _, field := range record {
			// TODO: decide what to do when field can't be converted
			f, err := strconv.ParseFloat(field, 64)
			if err != nil {
				return nil, err
			}
			// append the read data into mxData
			mxData = append(mxData, f)
		}
		rows += 1
	}
	// Initialize data matrix with the read data
	mx := mat64.NewDense(rows, cols, mxData)
	return mx, nil
}

// ExtractFeatures extracts features and labels from data matrix
// It returns features matrix and vector of data labels
// It returns error if the features can not be extracted
func ExtractFeatures(mx mat64.Matrix) (*mat64.Dense, *mat64.Vector, error) {
	// get matrix dimensions
	rows, cols := mx.Dims()
	if cols == 1 {
		return nil, nil, fmt.Errorf("Insufficient number of columns: %d\n", cols)
	}
	// makes sure the passed in matrix is a Dense matrix
	dataMx, ok := mx.(*mat64.Dense)
	if !ok {
		return nil, nil, errors.New("Passed in matrix must be of type *mat64.Dense")
	}
	// last column contains labels
	labels := dataMx.ColView(cols - 1)
	// features are stored in all but the last column
	features := dataMx.View(0, 0, rows, cols-1)
	// return extracted data
	return features.(*mat64.Dense), labels, nil
}
