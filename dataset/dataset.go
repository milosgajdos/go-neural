package dataset

import (
	"encoding/csv"
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

// DataSet represents training data set
type DataSet struct {
	mx      mat64.Matrix
	labeled bool
}

// NewDataSet returns *Data or fails with error if either the path to data set
// supplied as a parameter does not exist or if the data set file is encoded
// in an unsupported file format. File format is inferred from the file extension.
// You can specify if the data set is labeled or not
func NewDataSet(path string, labeled bool) (*DataSet, error) {
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
		mx:      mx,
		labeled: labeled,
	}, nil
}

// IsLabeled returns true if the loaded data set contains labels
// Labels are assumed to be in the last column of the data matrix
func (ds DataSet) IsLabeled() bool {
	return ds.labeled
}

// Data returns the data set represented as matrix
func (ds DataSet) Data() mat64.Matrix {
	return ds.mx
}

// Features returns features matrix from the underlying data matrix
// Data features are considered to be stored in all but the last column of
// the dataset matrix if the data set is labeled.
/// If the dataset is not labeled Features returns the raw data matrix
func (ds DataSet) Features() mat64.Matrix {
	if !(ds.labeled) {
		return ds.mx
	}
	// get matrix dimensions
	rows, cols := ds.mx.Dims()
	if cols == 1 {
		return ds.mx
	}
	// turn mat64.Matrix into mat64.Dense matrix
	dataMx := ds.mx.(*mat64.Dense)
	return dataMx.View(0, 0, rows, cols-1)
}

// Labels returns data labels from the raw data.
// If the data set is not labeled or if it only contains one columne it returns nil
func (ds DataSet) Labels() mat64.Matrix {
	if !(ds.labeled) {
		return nil
	}
	_, cols := ds.mx.Dims()
	if cols == 1 {
		return nil
	}
	dataMx := ds.mx.(*mat64.Dense)
	return dataMx.ColView(cols - 1)
}

// LoadCSV loads training set from the path supplied as a parameter.
// It returns data matrix that contains particular CSV fields in columns.
// It returns error if the supplied data set contains corrrupted data or
// if the data can not be converted to float numbers
func LoadCSV(r io.Reader) (*mat64.Dense, error) {
	// data matrix dimensions: rows x cols
	var rows, cols int
	// mxData contains ALL data read field by field
	var mxData []float64
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
		rows++
	}
	// Initialize data matrix with the read data
	mx := mat64.NewDense(rows, cols, mxData)
	return mx, nil
}

// Scale centers the data set to zero mean values and scales each column.
// It modifies the data stored in the data set. If your data contains also
// labeles in the last column, make sure you extract it before scaling.
func Scale(mx mat64.Matrix) mat64.Matrix {
	rows, cols := mx.Dims()
	// mean/stdev store each column mean/stdev values
	col := make([]float64, rows)
	mean := make([]float64, cols)
	stdev := make([]float64, cols)
	for i := 0; i < cols; i++ {
		// copy i-th column to col
		mat64.Col(col, i, mx)
		mean[i], stdev[i] = stat.MeanStdDev(col, nil)
	}
	scale := func(i, j int, x float64) float64 {
		return (x - mean[j]) / stdev[j]
	}
	dataMx := new(mat64.Dense)
	dataMx.Clone(mx)
	dataMx.Apply(scale, dataMx)
	return dataMx
}
