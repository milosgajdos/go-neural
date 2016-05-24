package main

import (
	"encoding/csv"
	"errors"
	"io"
	"os"
	"strconv"

	"github.com/gonum/matrix/mat64"
)

// LoadCSVData loads training data from the path specified as a parameter
// It returns data matrix that contains particular CSV fields as features.
// It returns error if either data file does not exist, it contains corrrupted data or the data can not be converted to floar numbers
func LoadCSVData(path string) (*mat64.Dense, error) {
	// Check if the training data file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, err
	}
	// Open training data file
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	// number of data matrix rows and columns
	var rows, cols int
	// mxData contains ALL matrix values; it's used to init matrix
	// dataRow contains slice of floats that are appended to mxData
	var mxData, dataRow []float64
	// create new CSV reader
	r := csv.NewReader(file)
	// read all data record by record
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		// only set the cols on the first iteration
		if rows == 0 {
			cols = len(record)
			// Allocate dataRow slice only once
			dataRow = make([]float64, cols)
		}
		// number of columns is not the same as in the record
		if cols != len(record) {
			// TODO: decide what to do when values are missing
			return nil, errors.New("Incosistent number of features")
		}
		// convert strings to flaots
		for i, field := range record {
			// TODO: decide what to do when field can't be converted
			f, err := strconv.ParseFloat(field, 64)
			if err != nil {
				return nil, err
			}
			dataRow[i] = f
		}
		// if the data is labelled, append to label vector
		mxData = append(mxData, dataRow...)
		rows += 1
	}
	// Data matrix
	mx := mat64.NewDense(rows, cols, mxData)
	return mx, nil
}

// ExtractFeatures extracts features and labels from raw data matrix
// It returns features matrix and vector of data labels
// It returns error if all the data features can not be extracted
func ExtractFeatures(dataMx *mat64.Dense) (*mat64.Dense, *mat64.Vector, error) {
	// get matrix dimensions
	rows, cols := dataMx.Dims()
	// extract labels from dataMx
	labelVec := dataMx.ColView(cols - 1)
	// create view on data features
	featView := dataMx.View(0, 0, rows, cols-1)
	// allocate new feature matrix
	featMx := mat64.NewDense(rows, cols-1, nil)
	// copy data from data matrix to the new feature matrix
	r, c := featMx.Copy(featView)
	// If we couldn't copy ALL data from data matrix we error
	if r != rows || c != cols-1 {
		return nil, nil, errors.New("Unable to copy all data")
	}
	return featMx, labelVec, nil
}
