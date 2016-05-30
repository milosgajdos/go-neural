package dataset

import (
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

var (
	fileName = "example.csv"
)

func setup() {
	// create a correct test file
	content := []byte("1,2,3")
	tmpPath := filepath.Join(os.TempDir(), fileName)
	if err := ioutil.WriteFile(tmpPath, content, 0666); err != nil {
		log.Fatal(err)
	}
}

func teardown() {
	// remove test file
	os.Remove(filepath.Join(os.TempDir(), fileName))
}

func TestMain(m *testing.M) {
	// set up tests
	setup()
	// run the tests
	retCode := m.Run()
	// delete test files
	teardown()
	// call with result of m.Run()
	os.Exit(retCode)
}

func TestDataSet(t *testing.T) {
	assert := assert.New(t)

	tmpPath := path.Join(os.TempDir(), fileName)
	ds, err := NewDataSet(tmpPath)
	assert.NoError(err)
	assert.NotNil(ds)
	mx := ds.Data()
	rows, cols := mx.Dims()
	assert.Equal(1, rows)
	assert.Equal(3, cols)

	// Unsupported file format
	tmpfile, err := ioutil.TempFile("", "example")
	defer os.Remove(tmpfile.Name())
	assert.NoError(err)
	ds, err = NewDataSet(tmpfile.Name())
	assert.Error(err)

	// Nonexistent file
	fileName3 := "nonexistent.csv"
	ds, err = NewDataSet(path.Join(".", fileName3))
	assert.Error(err)
}

func TestLoadCSV(t *testing.T) {
	assert := assert.New(t)

	// correct data
	tstRdr := strings.NewReader("1,2,3")
	mx, err := LoadCSV(tstRdr)
	assert.NoError(err)
	r, c := mx.Dims()
	assert.Equal(r, 1)
	assert.Equal(c, 3)

	// inconsisten data
	tstRdr = strings.NewReader("1,2,3\n4,5")
	mx, err = LoadCSV(tstRdr)
	assert.Error(err)
	assert.Nil(mx)

	// corrupted data
	tstRdr = strings.NewReader("1,sdfsdfd,3\n4,5")
	mx, err = LoadCSV(tstRdr)
	assert.Error(err)
	assert.Nil(mx)
}

func TestExtractFeatures(t *testing.T) {
	assert := assert.New(t)

	// read data from temp file
	tmpPath := path.Join(os.TempDir(), fileName)
	ds, err := NewDataSet(tmpPath)
	assert.NoError(err)
	assert.NotNil(ds)
	mx := ds.Data()

	// extract features from loaded data set
	features, labels, err := ExtractFeatures(mx)
	assert.NoError(err)
	r, c := features.Dims()
	assert.Equal(r, 1)
	assert.Equal(c, 2)
	r, c = labels.Dims()
	assert.Equal(r, 1)
	assert.Equal(c, 1)

	// can't extract features from vector
	tstVec := mat64.NewVector(2, nil)
	features, labels, err = ExtractFeatures(tstVec)
	assert.Nil(features)
	assert.Nil(labels)
	assert.Error(err)

	// dimensions of data matrix are too low
	tstMx := mat64.NewDense(1, 1, nil)
	features, labels, err = ExtractFeatures(tstMx)
	assert.Nil(features)
	assert.Nil(labels)
	assert.Error(err)
}
