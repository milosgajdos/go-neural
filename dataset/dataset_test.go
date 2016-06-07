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
	content := []byte("2.0,3.5\n4.5,5.5\n7.0,9.0")
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
	ds, err := NewDataSet(tmpPath, true)
	assert.NoError(err)
	assert.NotNil(ds)
	assert.True(ds.IsLabeled())
	// retrieve data and check dimensions
	mx := ds.Data()
	rows, cols := mx.Dims()
	assert.Equal(3, rows)
	assert.Equal(2, cols)
	// unlabeled data
	ds, err = NewDataSet(tmpPath, false)
	assert.NoError(err)
	assert.NotNil(ds)
	assert.False(ds.IsLabeled())

	// Unsupported file format
	tmpfile, err := ioutil.TempFile("", "example")
	defer os.Remove(tmpfile.Name())
	assert.NoError(err)
	ds, err = NewDataSet(tmpfile.Name(), true)
	assert.Error(err)

	// Nonexistent file
	fileName3 := "nonexistent.csv"
	ds, err = NewDataSet(path.Join(".", fileName3), true)
	assert.Error(err)
}

func TestFeaturesLabels(t *testing.T) {
	assert := assert.New(t)

	// read data from temp file
	tmpPath := path.Join(os.TempDir(), fileName)
	ds, err := NewDataSet(tmpPath, true)
	assert.NoError(err)
	assert.NotNil(ds)

	// extract features from loaded data set
	features := ds.Features()
	assert.NotNil(features)
	r, c := features.Dims()
	assert.Equal(r, 3)
	assert.Equal(c, 1)
	// extract labels from loaded data set
	labels := ds.Labels()
	assert.NotNil(labels)
	r, c = labels.Dims()
	assert.Equal(r, 3)
	assert.Equal(c, 1)

	// can't extract features from vector
	ds, err = NewDataSet(tmpPath, false)
	assert.NoError(err)
	assert.NotNil(ds)
	// features must be equal to Data
	features = ds.Features()
	assert.True(mat64.Equal(features, ds.Data()))
	// labels must be nil
	labels = ds.Labels()
	assert.Nil(labels)

	// only one column data in labeled data set
	content := []byte("2.0")
	tmpPath = filepath.Join(os.TempDir(), "tst.csv")
	err = ioutil.WriteFile(tmpPath, content, 0666)
	assert.NoError(err)
	ds, err = NewDataSet(tmpPath, true)
	assert.NoError(err)
	assert.NotNil(ds)
	// features are the same as raw data
	features = ds.Features()
	assert.True(mat64.Equal(features, ds.Data()))
	// labels must be nil
	labels = ds.Labels()
	assert.Nil(labels)
}

func TestScale(t *testing.T) {
	assert := assert.New(t)

	// unlabeled data set
	tmpPath := path.Join(os.TempDir(), fileName)
	ds, err := NewDataSet(tmpPath, false)
	assert.NoError(err)
	assert.NotNil(ds)
	features := ds.Features()
	scaled := []float64{
		-1, -0.8980265101338746,
		0, -0.1796053020267749,
		1, 1.0776318121606494,
	}
	scaledMx := mat64.NewDense(3, 2, scaled)
	scaledFeats := Scale(features)
	assert.True(mat64.Equal(scaledFeats, scaledMx))
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
