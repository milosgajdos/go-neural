package config

import (
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v1"

	"github.com/stretchr/testify/assert"
)

var (
	fileName = "manifest.yml"
)

func setup() {
	content := []byte(`kind: feedfwd
task: class
layers:
  input:
    size: 400
  hidden:
    size: [25]
    activation: sigmoid
  output:
    size: 10
    activation: softmax
training:
  kind: backprop
  params: "lambda=1.0"
optimize:
  method: bfgs
  iterations: 69`)

	tmpPath := filepath.Join(os.TempDir(), fileName)
	if err := ioutil.WriteFile(tmpPath, content, 0666); err != nil {
		log.Fatal(err)
	}
}

func teardown() {
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

func TestNewNetConfig(t *testing.T) {
	assert := assert.New(t)

	tmpPath := path.Join(os.TempDir(), fileName)
	c, err := NewNetConfig(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	// nonexistent file
	c, err = NewNetConfig(filepath.Join(os.TempDir(), "random"))
	assert.Nil(c)
	assert.Error(err)
	// incorrect file
	tmpfile, err := ioutil.TempFile("", "example.yml")
	defer os.Remove(tmpfile.Name())
	assert.NoError(err)
	c, err = NewNetConfig(tmpfile.Name())
	assert.Nil(c)
	assert.Error(err)
}

func TestParseLayers(t *testing.T) {
	assert := assert.New(t)

	var m Manifest
	tmpPath := path.Join(os.TempDir(), fileName)
	f, err := os.Open(tmpPath)
	defer f.Close()
	assert.NoError(err)
	mData, err := ioutil.ReadAll(f)
	assert.NoError(err)
	err = yaml.Unmarshal(mData, &m)
	assert.NoError(err)
	// empty net kind name
	origKind := m.Kind
	m.Kind = ""
	c, err := Parse(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Kind = "unsupported"
	c, err = Parse(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Kind = origKind
	// incorrect input layer size
	origInSize := m.Layers.Input.Size
	m.Layers.Input.Size = 0
	c, err = Parse(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Kind = origKind
	m.Layers.Input.Size = origInSize
	// unknown activation function
	origActFn := m.Layers.Hidden.Activation
	m.Layers.Hidden.Activation = "foobar"
	c, err = Parse(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Layers.Hidden.Activation = origActFn
	// incorrect output size
	origOutSize := m.Layers.Output.Size
	m.Layers.Output.Size = 0
	c, err = Parse(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Layers.Output.Size = origOutSize
	// output activation function
	origActFn = m.Layers.Output.Activation
	m.Layers.Output.Activation = "foobar"
	c, err = Parse(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Layers.Output.Activation = origActFn
}

func TestParseTraining(t *testing.T) {
	assert := assert.New(t)

	var m Manifest
	tmpPath := path.Join(os.TempDir(), fileName)
	f, err := os.Open(tmpPath)
	defer f.Close()
	assert.NoError(err)
	mData, err := ioutil.ReadAll(f)
	assert.NoError(err)
	err = yaml.Unmarshal(mData, &m)
	assert.NoError(err)
	// unsupported network has no training
	origNetKind := m.Kind
	m.Kind = "SOM"
	c, err := Parse(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Kind = origNetKind
	// unsupported training algorithm
	origTrAlg := m.Training.Kind
	m.Training.Kind = "foobar"
	c, err = Parse(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Training.Kind = origTrAlg
}

func TestParseOptimize(t *testing.T) {
	assert := assert.New(t)

	var m Manifest
	tmpPath := path.Join(os.TempDir(), fileName)
	f, err := os.Open(tmpPath)
	defer f.Close()
	assert.NoError(err)
	mData, err := ioutil.ReadAll(f)
	assert.NoError(err)
	err = yaml.Unmarshal(mData, &m)
	assert.NoError(err)
	// unsupported optimization method
	origOptimMethod := m.Optimize.Method
	m.Optimize.Method = "foobar"
	c, err := Parse(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Optimize.Method = origOptimMethod
}
