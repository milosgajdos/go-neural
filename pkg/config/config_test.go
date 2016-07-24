package config

import (
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"testing"

	yaml "gopkg.in/yaml.v1"

	"github.com/stretchr/testify/assert"
)

var (
	fileName = "manifest.yml"
)

func setup() {
	content := []byte(`kind: feedfwd
task: class
network:
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
  cost: xentropy
  params:
    lambda: 1.0
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

func TestNewConfig(t *testing.T) {
	assert := assert.New(t)

	tmpPath := path.Join(os.TempDir(), fileName)
	c, err := New(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	// test if the parsed parameters are correct
	assert.Equal(c.Network.Kind, "feedfwd")
	assert.Equal(c.Network.Arch.Input.Kind, "input")
	assert.Equal(c.Network.Arch.Input.Size, 400)
	assert.Equal(c.Network.Arch.Input.NeurFn, (*NeuronConfig)(nil))
	assert.Equal(c.Network.Arch.Hidden[0].Kind, "hidden")
	assert.Equal(c.Network.Arch.Hidden[0].Size, 25)
	assert.Equal(c.Network.Arch.Hidden[0].NeurFn.Activation, "sigmoid")
	assert.Equal(c.Network.Arch.Output.Kind, "output")
	assert.Equal(c.Network.Arch.Output.Size, 10)
	assert.Equal(c.Network.Arch.Output.NeurFn.Activation, "softmax")
	assert.Equal(c.Training.Kind, "backprop")
	assert.Equal(c.Training.Cost, "xentropy")
	assert.Equal(c.Training.Lambda, 1.0)
	assert.Equal(c.Training.Optimize.Method, "bfgs")
	assert.Equal(c.Training.Optimize.Iterations, 69)
	// nonexistent file
	c, err = New(filepath.Join(os.TempDir(), "random"))
	assert.Nil(c)
	assert.Error(err)
	// empty file
	tmpfile, err := ioutil.TempFile("", "example.yml")
	defer os.Remove(tmpfile.Name())
	assert.NoError(err)
	c, err = New(tmpfile.Name())
	assert.Nil(c)
	assert.Error(err)
}

func TestParseManifest(t *testing.T) {
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
	// correct config
	c, err := New(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	// empty net kind name
	origKind := m.Kind
	m.Kind = ""
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	// unsupported net kind
	m.Kind = "unsupported"
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Kind = origKind
}

func TestParseNetConfig(t *testing.T) {
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
	// correct config
	c, err := New(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	// incorrect input layer size
	origInSize := m.Network.Input.Size
	m.Network.Input.Size = 0
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Network.Input.Size = origInSize
	// incorrect hidden layer size
	origHidSize := m.Network.Hidden.Size[0]
	m.Network.Hidden.Size[0] = 0
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Network.Hidden.Size[0] = origHidSize
	// incorrect output size
	origOutSize := m.Network.Output.Size
	m.Network.Output.Size = 0
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Network.Output.Size = origOutSize
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
	// correct config
	c, err := New(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	// empty optimize method
	origOptimMethod := m.Training.Optimize.Method
	m.Training.Optimize.Method = ""
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	// unsupported optimization method
	m.Training.Optimize.Method = "foobar"
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Training.Optimize.Method = origOptimMethod
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
	// correct config
	c, err := New(tmpPath)
	assert.NotNil(c)
	assert.NoError(err)
	// empty training kind
	origTrAlg := m.Training.Kind
	m.Training.Kind = ""
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Training.Kind = origTrAlg
	// unsupported training algorithm
	m.Training.Kind = "foobar"
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Training.Kind = origTrAlg
	// empty cost function
	origCost := m.Training.Cost
	m.Training.Cost = ""
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	// unsupported cost function
	m.Training.Cost = "foocost"
	c, err = ParseManifest(&m)
	assert.NotNil(c)
	assert.NoError(err)
	assert.Equal(c.Training.Cost, "foocost")
	m.Training.Cost = origCost
	// incorrect lambda
	origLambda := m.Training.Params.Lambda
	m.Training.Params.Lambda = -1
	c, err = ParseManifest(&m)
	assert.Nil(c)
	assert.Error(err)
	m.Training.Params.Lambda = origLambda
	// correct parameters
	c, err = ParseManifest(&m)
	assert.NotNil(c)
	assert.NoError(err)
}
