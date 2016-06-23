package main

import (
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/neural"
	"github.com/milosgajdos83/go-neural/pkg/config"
	"github.com/milosgajdos83/go-neural/pkg/dataset"
	"github.com/milosgajdos83/go-neural/pkg/helpers"
	"github.com/milosgajdos83/go-neural/train/backprop"
)

var (
	// path to the training data set
	data string
	// is the data set labeled
	labeled bool
	// do we want to normalize data
	scale bool
	// manifest contains neural net config
	manifest string
)

func init() {
	flag.StringVar(&data, "data", "", "Path to training data set")
	flag.BoolVar(&labeled, "labeled", false, "Is the data set labeled")
	flag.BoolVar(&scale, "scale", false, "Require data scaling")
	flag.StringVar(&manifest, "manifest", "", "Path to a neural net manifest file")
}

func parseCliFlags() error {
	flag.Parse()
	// path to training data is mandatory
	if data == "" {
		return errors.New("You must specify path to training data set")
	}

	// path to manifest is mandatory
	if manifest == "" {
		return errors.New("You must specify path to manifest file")
	}
	return nil
}

func main() {
	// parse cli parameters
	if err := parseCliFlags(); err != nil {
		fmt.Printf("Error parsing cli flags: %s\n", err)
		os.Exit(1)
	}
	// Read in configuration file
	config, err := config.NewNetConfig(manifest)
	if err != nil {
		fmt.Printf("Error reading manifest file: %s\n", err)
		os.Exit(1)
	}
	// load new data set from provided file
	ds, err := dataset.NewDataSet(data, labeled)
	if err != nil {
		fmt.Printf("Unable to load Data Set: %s\n", err)
		os.Exit(1)
	}
	// extract features from data set
	features := ds.Features()
	// if we require features scaling, scale data
	if scale {
		features = dataset.Scale(features)
	}
	// extract data labels
	labels := ds.Labels()
	if labels == nil {
		fmt.Println("Data set does not contain any labels")
		os.Exit(1)
	}
	// Create new FEEDFWD network
	net, err := neural.NewNetwork(config)
	if err != nil {
		fmt.Printf("Error creating neural network: %s\n", err)
		os.Exit(1)
	}
	params, err := helpers.ParseParams(config.Training.Params)
	if err != nil {
		fmt.Printf("Error parsing training params: %s\n", err)
		os.Exit(1)
	}
	lambda, ok := params["lambda"]
	if !ok {
		fmt.Printf("Could not find lambda in training parameters")
		os.Exit(1)
	}

	// neural network training
	tc := config.Training
	c := &backprop.Config{
		Weights: nil,
		Optim:   tc.Optimize.Method,
		Lambda:  lambda,
		Labels:  config.Arch.Output.Size,
		Iters:   tc.Optimize.Iterations,
	}
	err = backprop.Train(net, c, features.(*mat64.Dense), labels.(*mat64.Vector))
	if err != nil {
		fmt.Printf("Error training network: %s\n", err)
		os.Exit(1)
	}
	// check the success rate i.e. successful number of classifications
	success, err := net.Validate(features.(*mat64.Dense), labels.(*mat64.Vector))
	if err != nil {
		fmt.Printf("Could not calculate success rate: %s\n", err)
		os.Exit(1)
	}
	fmt.Printf("\nNeural net accuracy: %f\n", success)
	// Example of sample classification: in this case it's 1st data sample
	sample := (features.(*mat64.Dense)).RowView(0).T()
	classMx, err := net.Classify(sample)
	if err != nil {
		fmt.Printf("Could not classify sample: %s\n", err)
		os.Exit(1)
	}
	fa := mat64.Formatted(classMx.T(), mat64.Prefix(""))
	fmt.Printf("\nClassification result:\n% v\n\n", fa)
}
