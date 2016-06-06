package main

import (
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/go-neural/dataset"
	"github.com/milosgajdos83/go-neural/learn/backprop"
	"github.com/milosgajdos83/go-neural/neural"
)

var (
	// path to the training data set
	data string
	// is the data set labeled
	labeled bool
	// do we want to normalize data
	scale bool
	// how many classes are in the datasets
	classes int
	// number of iterations
	iters int
	// regularization parameter
	lambda float64
)

func init() {
	flag.StringVar(&data, "data", "", "Path to training data set")
	flag.BoolVar(&labeled, "labeled", false, "Is the data set labeled")
	flag.BoolVar(&scale, "scale", false, "Require data scaling")
	flag.IntVar(&classes, "classes", 0, "How many classes are in the data set")
	flag.IntVar(&iters, "iters", 50, "Number of iterations")
	flag.Float64Var(&lambda, "lambda", 1.0, "Regularization parameter")
}

func parseCliFlags() error {
	flag.Parse()
	// path to training data is mandatory
	if data == "" {
		return errors.New("You must specify the path to training data set")
	}
	return nil
}

func main() {
	// parse cli parameters
	if err := parseCliFlags(); err != nil {
		fmt.Printf("Error parsing cli flags: %s\n", err)
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
	// extract labels
	labels := ds.Labels()
	if labels == nil {
		fmt.Println("Data set does not contain any labels")
		os.Exit(1)
	}
	// number of classes must be a positive integer
	if classes < 1 {
		fmt.Printf("Insufficient number of classes: %d\n", classes)
		os.Exit(1)
	}
	// Create new FEEDFWD network
	_, colsIn := features.Dims()
	hiddenLayers := []int{25}
	// we will classify the output to number of specified classes
	netArch := &neural.NetworkArch{Input: colsIn, Hidden: hiddenLayers, Output: classes}
	net, err := neural.NewNetwork(neural.FEEDFWD, netArch)
	if err != nil {
		fmt.Printf("Error creating network: %s\n", err)
		os.Exit(1)
	}
	c := &backprop.Config{Weights: nil, Lambda: lambda, Labels: classes, Iters: iters}
	if err := backprop.Train(net, c, features.(*mat64.Dense), labels.(*mat64.Vector)); err != nil {
		fmt.Printf("Error training network: %s\n", err)
	}
}
