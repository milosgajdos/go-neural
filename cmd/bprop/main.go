package main

import (
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/milosgajdos83/go-neural/dataset"
)

var (
	// path to the training data set
	data string
	// is the data set labeled
	labels bool
	// do we want to normalize data
	scale bool
	// number of iterations
	iters int
	// regularization parameter
	lambda float64
)

func init() {
	flag.StringVar(&data, "data", "", "Path to training data set")
	flag.BoolVar(&labels, "labels", false, "Is the data set labeled")
	flag.BoolVar(&scale, "scale", false, "Require data scaling")
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
	ds, err := dataset.NewDataSet(data, labels)
	if err != nil {
		fmt.Println("Unable to load Data Set: %s\n", err)
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
		fmt.Println("No labels available for supervised learning")
		os.Exit(1)
	}
}
