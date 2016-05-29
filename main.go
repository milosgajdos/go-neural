package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
)

var (
	// path to the training data set
	dataPath string
	// do we want to normalize data
	reqScale bool
	// number of labels
	labels int
	// number of iterations
	iters int
	// regularization parameter
	lambda float64
)

func init() {
	flag.StringVar(&dataPath, "data", "", "Path to training data set")
	flag.BoolVar(&reqScale, "scale", false, "Require data scaling")
	flag.IntVar(&labels, "labels", 0, "Number of class labels")
	flag.IntVar(&iters, "iters", 50, "Number of iterations")
	flag.Float64Var(&lambda, "lambda", 1.0, "Regularization parameter")
}

func parseCliFlags() error {
	flag.Parse()
	// path to training data is mandatory
	if dataPath == "" {
		return errors.New("You must specify the path to training data set")
	}
	return nil
}

func main() {
	f, err := os.Create("nn.profile")
	if err != nil {
		log.Fatal(err)
	}
	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()
	// parse cli parameters
	if err := parseCliFlags(); err != nil {
		fmt.Printf("Error parsing cli flags: %s\n", err)
		os.Exit(1)
	}
	// load training data set
	dataMx, err := LoadCSVData(dataPath)
	if err != nil {
		fmt.Printf("Error loading data: %s\n", err)
		os.Exit(1)
	}
	// extract features and labels from raw data matrix
	featMx, labelVec, err := ExtractFeatures(dataMx)
	if err != nil {
		fmt.Printf("Could not extract features: %s\n", err)
		os.Exit(1)
	}
	_, featCount := featMx.Dims()
	// Create new Neural Network:
	hiddenLayerSize := uint(25)
	// TODO: what happens if outputLayerSize > nrLabels - check the code
	// OutputLayer size MUST be at least as big as nrLabels
	outputLayerSize := uint(labels)
	if outputLayerSize != uint(labels) {
		fmt.Printf("Output layer must be same as number of labels\n")
		os.Exit(1)
	}
	layers := []uint{uint(featCount), hiddenLayerSize, outputLayerSize}
	nn, err := NewNetwork(FEEDFWD, layers)
	if err != nil {
		fmt.Printf("Unable to initialize %s Neural network: %s\n", FEEDFWD, err)
		os.Exit(1)
	}
	// Train the network and return the cost value
	if _, err := nn.Train(featMx, labelVec, labels, lambda, iters); err != nil {
		fmt.Printf("Unable to train %s network: %s\n", nn.Kind(), err)
		os.Exit(1)
	}
	success, err := nn.Validate(featMx, labelVec)
	if err != nil {
		fmt.Printf("UNable to calculate success rate: %s\n", err)
		os.Exit(1)
	}
	fmt.Printf("Neural net success: %f\n", success)
}
