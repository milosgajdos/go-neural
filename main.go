package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
)

var (
	// path to the training data set
	dataPath string
	// do we want to normalize data
	reqScale bool
)

func init() {
	flag.StringVar(&dataPath, "data", "", "Path to training data set")
	flag.BoolVar(&reqScale, "scale", false, "Require data scaling")
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
	// 1 input layer, 1 hidden layer, 1 output layer
	//layers := []uint{uint(featCount), 25, 5}
	layers := []uint{uint(featCount), 25, 10}
	nn, err := NewNetwork(FEEDFWD, layers)
	if err != nil {
		fmt.Printf("Unable to initialize %s Neural network: %s\n", FEEDFWD, err)
		os.Exit(1)
	}
	// Train the network and return the cost value
	cost, err := nn.Train(featMx, labelVec, 10, 1.0, 50)
	if err != nil {
		fmt.Printf("Unable to train %s network: %s\n", nn.Kind(), err)
		os.Exit(1)
	}
	fmt.Printf("Cost: \n%f\n", cost)
}