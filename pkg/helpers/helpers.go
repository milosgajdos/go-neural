package helpers

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"
)

// PseudoRandString generates a pseudoandom string of specified size
func PseudoRandString(size int) string {
	alphanum := "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	bytes := make([]byte, size)
	rand.Read(bytes)
	// iterate through all alphanum bytes
	for i, b := range bytes {
		bytes[i] = alphanum[b%byte(len(alphanum))]
	}
	return string(bytes)
}

// ParseParams parses parameters from supplied string and returns them in a map
func ParseParams(params string) (map[string]float64, error) {
	if params == "" {
		return nil, fmt.Errorf("Can't parse empty param string")
	}
	parMap := make(map[string]float64)
	// split parameter pairs by '&'
	pairs := strings.Split(params, "&")
	for _, pair := range pairs {
		// plit pair by '='
		param := strings.Split(pair, "=")
		if len(param) != 2 {
			return nil, fmt.Errorf("Incorrect parameter: %s\n", pair)
		}
		parName := param[0]
		parVal, err := strconv.ParseFloat(param[1], 64)
		if err != nil {
			return nil, fmt.Errorf("Incorrect parameter: %s\n", param[1])
		}
		parMap[parName] = parVal
	}
	return parMap, nil
}
