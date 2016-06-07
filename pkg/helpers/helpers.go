package helpers

import "math/rand"

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
