package helpers

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPseudoRandString(t *testing.T) {
	assert := assert.New(t)
	length := 10
	prev := PseudoRandString(length)
	assert.Len(prev, length)
	for i := 0; i < 10; i++ {
		new := PseudoRandString(length)
		assert.Len(new, length)
		assert.NotEqual(prev, new)
		prev = new
	}
}

func TestParseParams(t *testing.T) {
	assert := assert.New(t)
	testCases := []struct {
		params  string
		correct bool
	}{
		{"", false},
		{"foo=bar=baz", false},
		{"foo=bar", false},
		{"foo=2.3", true},
	}

	for _, testCase := range testCases {
		m, err := ParseParams(testCase.params)
		if testCase.correct {
			assert.NotNil(m)
			assert.NoError(err)
		} else {
			assert.Nil(m)
			assert.Error(err)
		}
	}
}
