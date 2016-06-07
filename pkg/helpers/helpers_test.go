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
