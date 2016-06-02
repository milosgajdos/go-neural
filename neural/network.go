package neural

const (
	// FEEDFWD is a feed forwardf Neural Network
	FEEDFWD NetworkKind = iota + 1
)

// NetworkKind defines a type of neural network
type NetworkKind uint

// String implements Stringer interface for pretty printing
func (nk NetworkKind) String() string {
	switch nk {
	case FEEDFWD:
		return "FEEDFWD"
	default:
		return "UNKNOWN"
	}
}

// Network represents a certain kind of Neural Network.
// It has an id and can have arbitrary number of layers.
type Network struct {
	id     string
	kind   NetworkKind
	layers []*Layer
}

// ID returns neural network id
func (n Network) ID() string {
	return n.id
}
