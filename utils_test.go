package kmeans

import (
	"testing"
)

func TestIn(t *testing.T) {
	array := []int{3, 4, 5, 6}

	if !in(6, array) {
		t.Errorf("Expected %d to be in %v", 6, array)
	}

	if in(10, array) {
		t.Errorf("Did not expect %d to be in %v", 10, array)
	}
}

func TestRandRange(t *testing.T) {
	min, max := 6, 9

	randomV := randRange(min, max)

	if randomV > max || randomV < min {
		t.Errorf("Expected the random value %d to be within the range %d to %d",
			randomV, min, max)
	}
}
