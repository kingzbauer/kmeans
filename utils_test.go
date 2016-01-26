package kmeans

import (
	mat "github.com/gonum/matrix/mat64"
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

// Calculates the distance between to vectors
func TestVectorDistance(t *testing.T) {
	vec1 := mat.NewVector(3, []float64{4, 6, 2})
	vec2 := mat.NewVector(3, []float64{1, 9, 3})
	expectedAns := float64(19)

	if expectedAns != vectorDistance(vec1, vec2) {
		t.Errorf("Expected %f, got %f", expectedAns, vectorDistance(vec1, vec2))
	}
}
