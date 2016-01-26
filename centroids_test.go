package kmeans

import (
	mat "github.com/gonum/matrix/mat64"
	"testing"
)

func TestNearestCentroid(t *testing.T) {
	v := mat.NewVector(3, []float64{6, 2, 9})
	var Mu mat.Matrix
	Mu = mat.NewDense(3, 3, []float64{
		6, 4, 2,
		4, 7, 1,
		10, 3, 9,
	})

	nearest := 2
	if NearestCentroid(v, Mu) != nearest {
		t.Errorf("Expected %d, got %d", nearest, NearestCentroid(v, Mu))
	}
}
