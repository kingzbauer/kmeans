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

func TestAssignCentroid(t *testing.T) {
	X := mat.NewDense(4, 3, []float64{
		2, 6, 1,
		3, 4, 9,
		9, 8, 2,
		6, 4, 0,
	})
	Mu := mat.NewDense(2, 3, []float64{
		4, 9, 2,
		3, 1, 4,
	})
	m, _ := X.Dims()
	expectedVec := mat.NewVector(m, []float64{
		0,
		1,
		0,
		0,
	})

	Idx := AssignCentroid(X, Mu)
	if !mat.Equal(expectedVec, Idx) {
		t.Errorf("Expected \n%v, found \n%v", printMatrix(expectedVec), printMatrix(Idx))
	}
}
