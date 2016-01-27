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

func TestMoveCentroids(t *testing.T) {
	Mu := mat.NewDense(2, 3, []float64{
		4, 9, 2,
		3, 1, 4,
	})
	X := mat.NewDense(4, 3, []float64{
		2, 6, 1,
		3, 4, 9,
		9, 8, 2,
		6, 4, 0,
	})
	ExpectedMu := mat.NewDense(2, 3, []float64{
		17 / float64(3), 6, 1,
		3, 4, 9,
	})

	idx := AssignCentroid(X, Mu)
	ResultMu := MoveCentroids(idx, X, Mu)

	if !mat.EqualApprox(ExpectedMu, ResultMu, 1e-7) {
		t.Errorf("Expected \n%v, got\n%v",
			printMatrix(ExpectedMu), printMatrix(ResultMu))
	}
}

func TestJ(t *testing.T) {
	X := mat.NewDense(4, 3, []float64{
		6, 4, 1,
		2, 3, 4,
		9, 1, 6,
		6, 2, 2,
	})
	Mu := mat.NewDense(2, 3, []float64{
		4, 9, 2,
		3, 1, 4,
	})
	idx := mat.NewVector(4, []float64{
		0,
		1,
		0,
		1,
	})
	expectedCost := 38.5

	cost := J(idx, X, Mu)
	if cost != expectedCost {
		t.Errorf("Expected the cost to be %.5f, got %.5f", expectedCost, cost)
	}
}
