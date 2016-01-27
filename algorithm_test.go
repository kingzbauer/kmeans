package kmeans

import (
	mat "github.com/gonum/matrix/mat64"
	"testing"
)

func TestAppIterMu(t *testing.T) {
	ExpectedMatrix := mat.NewDense(4, 3, []float64{
		6, 4, 1,
		2, 3, 4,
		9, 1, 6,
		6, 2, 2,
	})
	iter := &AppIter{mu: ExpectedMatrix}

	ResultMat := iter.Mu()
	if !mat.Equal(ExpectedMatrix, ResultMat) {
		t.Errorf("Expected \n%v, got \n%v",
			printMatrix(ExpectedMatrix), printMatrix(ResultMat))
	}
}

func TestAppIterIdx(t *testing.T) {
	expectedVec := mat.NewVector(4, []float64{
		3,
		4,
		8,
		1,
	})
	iter := &AppIter{idx: expectedVec}

	resultVec := iter.Idx()
	if !mat.Equal(expectedVec, resultVec) {
		t.Errorf("Expected \n%v got,\n%v",
			printMatrix(expectedVec), printMatrix(resultVec))
	}
}

func TestAppIterCost(t *testing.T) {
	cost := 4.34
	iter := &AppIter{currentCost: cost}

	resultCost := iter.Cost()
	if resultCost != cost {
		t.Errorf("Expected %.5f, got %.5f", cost, resultCost)
	}
}
