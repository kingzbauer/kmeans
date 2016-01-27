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

func TestGetRowVector(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		6, 4, 2,
		10, 3, 9,
		4, 7, 1,
	})
	rowIndex := 1
	expectedVector := mat.NewVector(3, []float64{
		10,
		3,
		9,
	})

	if !mat.Equal(expectedVector, getRowVector(rowIndex, A)) {
		t.Errorf("Expected true, got false")
	}
}

func TestFindIn(t *testing.T) {
	v := mat.NewVector(4, []float64{
		1,
		0,
		1,
		0,
	})
	x := 1
	expectedVec := mat.NewVector(2, []float64{
		0,
		2,
	})

	result := findIn(float64(x), v)
	if !mat.Equal(result, expectedVec) {
		t.Errorf("Expected \n%v, found \n%v",
			printMatrix(expectedVec), printMatrix(result))
	}
}

func TestRowIndexIn(t *testing.T) {
	Matrix := mat.NewDense(4, 3, []float64{
		6, 4, 1,
		2, 3, 4,
		9, 1, 6,
		6, 2, 2,
	})
	vecIndex := mat.NewVector(2, []float64{
		1,
		3,
	})
	ExpectedMatrix := mat.NewDense(2, 3, []float64{
		2, 3, 4,
		6, 2, 2,
	})

	Result := rowIndexIn(vecIndex, Matrix)
	if !mat.Equal(ExpectedMatrix, Result) {
		t.Errorf("Expected \n%v, got \n%v",
			printMatrix(ExpectedMatrix),
			printMatrix(Result))
	}
}

func TestGetColumnVector(t *testing.T) {
	M := mat.NewDense(4, 3, []float64{
		6, 4, 1,
		2, 3, 4,
		9, 1, 6,
		6, 2, 2,
	})
	columnIndex := 1
	vectorLen, _ := M.Dims()
	expVec := mat.NewVector(vectorLen, []float64{
		4,
		3,
		1,
		2,
	})

	resultVec := getColumnVector(columnIndex, M)
	if !mat.Equal(expVec, resultVec) {
		t.Errorf("Expected \n%v, got\n %v", printMatrix(expVec), printMatrix(resultVec))
	}
}

func TestColumnSum(t *testing.T) {
	M := mat.NewDense(4, 3, []float64{
		6, 4, 1,
		2, 3, 4,
		9, 1, 6,
		6, 2, 2,
	})
	_, cols := M.Dims()
	ExpectedRes := mat.NewDense(1, cols, []float64{
		23, 10, 13,
	})

	Result := columnSum(M)
	if !mat.Equal(ExpectedRes, Result) {
		t.Errorf("Expected \n%v, got\n%v", printMatrix(ExpectedRes), printMatrix(Result))
	}
}

func TestColumnMean(t *testing.T) {
	M := mat.NewDense(4, 3, []float64{
		6, 4, 1,
		2, 3, 4,
		9, 1, 6,
		6, 2, 2,
	})
	_, cols := M.Dims()
	ExpectedRes := mat.NewDense(1, cols, []float64{
		5.75, 2.5, 3.25,
	})

	Result := columnMean(M)
	if !mat.Equal(ExpectedRes, Result) {
		t.Errorf("Expected \n%v, got\n%v", printMatrix(ExpectedRes), printMatrix(Result))
	}
}
