package kmeans

import (
	"fmt"
	mat "github.com/gonum/matrix/mat64"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().Unix())
}

func randArray(min, max, length int, unique bool) []int {
	array := make([]int, length)

	var i int
	for i = 0; i < length; i++ {
		randValue := randRange(min, max)
		if unique {
			if !in(randValue, array) {
				array[i] = randValue
			} else {
				i--
			}
		} else {
			array[i] = randValue
		}
	}

	return array
}

func randRange(min, max int) int {
	return rand.Intn(max-min) + min
}

func in(i int, s []int) (present bool) {

	for _, v := range s {
		if v == i {
			return true
		}
	}

	return
}

func vectorDistance(vec1, vec2 *mat.Vector) (v float64) {
	result := mat.NewVector(vec1.Len(), nil)

	result.SubVec(vec1, vec2)
	result.MulElemVec(result, result)
	v = mat.Sum(result)

	return
}

func getRowVector(index int, M mat.Matrix) *mat.Vector {
	_, cols := M.Dims()

	rowData := mat.Row(nil, index, M)
	return mat.NewVector(cols, rowData)
}

func getColumnVector(index int, M mat.Matrix) *mat.Vector {
	rows, _ := M.Dims()

	colData := mat.Col(nil, index, M)
	return mat.NewVector(rows, colData)
}

func printMatrix(M mat.Matrix) fmt.Formatter {
	return mat.Formatted(M, mat.Prefix(""))
}

// findIn returns the indexes of the values in vec that match scalar
func findIn(scalar float64, vec *mat.Vector) *mat.Vector {
	var result []float64

	for i := 0; i < vec.Len(); i++ {
		if scalar == vec.At(i, 0) {
			result = append(result, float64(i))
		}
	}

	return mat.NewVector(len(result), result)
}

// rowIndexIn returns a matrix contains the rows in indexes vector
func rowIndexIn(indexes *mat.Vector, M mat.Matrix) mat.Matrix {
	m := indexes.Len()
	_, n := M.Dims()
	Res := mat.NewDense(m, n, nil)

	for i := 0; i < m; i++ {
		Res.SetRow(i, mat.Row(
			nil,
			int(indexes.At(i, 0)),
			M))
	}

	return Res
}

func columnSum(M mat.Matrix) mat.Matrix {
	_, cols := M.Dims()

	floatRes := make([]float64, cols)
	for i := 0; i < cols; i++ {
		floatRes[i] = mat.Sum(getColumnVector(i, M))
	}

	return mat.NewDense(1, cols, floatRes)
}

func columnMean(M mat.Matrix) mat.Matrix {
	r, c := M.Dims()

	SumMatrix := columnSum(M)

	switch t := SumMatrix.(type) {
	case *mat.Dense:
		M := mat.NewDense(1, c, nil)
		M.Scale(1/float64(r), SumMatrix)
		return M
	case mat.Mutable:
		_ = t
		V := SumMatrix.(mat.Mutable)
		_, cols := V.Dims()

		for i := 0; i < cols; i++ {
			V.Set(0, i, SumMatrix.At(0, i)/float64(r))
		}

		return V
	default:
		panic("M is of an unknown type")
	}

}

func rowSum(M mat.Matrix) mat.Matrix {
	rows, _ := M.Dims()

	floatRes := make([]float64, rows)
	for i := 0; i < rows; i++ {
		floatRes[i] = mat.Sum(getRowVector(i, M))
	}

	return mat.NewDense(rows, 1, floatRes)
}

func constructXCentroidMatrix(idx *mat.Vector, Mu mat.Matrix) mat.Matrix {

	return rowIndexIn(idx, Mu)
}
