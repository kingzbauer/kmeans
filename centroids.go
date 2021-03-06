package kmeans

import (
	mat "github.com/gonum/matrix/mat64"
)

// Given number of clusters K and the training set X of dimension (m*n)
// InitializeCentroids returns matrix Mu of dimension (K*n)
// The steps to initialize the centroids are as follows
//     1. Randomly pick K training examples (Make sure the selected examples are unique)
//     2. Set Mu to the K examples
func InitializeCentroids(K int, X mat.Matrix) (Mu mat.Matrix) {
	m, n := X.Dims()

	// panic if K >= m
	if K >= m {
		panic("K should be less than the size of the training set")
	}

	randomIndexes := randArray(0, m, K, true) // 1. pick K training examples

	// 2. set Mu
	Mu = mat.NewDense(K, n, nil)
	for i := 0; i < K; i++ {
		Mu.(*mat.Dense).SetRow(i, mat.Row(nil, randomIndexes[i], X))
	}

	return
}

// NearestCentroid returns the index of the row in Mu for which its
// vector magnitude with x is the least.
// x should be (n*1) space and M (K*n) space
func NearestCentroid(x *mat.Vector, Mu mat.Matrix) (rowIndex int) {
	k, _ := Mu.Dims()
	rowIndex = 0
	leastDistance := vectorDistance(x, getRowVector(rowIndex, Mu))

	for i := 1; i < k; i++ {
		distance := vectorDistance(x, getRowVector(i, Mu))
		if distance < leastDistance {
			leastDistance = distance
			rowIndex = i
		}
	}

	return
}

// AssignCentroid assigns all of the examples in X to one of the groups
// in Mu
// X -> (m*n), Mu -> (K*n)
// returns (m*1)
func AssignCentroid(X, Mu mat.Matrix) *mat.Vector {
	m, _ := X.Dims()
	idx := mat.NewVector(m, nil)

	for i := 0; i < m; i++ {
		x := getRowVector(i, X)
		idx.SetVec(i, float64(NearestCentroid(x, Mu)))
	}

	return idx
}

// MoveCentroid computes the averages for all the points inside each of the cluster
// centroid groups, then move the cluster centroid points to those averages.
// It then returns the new Centroids
func MoveCentroids(idx *mat.Vector, X, Mu mat.Matrix) mat.Matrix {
	muRows, muCols := Mu.Dims()
	NewMu := mat.NewDense(muRows, muCols, nil)

	for k := 0; k < muRows; k++ {
		CentroidKMean := columnMean(rowIndexIn(findIn(float64(k), idx), X))
		NewMu.SetRow(k,
			mat.Row(nil, 0, CentroidKMean))
	}

	return NewMu
}

func J(idx *mat.Vector, X, Mu mat.Matrix) float64 {
	Mux := ConstructXCentroidMatrix(idx, Mu)
	xRows, xCols := X.Dims()

	Diff := mat.NewDense(xRows, xCols, nil)
	Diff.Sub(X, Mux)
	Diff.MulElem(Diff, Diff)
	Diff = rowSum(Diff).(*mat.Dense)

	return columnSum(Diff).At(0, 0) / float64(xRows)
}
