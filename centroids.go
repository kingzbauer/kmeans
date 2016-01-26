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
