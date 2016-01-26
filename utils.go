package kmeans

import (
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
