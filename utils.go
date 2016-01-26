package kmeans

import (
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
