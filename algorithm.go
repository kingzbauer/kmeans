package kmeans

import (
	mat "github.com/gonum/matrix/mat64"
)

// AppIter holds the result of making one run through the whole algorithm as defined by
// KmeansApp.OuterIters
type AppIter struct {
	App         *KmeansApp
	mu          mat.Matrix
	idx         *mat.Vector
	currentCost float64
}

// The main kmeans app with the needed parameters
type KmeansApp struct {
	X                   mat.Matrix
	K                   int
	RandInitializations int
	InnerIters          int
	Iters               []*AppIter
	BestIter            *AppIter
}

// NewApp returns a pointer to a new KmeansApp
func NewApp(X mat.Matrix, randInitializations, InnerIters int) *KmeansApp {
	return &KmeansApp{
		X:                   X,
		RandInitializations: randInitializations,
		InnerIters:          InnerIters,
	}
}

// A wrapper around `kmeans.J`
func (appIter *AppIter) J() float64 {
	return J(appIter.idx, appIter.App.X, appIter.mu)
}

// Walk through the Kmeans steps
func (appIter *AppIter) run() {
	// 1. Initialize centroids
	appIter.mu = InitializeCentroids(
		appIter.App.K,
		appIter.App.X,
	)
	// initial assignments
	appIter.idx = AssignCentroid(appIter.App.X, appIter.mu)
	prevCost := appIter.J()

	// loop to get idx and mu values at convergence or App.InnerIters whichever
	// comes first
	// If InnerIters is lower than 1, run until convergence
	if appIter.App.InnerIters > 0 {
		for i := 1; i < appIter.App.InnerIters; i++ {
			appIter.singleRun()
			if appIter.currentCost == prevCost {
				break
			} else {
				prevCost = appIter.currentCost
			}
		}
	} else {
		for {
			appIter.singleRun()
			if appIter.currentCost == prevCost {
				// We are at convergence
				break
			} else {
				prevCost = appIter.currentCost
			}
		}
	}
}

// Makes a single run through cluster assignment and moving and the cost of the run
func (appIter *AppIter) singleRun() {
	appIter.idx = AssignCentroid(appIter.App.X, appIter.mu)
	appIter.mu = MoveCentroids(appIter.idx, appIter.App.X, appIter.mu)
	appIter.currentCost = appIter.J()
}

// The main starting point of the algorithm
// ```for i = 1 to RandInitializations:
//        randomly initialize k-means
//        run k-means to get 'idx' and 'mu'
//        compute the cost function (distortion) J(idx,mu)
//    pick the clustering that gave us the lowest cost```
func (app *KmeansApp) Run() {
	// make an initial run
	iter := &AppIter{App: app}
	iter.run()
	app.Iters = append(app.Iters, iter)

	bestIteration := iter
	for i := 1; i < app.RandInitializations; i++ {
		iter := &AppIter{App: app}
		iter.run()

		app.Iters = append(app.Iters, iter)

		if iter.currentCost < bestIteration.currentCost {
			bestIteration = iter
		}
	}

	app.BestIter = bestIteration
}
