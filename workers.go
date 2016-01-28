// A simple producer -> consumer architecture to be used to dispatch computational
// work over multiple threads

package kmeans

var (
	workChannel     chan *AppIter
	finishedChannel chan *AppIter
	doneChannel     chan bool
)

func init() {
	// initialize the channels
	workChannel = make(chan *AppIter, 1)
	finishedChannel = make(chan *AppIter, 1)
	doneChannel = make(chan bool)
}

// Picks work from the workChannel
// Onces finished, moves it to the finished channel
func worker() {
	for work := range workChannel {
		work.run()
		// work done, send it to the finishedChannel
		finishedChannel <- work
	}
}

func collectorWorker(app *KmeansApp) {
	counter := 0
	for work := range finishedChannel {
		counter++
		app.Iters = append(app.Iters, work)
		if app.BestIter == nil {
			app.BestIter = work
		} else {
			if work.currentCost < app.BestIter.currentCost {
				app.BestIter = work
			}
		}

		if counter == app.RandInitializations {
			break
		}
	}

	// signal that we are done here
	doneChannel <- true
}

func closeChannels() {
	close(workChannel)
	close(finishedChannel)
	close(doneChannel)
}
