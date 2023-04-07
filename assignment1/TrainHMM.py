import sys
import pickle
from HMM import HMM
from DataSet import DataSet
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    """Read in data, call code to train HMM, and save model."""

    # This function should be called with one argument: trainingdata.txt
    if (len(sys.argv) != 2):
        # print("Usage: TrainMM.py trainingdata.txt")ZX
        print("Usage: TrainMM.py randomwalk.train.txt")
        sys.exit(0)

    dataset = DataSet(sys.argv[1])
    dataset.readFile()

    max_iter = 10
    hmm = HMM(dataset.envShape)
    loglike = hmm.train(dataset.observations, max_iter)

    # Save the model for future use
    fileName = "trained-model.pkl"
    print("Saving trained model as " + fileName)
    pickle.dump({'T': hmm.T, 'M': hmm.M, 'pi': hmm.pi}, open(fileName, "wb"))

    # Save the log likelihood to determine convergence
    pngName = "loglikelihood.png"
    plt.plot(np.arange(max_iter) + 1, loglike)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.title("Baum-Welch Algroithm")
    plt.savefig(pngName)

