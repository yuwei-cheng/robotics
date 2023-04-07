import numpy as np
from DataSet import DataSet
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
class HMM(object):
    """A class for implementing HMMs.

    Attributes
    ----------
    envShape : list
        A two element list specifying the shape of the environment
    states : list
        A list of states specified by their (x, y) coordinates
    observations : list
        A list specifying the sequence of observations
    T : numpy.ndarray
        An N x N array encoding the transition probabilities, where
        T[i,j] is the probability of transitioning from state i to state j.
        N is the total number of states (envShape[0]*envShape[1])
    M : numpy.ndarray
        An M x N array encoding the emission probabilities, where
        M[k,i] is the probability of observing k from state i.
    pi : numpy.ndarray
        An N x 1 array encoding the prior probabilities

    Methods
    -------
    train(observations)
        Estimates the HMM parameters using a set of observation sequences
    viterbi(observations)
        Implements the Viterbi algorithm on a given observation sequence
    setParams(T, M, pi)
        Sets the transition (T), emission (M), and prior (pi) distributions
    getParams
        Queries the transition (T), emission (M), and prior (pi) distributions
    sub2ind(i, j)
        Convert integer (i,j) coordinates to linear index.
    """

    def __init__(self, envShape, T=None, M=None, pi=None):
        """Initialize the class.

        Attributes
        ----------
        envShape : list
            A two element list specifying the shape of the environment
        T : numpy.ndarray, optional
            An N x N array encoding the transition probabilities, where
            T[i,j] is the probability of transitioning from state i to state j.
            N is the total number of states (envShape[0]*envShape[1])
        M : numpy.ndarray, optional
            An M x N array encoding the emission probabilities, where
            M[k,i] is the probability of observing k from state i.
        pi : numpy.ndarray, optional
            An N x 1 array encoding the prior probabilities
        """
        self.envShape = envShape
        self.numStates = envShape[0] * envShape[1]

        if T is None:
            # Initial estimate of the transition function
            # where T[sub2ind(i',j'), sub2ind(i,j)] is the likelihood
            # of transitioning from (i,j) --> (i',j')
            self.T = np.zeros((self.numStates, self.numStates))

            # Self-transitions
            for i in range(self.numStates):
                self.T[i, i] = 0.2

            # Black rooms
            self.T[self.sub2ind(0, 0), self.sub2ind(0, 0)] = 1.0
            self.T[self.sub2ind(1, 1), self.sub2ind(1, 1)] = 1.0
            self.T[self.sub2ind(0, 3), self.sub2ind(0, 3)] = 1.0
            self.T[self.sub2ind(3, 2), self.sub2ind(3, 2)] = 1.0

            # (1, 0) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(1, 0)] = 0.8

            # (2, 0) -->
            self.T[self.sub2ind(1, 0), self.sub2ind(2, 0)] = 0.8/3.0
            self.T[self.sub2ind(2, 1), self.sub2ind(2, 0)] = 0.8/3.0
            self.T[self.sub2ind(3, 0), self.sub2ind(2, 0)] = 0.8/3.0

            # (3, 0) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(3, 0)] = 0.8/2.0
            self.T[self.sub2ind(3, 1), self.sub2ind(3, 0)] = 0.8/2.0

            # (0, 1) --> (0, 2)
            self.T[self.sub2ind(0, 2), self.sub2ind(0, 1)] = 0.8

            # (2, 1) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(2, 1)] = 0.8/3.0
            self.T[self.sub2ind(3, 1), self.sub2ind(2, 1)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(2, 1)] = 0.8/3.0

            # (3, 1) -->
            self.T[self.sub2ind(2, 1), self.sub2ind(3, 1)] = 0.8/2.0
            self.T[self.sub2ind(3, 0), self.sub2ind(3, 1)] = 0.8/2.0

            # (0, 2) -->
            self.T[self.sub2ind(0, 1), self.sub2ind(0, 2)] = 0.8/2.0
            self.T[self.sub2ind(1, 2), self.sub2ind(0, 2)] = 0.8/2.0

            # (1, 2) -->
            self.T[self.sub2ind(0, 2), self.sub2ind(1, 2)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(1, 2)] = 0.8/3.0
            self.T[self.sub2ind(1, 3), self.sub2ind(1, 2)] = 0.8/3.0

            # (2, 2) -->
            self.T[self.sub2ind(1, 2), self.sub2ind(2, 2)] = 0.8/3.0
            self.T[self.sub2ind(2, 1), self.sub2ind(2, 2)] = 0.8/3.0
            self.T[self.sub2ind(2, 3), self.sub2ind(2, 2)] = 0.8/3.0

            # (1, 3) -->
            self.T[self.sub2ind(1, 2), self.sub2ind(1, 3)] = 0.8/2.0
            self.T[self.sub2ind(2, 3), self.sub2ind(1, 3)] = 0.8/2.0

            # (2, 3) -->
            self.T[self.sub2ind(1, 3), self.sub2ind(2, 3)] = 0.8/3.0
            self.T[self.sub2ind(3, 3), self.sub2ind(2, 3)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(2, 3)] = 0.8/3.0

            # (3, 3) --> (2, 3)
            self.T[self.sub2ind(2, 3), self.sub2ind(3, 3)] = 0.8
        else:
            self.T = T

        if M is None:
            # Initial estimates of emission likelihoods, where
            # M[k, sub2ind(i,j)]: likelihood of observation k from state (i, j)
            self.M = np.ones((4, 16)) * 0.1

            # Black states
            self.M[:, self.sub2ind(0, 0)] = 0.25
            self.M[:, self.sub2ind(1, 1)] = 0.25
            self.M[:, self.sub2ind(0, 3)] = 0.25
            self.M[:, self.sub2ind(3, 2)] = 0.25

            self.M[self.obs2ind('r'), self.sub2ind(0, 1)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(0, 2)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(1, 0)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(1, 2)] = 0.7
            self.M[self.obs2ind('r'), self.sub2ind(1, 3)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(2, 0)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(2, 1)] = 0.7
            self.M[self.obs2ind('r'), self.sub2ind(2, 2)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(2, 3)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(3, 0)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(3, 1)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(3, 3)] = 0.7
        else:
            self.M = M

        if pi is None:
            # Initialize estimates of prior probabilities where
            # pi[(i, j)] is the likelihood of starting in state (i, j)
            self.pi = np.ones((16, 1))/12
            self.pi[self.sub2ind(0, 0)] = 0.0
            self.pi[self.sub2ind(1, 1)] = 0.0
            self.pi[self.sub2ind(0, 3)] = 0.0
            self.pi[self.sub2ind(3, 2)] = 0.0
        else:
            self.pi = pi

    def setParams(self, T, M, pi):
        """Set the transition, emission, and prior probabilities."""
        self.T = T
        self.M = M
        self.pi = pi

    def getParams(self):
        """Get the transition, emission, and prior probabilities."""
        return (self.T, self.M, self.pi)

    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data
    def train(self, observations, max_iter):
        """Estimate HMM parameters from training data via Baum-Welch.

        Parameters
        ----------
        observations : list
            A list specifying a set of observation sequences
            where observations[i] denotes a distinct sequence
        max_iter: integer
            An integer that defines the maximum iteration
            that we train the algorithm
        """
        # This function should set self.T, self.M, and self.pi

        loglike = np.zeros(max_iter)  # store log probabilities
        for iter in tqdm(range(max_iter)):
            pi = np.zeros((16, 1))
            T = np.zeros((self.numStates, self.numStates))
            M = np.zeros((4, 16))

            for i in range(len(observations)):
                alpha = self.forward(observations[i])
                loglike[iter] = loglike[iter] + np.log(np.sum(alpha[-1, :])) / len(observations)  # p(z^T, \lambda') = \sum_{x_T}\alpha_T(x_T)

                beta = self.backward(observations[i])
                gamma = self.computeGamma(alpha, beta)
                xi = self.computeXis(alpha, beta, observations[i])

                pi = pi + gamma[0, :][:, np.newaxis]/len(observations) # \pi'_i = \gamma_{0}(i)
                T = T + self.updateT(gamma, xi)/len(observations)
                M = M + self.updateM(gamma, observations[i]) /len(observations)

            self.setParams(T, M, pi)
        return loglike

    def updateT(self, gamma, xi):
        """"Update transition matrix"""
        T = np.zeros((self.numStates, self.numStates))
        for ri in range(4):
            for ci in range(4):
                for rj in range(4):
                    for cj in range(4):
                        if np.sum(gamma[:(-1), self.sub2ind(ri, ci)]) == 0:
                            T[self.sub2ind(rj, cj), self.sub2ind(ri, ci)] = 0
                        else:
                            T[self.sub2ind(rj, cj), self.sub2ind(ri, ci)] = np.sum(xi[:, self.sub2ind(ri, ci), self.sub2ind(rj, cj)]) / np.sum(gamma[:(-1), self.sub2ind(ri, ci)])

        return T

    def updateM(self, gamma, z):
        """"Update measurement matrix"""
        M = np.zeros((4, 16))
        for ri in range(4):
            for ci in range(4):
                for m in ["r", "g", "b", "y"]:
                    if np.sum(gamma[:, self.sub2ind(ri, ci)]) == 0:
                        M[self.obs2ind(m), self.sub2ind(ri, ci)] = 0
                    else:
                        M[self.obs2ind(m), self.sub2ind(ri, ci)] = np.sum(gamma[:, self.sub2ind(ri, ci)]*np.where(np.array(z) == m, 1, 0))/ np.sum(gamma[:, self.sub2ind(ri, ci)])

        return M

    def viterbi(self, observations):
        """Implement the Viterbi algorithm.

        Parameters
        ----------
        observations : list
            A list specifying the sequence of observations, where each o
            observation is a string (e.g., 'r')

        Returns
        -------
        states : list
            List of predicted sequence of states, each specified as (x, y) pair
        """
        # CODE GOES HERE
        # Return the list of predicted states, each specified as (x, y) pair
        delta = np.zeros((len(observations), self.numStates))
        pre = np.zeros((len(observations), self.numStates))
        # initialization
        sequence = []

        for r in range(4):
            for c in range(4):
                delta[0, self.sub2ind(r, c)] = self.M[self.obs2ind(observations[0]), self.sub2ind(r, c)] * self.pi[self.sub2ind(r, c)]

        # Repeat
        for t in range(1, len(observations), 1):
            for r in range(4):
                for c in range(4):
                    temp = self.T[self.sub2ind(r, c), :] * delta[t-1, :]
                    # print(f"The length of temp is {len(temp)}")
                    # print(temp)
                    # print(f"The argmax of temp is {np.argmax(temp)}")
                    delta[t, self.sub2ind(r, c)] = self.M[self.obs2ind(observations[t]), self.sub2ind(r, c)] * np.max(temp)
                    pre[t, self.sub2ind(r, c)] = np.argmax(temp)

        # Select the most likely terminal state
        sequence.append(self.ind2sub(np.argmax(delta[len(observations)-1, :])))

        # Determine the most likely sequence
        for t in range(len(observations)-2, -1, -1):
            # print(sequence)
            # print(self.sub2ind(sequence[-1][0], sequence[-1][1]))
            sequence.append(self.ind2sub(pre[t+1, self.sub2ind(sequence[-1][0], sequence[-1][1])]))

        return sequence[::-1] # reverse the order

    def forward(self, z):
        """Implement one forward step."""
        alpha = np.zeros((len(z), 16))
        # When t = 0
        for r in range(4):
            for c in range(4):
                # \alpha_0(i) = P(z0 | X0=i) * P(X0 = i)
                alpha[0, self.sub2ind(r, c)] = self.M[self.obs2ind(z[0]), self.sub2ind(r, c)] * self.pi[self.sub2ind(r, c)]

        # When t in 1 to 199
        for t in range(1, len(z), 1):
            for r in range(4):
                for c in range(4):
                    # \alpha_t(i) = P(z_t | X_t = i) * \sum_{x_{t-1}} p(Xt = i | X_{t-1}) * alpha_{t-1}(i)
                    alpha[t, self.sub2ind(r, c)] = self.M[self.obs2ind(z[t]), self.sub2ind(r, c)] * np.sum(self.T[self.sub2ind(r, c), :] * alpha[t - 1, :])

        # normalize alpha by column
        # print(np.sum(alpha, axis = 0))
        return alpha

    def backward(self, z):
        """Implement one backward step"""
        # Set beta_T(i) = 1
        beta = np.ones((len(z), 16))
        # t in T-1 to 0
        for t in range(len(z)-2, -1, -1):
            for r in range(4):
                for c in range(4):
                    # \beta_{t}(i) = \sum_{X_{t+1}} P(Z_{t+1}|X_{t+1})P(X_{t+1}|X_t = i)\beta_{t+1}(X_{t+1})
                    beta[t, self.sub2ind(r, c)] = np.sum(self.M[self.obs2ind(z[t+1]), :] * self.T[self.sub2ind(r, c), :] * beta[t+1, :])
        return beta

    def computeGamma(self, alpha, beta):
        """Compute P(X[t] | Z^T)."""
        # CODE GOES HERE
        norm = np.sum(alpha, axis=1)
        normalized_alpha = alpha / norm[:, np.newaxis]
        normalized_beta = beta / norm[:, np.newaxis]
        gamma = normalized_alpha * normalized_beta
        gamma_norm = np.sum(gamma, axis=1)
        gamma = gamma / gamma_norm[:, np.newaxis]
        return gamma

    def computeXis(self, alpha, beta, z):
        """Compute xi as an array comprised of each xi-xj pair."""
        # \xi_t(i, j) \sim \alpha_t(i) * P(X_{t+1} = j | X_t = i)P(Z_{t+1} | X_{t+1} = j)\beta_{t+1}(j)
        norm = np.sum(alpha, axis=1)
        alpha = alpha / norm[:, np.newaxis]
        beta = beta / norm[:, np.newaxis]
        xi = np.zeros((len(z)-1, 16, 16))
        for t in range(len(z) - 1):
            for ri in range(4):
                for ci in range(4):
                    for rj in range(4):
                        for cj in range(4):
                            xi[t, self.sub2ind(ri, ci), self.sub2ind(rj, cj)] = alpha[t, self.sub2ind(ri, ci)] * self.T[self.sub2ind(ri, ci), self.sub2ind(rj, cj)]*self.M[self.obs2ind(z[t+1]), self.sub2ind(rj, cj)]*beta[t+1, self.sub2ind(rj, cj)]
            # to make sure sum of them = 1
            xi[t, :, :] = xi[t, :, :] / np.sum(xi[t, :, :])
        return xi

    def getLogStartProb(self, state):
        """Return the log probability of a particular state."""
        return np.log(self.pi[state])

    def getLogTransProb(self, fromState, toState):
        """Return the log probability associated with a state transition."""
        return np.log(self.T[toState, fromState])

    def getLogOutputProb(self, state, output):
        """Return the log probability of a state-dependent observation."""
        return np.log(self.M[output, state])

    def sub2ind(self, i, j):
        """Convert subscript (i,j) to linear index."""
        return (self.envShape[1]*i + j)

    def ind2sub(self, idx):
        """Convert linear index to subscript(i, j)"""
        j = int(idx % self.envShape[1])
        idx = idx - j
        i = int(idx / self.envShape[1])
        return (i, j)

    def obs2ind(self, obs):
        """Convert observation string to linear index."""
        obsToInt = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
        return obsToInt[obs]

# if __name__ == '__main__':
#     dataset = DataSet("C:/Users/yuwei/Desktop/robotics/assignment1/data/randomwalk.train.txt")
#     dataset.readFile()
#     hmm = HMM(dataset.envShape)
#     # print(hmm.sub2ind(2,3))
#     # print(hmm.ind2sub(11))
#     T, M, pi = hmm.getParams()
#     delta = np.zeros((201, 16))
#     pre = np.zeros((201, 16))
#     # initialization
#     sequence = []
#
#     for r in range(4):
#         for c in range(4):
#             delta[0, hmm.sub2ind(r, c)] = M[hmm.obs2ind(dataset.observations[0][0]), hmm.sub2ind(r, c)] * pi[hmm.sub2ind(r, c)]
#     # print(delta[0, :])
#
#     for r in range(4):
#         for c in range(4):
#             print(T[hmm.sub2ind(r, c), :])
#             print(delta[0, :])
#             temp = T[hmm.sub2ind(r, c), :] * delta[0, :]
#             # print(f"The length of temp is {len(temp)}")
#             print(temp)
#             print(f"The argmax of temp is {np.argmax(temp)}")
#             M[hmm.obs2ind(dataset.observations[0][1]), hmm.sub2ind(r, c)] * np.max(temp)
#             pre[1, hmm.sub2ind(r, c)] = np.argmax(temp)
    # print(T[hmm.sub2ind(2, 3), :])

    # print("===================Start Training===============================")
    # start = time.time()
    # loglike = hmm.train(dataset.observations, 5)
    # end = time.time()
    # print(f"Training takes {end - start} seconds")
    # T, M, pi = hmm.getParams()
    # plt.plot(loglike)
    # plt.show()



