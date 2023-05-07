import numpy as np
import matplotlib.pyplot as plt
from Renderer import Renderer
from Visualization import Visualization

class EKFSLAM(object):
    """A class for implementing EKF-based SLAM

        Attributes
        ----------
        mu :           The mean vector (numpy.array)
        Sigma :        The covariance matrix (numpy.array)
        R :            The process model covariance matrix (numpy.array)
        Q :            The measurement model covariance matrix (numpy.array)
        XGT :          Array of ground-truth poses (optional, may be None) (numpy.array)
        MGT :          Ground-truth map (optional, may be None)

        Methods
        -------
        prediction :   Perform the prediction step
        update :       Perform the measurement update step
        augmentState : Add a new landmark(s) to the state
        run :          Main EKF-SLAM loop
        render :       Render the filter
    """

    def __init__(self, mu, Sigma, R, Q, XGT = None, MGT = None):
        """Initialize the class

            Args
            ----------
            mu :           The initial mean vector (numpy.array)
            Sigma :        The initial covariance matrix (numpy.array)
            R :            The process model covariance matrix (numpy.array)
            Q :            The measurement model covariance matrix (numpy.array)
            XGT :          Array of ground-truth poses (optional, may be None) (numpy.array)
            MGT :          Ground-truth map (optional, may be None)
        """
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q

        self.XGT = XGT
        self.MGT = MGT

        self.MU = mu
        self.VAR = np.diag(self.Sigma).reshape(3, 1)

        if (self.XGT is not None and self.MGT is not None):
            xmin = min(np.amin(XGT[1, :]) - 2, np.amin(MGT[1, :]) - 2)
            xmax = min(np.amax(XGT[1, :]) + 2, np.amax(MGT[1, :]) + 2)
            ymin = min(np.amin(XGT[2, :]) - 2, np.amin(MGT[2, :]) - 2)
            ymax = min(np.amax(XGT[2, :]) + 2, np.amax(MGT[2, :]) + 2)
            xLim = np.array((xmin, xmax))
            yLim = np.array((ymin, ymax))
        else:
            xLim = np.array((-8.0, 8.0))
            yLim = np.array((-8.0, 8.0))

        self.renderer = Renderer(xLim, yLim, 3, 'red', 'green')

        # Draws the ground-truth map
        if self.MGT is not None:
            self.renderer.drawMap(self.MGT)


        # You may find it useful to keep a dictionary that maps a feature ID
        # to the corresponding index in the mean vector and covariance matrix
        self.mapLUT = {}

    def prediction(self, u):
        """Perform the prediction step to determine the mean and covariance
           of the posterior belief given the current estimate for the mean
           and covariance, the control data, and the process model

            Args
            ----------
            u :  The forward distance and change in heading (numpy.array)
        """

        # TODO: Your code goes here
        mu_pred = np.zeros(3)
        mu_pred[0] = self.mu[0] + u[0]*np.cos(self.mu[2])
        mu_pred[1] = self.mu[1] + u[0]*np.sin(self.mu[2])
        mu_pred[2] = self.angleWrap(self.mu[2] + u[1])

        F = np.eye(3)
        F[0, 2] = -np.sin(self.mu[2])*u[0]
        F[1, 2] = np.cos(self.mu[2])*u[0]

        # Update covariance matrix
        R_hat = np.zeros((3,3))
        R_hat[0:2, 0:2] = self.R[0, 0]
        R_hat[0:2, 2] = self.R[0, 1]
        R_hat[2, 0:2] = self.R[0, 1]
        R_hat[2,2] = self.R[1, 1]

        self.Sigma[0:3, 0:3] = F@self.Sigma[0:3, 0:3]@F.transpose() + R_hat
        self.Sigma[0:3, 3:] = F@self.Sigma[0:3, 3:]
        self.Sigma[3:, 0:3] = self.Sigma[3:, 0:3]@F.transpose()

        # Do prediction update
        self.mu[0:3] = mu_pred

    def update(self, z, id):
        """Perform the measurement update step to compute the posterior
           belief given the predictive posterior (mean and covariance) and
           the measurement data

            Args
            ----------
            z :  The Cartesian coordinates of the landmark
                 in the robot's reference frame (numpy.array)
            id : The ID of the observed landmark (int)
        """
        # TODO: Your code goes here
        k = np.size(self.mu)
        midx = int(self.mapLUT[str(id)])

        H = np.zeros((2, k))
        H[0, 0] = -np.cos(self.mu[2])
        H[0, 1] = -np.sin(self.mu[2])
        H[0, 2] = -np.sin(self.mu[2]) * (z[0] - self.mu[0]) + np.cos(self.mu[2])*(z[1] - self.mu[1])
        H[0, midx] = np.cos(self.mu[2])
        H[0, midx + 1] = np.sin(self.mu[2])

        H[1, 0] = np.sin(self.mu[2])
        H[1, 1] = -np.cos(self.mu[2])
        H[1, 2] = -np.cos(self.mu[2]) * (z[0] - self.mu[0]) - np.sin(self.mu[2])*(z[1] - self.mu[1])
        H[1, midx] = -np.sin(self.mu[2])
        H[1, midx] = np.cos(self.mu[2])

        K = self.Sigma@H.transpose()@np.linalg.inv(H@self.Sigma@H.transpose() + self.Q)
        rotation_mat = np.matrix([[np.cos(self.mu[2]), np.sin(self.mu[2])], [-np.sin(self.mu[2]), np.cos(self.mu[2])]])
        z_pred = np.asarray(rotation_mat@(self.mu[midx:midx+2] - self.mu[0:2])).reshape(-1)
        self.mu = self.mu + K@(z - z_pred)
        self.Sigma = (np.eye(k) - K@H)@self.Sigma

    def augmentState(self, z, id):
        """Augment the state vector to include the new landmark

            Args
            ----------
            z :  The Cartesian coordinates of the landmark
                 in the robot's reference frame (numpy.array)
            id : The ID of the observed landmark
        """

        # TODO: Your code goes here
        k = np.size(self.mu)
        self.mapLUT[str(id)] = int(k)
        rotation_mat = np.matrix([[np.cos(self.mu[2]), np.sin(self.mu[2])], [-np.sin(self.mu[2]), np.cos(self.mu[2])]])
        mhat = np.asarray(rotation_mat.transpose()@z + self.mu[0:2]).reshape(-1)
        self.mu = np.concatenate((self.mu, mhat), axis=0)
        G = np.zeros((2, k))
        G[0, 0] = 1
        G[0, 2] = -np.sin(self.mu[2])*z[0]-np.cos(self.mu[2])*z[1]
        G[1, 1] = 1
        G[1, 2] = np.cos(self.mu[2])*z[0]-np.sin(self.mu[2])*z[1]
        self.Sigma = np.concatenate((np.concatenate((self.Sigma, self.Sigma @ G.transpose()), axis=1),
                                    np.concatenate((G@self.Sigma, G@self.Sigma@G.transpose() + self.Q), axis=1)), axis=0)

    def angleWrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        while theta < -np.pi:
            theta = theta + 2*np.pi

        while theta > np.pi:
            theta = theta - 2*np.pi

        return theta

    def run(self, U, Z):
        """The main loop of EKF-based SLAM

            Args
            ----------
            U :   Array of control inputs, one column per time step (numpy.array)
            Z :   Array of landmark observations in which each column
                  [t; id; x; y] denotes a separate measurement and is
                  represented by the time step (t), feature id (id),
                  and the observed (x, y) position relative to the robot
        """
        # TODO: Your code goes here
        for t in range(np.size(U, 1)):
            # print(t)
            if str(Z[1, t]) in self.mapLUT:  # If we observe known map features
                self.update(Z[2:, t], int(Z[1, t]))
            else:  # If we observe new map features
                self.augmentState(Z[2:, t], int(Z[1, t]))

            self.MU = np.column_stack((self.MU, self.mu[0:3]))
            self.VAR = np.column_stack((self.VAR, np.diag(self.Sigma[0:3, 0:3])))

            # self.MU = np.column_stack((self.MU, self.mu))
            # self.VAR = np.column_stack((self.VAR, np.diag(self.Sigma)))
            # self.renderer.render(self.mu, self.Sigma, self.XGT[1:4, t], Z, self.mapLUT)
            # self.renderer.render(self.mu, self.Sigma, self.XYT[:, t])
        # print(self.mapLUT)
        for t in range(np.size(U, 1)):
            self.renderer.render(self.mu, self.Sigma, self.XGT[1:4, t], Z, self.mapLUT)
        self.renderer.drawTrajectory(self.MU[0:2, :], self.XGT[1:4, :])
        self.renderer.plotError(self.MU, self.XGT[1:4, :], self.VAR)


        # You may want to call the visualization function between filter steps where
        #       self.XGT[1:4, t] is the column of XGT containing the pose the current iteration
        #       Zt are the columns in Z for the current iteration
        #       self.mapLUT is a dictionary where the landmark IDs are the keys
        #                   and the index in mu is the value
        plt.ioff()
        plt.show()
