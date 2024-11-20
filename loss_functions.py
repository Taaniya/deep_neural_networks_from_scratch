import numpy as np


def l1(yhat, y):
        """
        Args:
        yhat: vector of size m (predicted labels)
        y: vector of size m (true labels)

        Returns:
        loss: the value of the L1 loss function
        """
        diff = abs(np.subtract(y.reshape(-1, 1), yhat.reshape(-1, 1)))  # shape = (m,1)
        loss = np.sum(diff, axis=0)     # shape = (1,)
        return loss[0]      # scalar value


def l2(yhat, y):
        """
        Args:
        yhat: vector of size m (predicted labels)
        y: vector of size m (true labels)

        Returns:
        loss: the value of the L2 loss function
        """
        diff = np.subtract(y.reshape(-1, 1), yhat.reshape(-1, 1))  # shape = (m,1)
        loss = np.dot(diff.T, diff)     # shape = (1,1)
        return loss[0][0]       # scalar value

