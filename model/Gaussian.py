import numpy as np
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from functools import lru_cache

class GaussianData():

    def __init__(self,
                 n,
                 d=1,
                 muX=0,
                 muY=0,
                 rho=0.9,
                 prior=1/2,
                 rng=None,
                 seed=None,
                 device=None):
        """
        Construct n i.i.d. samples of jointly gaussian (X, Y) for mutual
        information estimation.

        The model for data is
            X:=(x1, x2, ..., xd)
            Y:=(y1, y2, ..., yd)
        where (xi, yi)'s are i.i.d. jointly gaussian with
            mean: (muX, muY)
            covariance:
                [[1, rho],
                 [rho, 1]]

        Parameters:
        -----------
        n: int
            Number of samples.
        d: int
            Number of elements of X (and Y).
        muX: float
            Mean of each element xi, same for all i.
        muY: float
            Mean of each element yi, same for all i.
        rho: float within (-1, 1)
            Correlation between xi and yi.
        rng: numpy.random._generator.Generator
            For generating the random samples.
        seed:
            If rng is None, it is set to numpy.random.default_rng(seed).
        device:
            Default device to use for samples of X and Y.
        """
        self.d, self.n, self.rho, self.device = d, n, rho, device
        rng = rng or np.random.default_rng(seed)
        self.sampler = lambda *args, **kwargs: rng.multivariate_normal(
            (muX, muY), [[1, rho], [rho, 1]], *args, **kwargs)
        self.resample()

    def to(self, *args, **kwargs):
        """
        Move and/or casts the tensors self.X and self.Y.

        Calls torch.Tensor.to(*args, **kwargs) on self.X and self.Y.
        """
        self._X = self._X.to(*args, **kwargs)
        self._Y = self._Y.to(*args, **kwargs)

    def resample(self):
        """
        Resamples X and Y.
        """
        self._XY = self.sampler(self.n * self.d).reshape(self.n, self.d, 2,
                                               order='F').transpose(0, 2, 1)
        self._X = Tensor(self._XY[:, 0, :])
        self._Y = Tensor(self._XY[:, 1, :])
        if self.device is not None:
            self.to(self.device)
        return self

    @property
    def X(self):
        """
        Tensor: Samples of X.
            Dimension 0: samples of X
            Dimension 1: features of X
        """
        return self._X

    @property
    def Y(self):
        """
        Tensor: Samples of Y.
            Dimension 0: samples of X
            Dimension 1: features of X
        """
        return self._Y

    # @lru_cache
    def mutual_information(self):
        """
        Returns the ground truth mutual information I(X;Y).
        """
        return -0.5 * np.log(1 - self.rho**2) * self.d

    IXY = property(mutual_information)

    def pointwise_mutual_information(self, x, y):
        """
        Returns the log density ration pXY/(pX*pY).

        Parameters:
        -----------
        x: Tensor
            Dimension 0: samples of X
            Dimension 1: features of X
        y: Tensor
            Dimension 0: samples of X
            Dimension 1: features of X
        """
        c = self.rho / (2 * (1 - self.rho**2))
        return self.mutual_information() + c * (2 * x * y - self.rho *
                                                (x**2 + y**2)).sum(dim=1)

    iXY = pointwise_mutual_information

    def mutual_information_std(self, size=1000000):
        """
        Returns the standard deviation of the sample average of pointwise mutual information.

        This is the variation in the estimate attributed solely to the number of samples limited to n, 
        since true log density ratio is used to compute the sample average.
        """
        xy = self.sampler(size)
        x = xy[:, 0]
        y = xy[:, 1]
        delta_squared = (2 * x * y - self.rho * (x**2 + y**2))**2
        m, s = delta_squared.mean(), delta_squared.std(ddof=1)
        c = (self.d / self.n)**0.5 * self.rho / (2 * (1 - self.rho**2))
        print(f'percentage error: +/- {2*s/m/(size)**0.5 * 100:.1g} %')
        return m * c

    IXY_std = property(mutual_information_std)

    def plot(self, dimX=0, dimY=0, ax=None, hide_details=False, **kwargs):
        """
        Scatter plot of X[dimX] and Y[dimY].

        Parameters:
        -----------
        dimX: int
            The dimension of X to plot along the x-axis.
        dimY: int
            The dimension of X to plot along the y-axis.
        ax: axis
            The axis to plot the data
        hide_details: bool
            Add xlabel, ylabel, and title if false.
        """
        if ax is None:
            ax = plt.gca()
        if not hide_details:
            ax.set(xlabel=f'$\\mathsf{{x}}_{{{dimX}}}$',
                   ylabel=f'$\\mathsf{{y}}_{{{dimY}}}$',
                   title=f'Data samples')
        ax.scatter(self._XY[:, 0, dimX], self._XY[:, 1, dimY], **kwargs)