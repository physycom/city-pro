import numpy as np


def MarchenkoPastur(bins_centers,Q,Lmin,Lmax,sigma):
    """
        @params: bins_centers: Array of bins centers.
        @params: Q: Number of samples.
        @params: Lmin: Minimum eigenvalue.
        @params: Lmax: Maximum eigenvalue.
        @params: sigma: Standard deviation.
        @describe: Compute the Marchenko-Pastur distribution
    """
    bins_centers = np.array(bins_centers)
    Denominator = 2*np.pi*sigma**2*bins_centers
    Numerator = np.sqrt((Lmax - bins_centers)*(bins_centers - Lmin))*Q
    return Numerator/Denominator

def SemiCircle(bins_centers,Lmin = 2,Lmax = 2):
    """
        @params: bins_centers: Array of bins centers.
        @params: Lmin: Minimum eigenvalue.
        @params: Lmax: Maximum eigenvalue.
        @describe: Compute the SemiCircle distribution
        NOTE: It is the Marchenko-Pastur distribution with Q = 1, sigma = 1
    """
    bins_centers = np.array(bins_centers)
    return np.sqrt((Lmax - bins_centers)*(bins_centers - Lmin))/(np.pi*(Lmax - Lmin))


FunctionName2Distribution = {"Marchenko-Pastur": MarchenkoPastur}