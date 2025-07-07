import matplotlib.pyplot as plt
import numpy as np
import os
from RandomMatrixKnownDistributions import FunctionName2Distribution

def PlotDistributionEigenvalues(Eigenvalues, FunctionName , Q,Lmin, Lmax,sigma,bins,PlotDir,FunctionName2Distribution = FunctionName2Distribution):
    """
        @params: Eigenvalues: Array of eigenvalues.
        @params: FunctionName2Distribution: Function 2 be applied to bins.
        @params: Lmin: Minimum eigenvalue.
        @params: Lmax: Maximum eigenvalue.
        @params: bins: Number of bins.
        @describe: Plot the distribution of the eigenvalues
    """
    if FunctionName not in FunctionName2Distribution.keys():
        raise ValueError(f"FunctionName {FunctionName} not in {FunctionName2Distribution.keys()}")
    Eigenvalues = Eigenvalues[Eigenvalues>0.000001]
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    n,bins = np.histogram(Eigenvalues, bins)
    bins_centers = (bins[1:] + bins[:-1])/2
    ax.scatter(bins_centers,n/np.sum(n), alpha=0.5, label='Empirical distribution')
    ax.plot(bins_centers,FunctionName2Distribution[FunctionName](bins_centers,Q,Lmin,Lmax,sigma), label=FunctionName)
    ax.vlines(Lmin, 0, 1, colors='r',linestyles='dashed')
    ax.vlines(Lmax, 0, 1, colors='r',linestyles='dashed')
    ax.set_xlim(0)
    ax.set_ylim(0,1.1)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$P(\lambda)$")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(PlotDir,"ComparisonDistribution_" + FunctionName + ".png"))
    plt.show()
    plt.close()



def PlotDyadicExpansion(DyadicExpansion,NumberEigenvalues,PlotDir):
    """
        @params: DyadicExpansion: Matrix N x N
        @describe: Plot the dyadic expansion of the correlation matrix
    """
    if NumberEigenvalues%2==0:
        Axisx = 2
        Axisy = int(NumberEigenvalues/2)
    else:
        Axisx = 2
        Axisy = int(NumberEigenvalues//2 + 1)
    fig,ax = plt.subplots(Axisx,Axisy,figsize = (10,10))
    for i in range(NumberEigenvalues):
        ky = int(i/Axisy)
        Indexx = ky
        Indexy = i - ky*Axisy 
        cax = ax[Indexx,Indexy].imshow(DyadicExpansion[-i], cmap='viridis', aspect='auto')
    # Add colorbar
        cbar = fig.colorbar(cax)
        ax[Indexx,Indexy].set_title(r"$\lambda_{{{}}}$".format(i))

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(os.path.join(PlotDir,"DyadicExpansion.png"))
    plt.show()
    plt.close()