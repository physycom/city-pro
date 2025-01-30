import numpy as np  

def ComputeCorrelationMatrix(Signals,case):
    """
        @params: Signals: Matrix N x T
        @params: case: 1 -> Shifted, 2 -> Shifted and Scaled
        @describe: Compute the correlation matrix of the signals
    """
    if case == "shifted":
        Signals_shift = Signals - np.nanmean(Signals, axis=1, keepdims=True)
        CorrelationMatrix = np.corrcoef(Signals_shift)
        return CorrelationMatrix,Signals_shift
    elif case == "shifted_scaled":
        Signals_shift = Signals - np.nanmean(Signals, axis=1, keepdims=True)
        Signals_shift_scaled = Signals_shift/ np.nanstd(Signals )
        CorrelationMatrix = np.corrcoef(Signals_shift_scaled)
        return CorrelationMatrix,Signals_shift_scaled

def ComputeEigenvectors(CorrelationMatrix):
    """
        @params: CorrelationMatrix: Matrix N x N
        @describe: Compute the eigenvectors of the correlation matrix
    """
    eigvals, eigvecs = np.linalg.eig(CorrelationMatrix)
    return eigvals, eigvecs

def SortEigvalsEigVec(eigenvalues,eigenvectors):
    """
    Sort eigenvalues and eigenvectors, keeping the indices matched.
    :param eigenvalues: Array of eigenvalues.
    :param eigenvectors: Matrix of eigenvectors.
    :return: Sorted eigenvalues, sorted eigenvectors, idxeigvals2idxordereigvals, idxordereigvals2idxeigvals
    """
    # Get the sorting indices
    IdxEigvals2IdxOrderEigvals = np.argsort(eigenvalues)
    # Sort eigenvalues and eigenvectors
    SortedEigenvalues = eigenvalues[IdxEigvals2IdxOrderEigvals]
    SortedEigenvectors = eigenvectors[:, IdxEigvals2IdxOrderEigvals]
    # Create the inverse mapping
    IdxOrderEigvals2IdxEigvals = np.argsort(IdxEigvals2IdxOrderEigvals)
    return SortedEigenvalues,SortedEigenvectors,IdxEigvals2IdxOrderEigvals,IdxOrderEigvals2IdxEigvals


def ComputeDyadicExpansion(Eigenvalues,Eigenvectors):
    """
        @params: Eigenvalues: Array of eigenvalues.
        @params: Eigenvectors: Matrix of eigenvectors.
        @describe: Compute the dyadic expansion of the correlation matrix
    """
    N = len(Eigenvalues)
    DyadicExpansion = np.zeros((N,N,N))
    for i in range(N):
        DyadicExpansion[i] = Eigenvalues[i] * np.outer(Eigenvectors[:,i],Eigenvectors[:,i])
    return DyadicExpansion


def MarchenkoPasturLimits(N,T,sigma):
    """
        @params: Eigenvalues: Array of eigenvalues.
        @describe: Compute the Marchenko-Pastur limits
        It consists of:
            \lambda_min = sigma^2 * (1 + 1/Q - sqrt(1/Q))
            \lambda_max = sigma^2 * (1 + 1/Q + sqrt(1/Q))
        Where Q = T/N if N > T, Q = N/T if T > N in [0,1]

    """
    if N > T:
        Q = N/T
    else:
        Q = T/N

    LambdaMinus = sigma**2 * (1 + 1/Q - np.sqrt(1/Q))**2
    LambdaPlus = sigma**2 * (1 + 1/Q + np.sqrt(1/Q))**2
    return LambdaMinus,LambdaPlus