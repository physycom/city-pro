import numpy as np
import numba
from RandomMatrixMethods import *
from PlotRandomMatrices import *
#@numba.jit("[float64,float64]",parallel=True)
def build_matrix_from_difference_vector(phi1, phi2):
    """
    Given two vectors, return a matrix with the differences of the vectors in each row.
    
    @param phi1: The first vector (field on the indices of the vector)
    @param phi2: The second vector
    @return: A matrix where each element (i, j) is phi1[i] - phi2[j]
    """
    phi1 = np.asarray(phi1)
    phi2 = np.asarray(phi2)
    difference_matrix = phi1[:, np.newaxis] - phi2[np.newaxis, :]
    return difference_matrix    
    
class AnalysisCorrelationRandomMatrix:
    """
        NOTE: We call Signals, Si(t) in the description. Where i stands for sources, t time.
        @param Matrix: Matrix to be analyzed (N x T)
        This class is intended to evaluate from time series of signals correlation matrix and estimate 
        the nature of correlations by relating the spectrum of the correlation matrix to features of random matrix theory.
        Tracy-Widom -> distribution of largest eigenvalue 
    """
    def __init__(self,Signals,PlotDir):
        # Matrix N x T
        self.Signals = Signals
        # 
        if isinstance(Signals, np.ndarray):
            self.N = Signals.shape[0]
            self.T = Signals.shape[1]
        else:
            self.N = len(Signals)
            self.T = len(Signals[0])
        # NOTE: The largest eigenvalue must be always be positive. (Tracy-Widom limit)
        if self.N > self.T:
            self.LambdaMinus = np.sqrt(self.N) - np.sqrt(self.T)
            self.LambdaPlus = np.sqrt(self.N) + np.sqrt(self.T)
            self.Q = self.N/self.T

        else:
            self.LambdaMinus = np.sqrt(self.T) - np.sqrt(self.N)
            self.LambdaPlus = np.sqrt(self.T) + np.sqrt(self.N)
            self.Q = self.T/self.N

        # Array len = T
        self.AvgCorr_t = None
        # Array len = T
        self.Variance_t = None 
        self.PlotDir = PlotDir         

#### ----------------- Preprocessing ----------------- ####
    def ComputeCorrelationMatrix(self):
        """
            @describe:
                self.Signal_shifted: Si(t) - <Si(t)>
                self.Signal_shifted_scaled: (Si(t) - <Si(t)>)/std(Si(t))
            NOTE: The correlation matrices are the Wishart matrices that are the basis of the analysis.

        """
        self.CorrelationShifted, self.Signals_shift = ComputeCorrelationMatrix(self.Signals,"shifted")
        
        self.CorrelationMatrixShiftScaled, self.Signals_shift_scaled = ComputeCorrelationMatrix(self.Signals,"shifted_scaled")

    def ComputeEigenvectorsAndEigenValues(self):
        """
            @describe:
                Compute the eigenvectors of the correlation matrix
                Counts the number of zero eigenvalues.
                NOTE: An important feature of this analysis is the fact that the matrix is highly not maximal rank.
                Since either the sources or the time intervals are different .
        """
        self.EigenvaluesShiftedScaled,self.EigenvectorsShiftedScaled = ComputeEigenvectors(self.CorrelationMatrixShiftScaled)
        self.EigenvaluesShifted,self.EigenvectorsShifted = ComputeEigenvectors(self.CorrelationShifted)
        self.NZeros = np.sum(self.EigenvaluesShiftedScaled == 0)


####------------------ Compute Correlations ----------------- ####    

    def SortEigvalsEigVec(self):
        """
        Sort eigenvalues and eigenvectors, keeping the indices matched.
        :param eigenvalues: Array of eigenvalues.
        :param eigenvectors: Matrix of eigenvectors.
        :return: Sorted eigenvalues, sorted eigenvectors, idxeigvals2idxordereigvals, idxordereigvals2idxeigvals
        """
        # Sort Shifted
        self.SortedEigenvaluesShifted,self.SortedEigenvectorsShifted,self.IdxEigvals2IdxOrderEigvalsShifted,self.IdxOrderEigvals2IdxEigvalsShifted = SortEigvalsEigVec(self.EigenvaluesShifted,self.EigenvectorsShifted)
        # Sort Shifted and Scaled
        self.SortedEigenvaluesShiftedScaled,self.SortedEigenvectorsShiftedScaled,self.IdxEigvals2IdxOrderEigvalsShiftedScaled,self.IdxOrderEigvals2IdxEigvalsShiftedScaled = SortEigvalsEigVec(self.EigenvaluesShiftedScaled,self.EigenvectorsShiftedScaled)

    def SplitRealAndImaginary(self):
        """
            @describe:
                Since eigenvectors and eigenvalues are computed and cast directly into real and imaginary parts,
                we need to differentiate among them if we want to do the dyadic expansion using np.outer.
                Otherwise we have error.
                NOTE: If the matrices in inpute are symmetric: ALWAYS for the case study,
                we have that both eigenvectors and eigenvalues have imaginary part that can be discarded.
                We keep it, as we want to construct some more general package.

        """
        self.SortedEigenvaluesShiftedReal = np.real(self.SortedEigenvaluesShifted)
        self.SortedEigenvaluseShiftedScaledReal = np.real(self.SortedEigenvaluesShiftedScaled)
        self.SortedEigenvaluesShiftedImag = np.imag(self.SortedEigenvaluesShifted)
        self.SortedEigenvaluesShiftedScaledImag = np.imag(self.SortedEigenvaluesShiftedScaled)
        self.SortedEigenvectorsShiftedReal = np.real(self.SortedEigenvectorsShifted)
        self.SortedEigenvectorsShiftedScaledReal = np.real(self.SortedEigenvectorsShiftedScaled)
        self.SortedEigenvectorsShiftedImag = np.imag(self.SortedEigenvectorsShifted)
        self.SortedEigenvectorsShiftedScaledImag = np.imag(self.SortedEigenvectorsShiftedScaled)


    def ComputeDistributionLambda(self):
        """
            Compute the spectrum of the correlation matrix
        """
        self.DistributionLambdaShiftedRescaledReal = np.histogram(self.SortedEigenvaluseShiftedScaledReal,bins=100)
        self.DistributionLambdaShiftedReal = np.histogram(self.SortedEigenvaluesShiftedReal,bins=100)
        self.ObservedEigenvectorsShiftedRescaledReal = np.reshape(self.SortedEigenvectorsShiftedScaledReal,(self.SortedEigenvectorsShiftedScaledReal.shape[0]*self.SortedEigenvectorsShiftedScaledReal.shape[1]))
        self.DistributionEigenvectorEntranceShiftRescaledReal = np.histogram(self.ObservedEigenvectorsShiftedRescaledReal,bins=100)
        self.ObservedEigenvectorsShiftedReal = np.reshape(self.SortedEigenvectorsShiftedReal,(self.SortedEigenvectorsShiftedReal.shape[0]*self.SortedEigenvectorsShiftedReal.shape[1]))
        return self.DistributionLambdaShiftedRescaledReal

    def ComputeDyadicExpansion(self):
        """
            @describe: Compute the dyadic expansion of the correlation matrix
            M = sum_i lambda_i * v_i * v_i^T
            This is useful to extract the comoving frame of the correlation matrix.
        """
        # Shifted 
        self.DyadicExpansionShifted = ComputeDyadicExpansion(self.SortedEigenvaluesShiftedReal,self.SortedEigenvectorsShiftedReal)
        # Shifted and Scaled
        self.DyadicExpansionShiftedScaled = ComputeDyadicExpansion(self.SortedEigenvaluseShiftedScaledReal,self.SortedEigenvectorsShiftedScaledReal)
        


    def ComputeReducedRankCorrelationMatrix(self):
        """
            @Description:
                This snippet divides the correlation matrix into three different classes that were introduced to me by:
                Quasi-stationary states in temporal correlations for traffic systems: Cologne orbital motorway as an example
                Schreckenberg
            NOTE: 
                self.a, self.b, self.t0 are the parameters with the names in that paper..
        """
        self.a = None
        self.b = None
        self.t0 = None
        self.IndicesOrderedEigvalsNoise = []
        self.IndicesOrderedEigvalsab = []
        self.IndicesOrderedEigvalsSignal = []
        for i,Eigval in enumerate(self.SortedEigenvalues):
            if Eigval == 0:
                self.IndicesOrderedEigvalsNoise.append(i)
            if Eigval !=0:
                pass
    def SeparateSinglaFromNoise(self):
        """
            Separate the signal from the noise
        """
        self.Signal = np.dot(self.Eigenvectors,np.dot(np.diag(self.Eigenvalues),self.Eigenvectors.T))
        self.Noise = self.CorrelationMatrix - self.Signal
        return self.Signal,self.Noise

    def GreenFunction(self,z):
        """
            @describe: Compute the Green function of the correlation matrix
        """
        self.GreenShiftScaled = np.sum(1/(z - self.SortedEigenvaluesShiftedScaledReal**2))
        self.GreenShift = np.sum(1/(z - self.SortedEigenvaluesShiftedReal**2))
        return self.GreenShiftScaled,self.GreenShift
    
    def MarchenkoPasturLimits(self):
        """
            @describe: Compute the Marchenko-Pastur limits
            
        """
        self.LambdaMinShiftedScaled, self.LambdaMaxShiftedScaled = MarchenkoPasturLimits(self.N,self.T,1)
        self.LambdaMinShifted, self.LambdaMaxShifted = MarchenkoPasturLimits(self.N,self.T,1)

        return self.LambdaMinus,self.LambdaPlus

    def PlotAnalysis(self):
        PlotDistributionEigenvalues(self.SortedEigenvaluseShiftedScaledReal,
                                    "Marchenko-Pastur",
                                    self.Q, 
                                    self.LambdaMinShiftedScaled,
                                    self.LambdaMaxShiftedScaled,
                                    1,
                                    200,
                                    self.PlotDir)
        PlotDyadicExpansion(self.DyadicExpansionShiftedScaled,8,self.PlotDir)

if __name__=="__main__":
    # Example Usage
    Sources = 200
    TimeIntervals = 100
    Signal = np.random.rand(Sources,TimeIntervals)
    RMA = AnalysisCorrelationRandomMatrix(Signal)
    # NOTE: Returns Also The Correlation Matrix
    RMA.ComputeCorrelationMatrix()
    # Expansive Computations
#    RMA.ComputeEigenvalues()
    RMA.ComputeEigenvectorsAndEigenValues()
    #
    RMA.SortEigvalsEigVec()
    RMA.SplitRealAndImaginary()
    RMA.ComputeDyadicExpansion()
    RMA.MarchenkoPasturLimits()
    RMA.PlotAnalysis()
    RMA.ComputeDistributionLambda()