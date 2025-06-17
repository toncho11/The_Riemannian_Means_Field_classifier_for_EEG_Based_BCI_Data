"""
Contains helper code for the article:

"The Riemannian Means Field Classifier for EEG-Based BCI Data"

Available at:
    Sensors: https://www.mdpi.com/1424-8220/25/7/2305
    HAL:     https://hal.science/hal-05043032

It contains a Scikit Learn transformer called ADCSP (Adaptive Double CSP).
"""
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.spatialfilters import CSP

class ADCSP(BaseEstimator, TransformerMixin):
    """    
    Implements an Adaptive Double CSP (Common Spatial Filter) transformer for
    MOABB. The first stage targets fast dimensionality reduction and the second, 
    if needed, finds the optimal spatial filter to enhance class separability.
    ADCSP and CSP are usually used with motor imagery datasets.
    
    Mode selects two stages:
    Stage 1: nfilter_high_electrodes_count = 28
    Stage 2: nfilter_low_electrodes_count  = 10
    
    Stage 1 is entered if the dimension of the covariance matrices is ≥ 28 and
    reduces the dimension to 28. The second stage, always executed after the
    first, is entered if the dimension ≥ 10 and reduces the dimension to 10.
    The two-stage procedure of the ADCSP guarantees a maximum dimension of 10 
    for the covariance matrices to be submitted for classification. Stage 1
    should be executed before Stage 2. If number of lectrodes is lower than 10,
    then no CSP is applied.
    
    Example how to use in pipeline:
        ADCSP(mode="high_electrodes_count"),
        ADCSP(mode="low_electrodes_count"),
        
    """
    
    def __init__(self, mode):
        
        self.mode    = mode
        self.nfilter_low_electrodes_count  = 10 # target covariance matrix size
        self.nfilter_high_electrodes_count = 28 # target covariance matrix size
    
    def fit(self, X, y=None):
          
        self.n_electrodes = X.shape[1]
        
        if self.n_electrodes <= self.nfilter_low_electrodes_count:
            return self
                           
        elif self.mode == "high_electrodes_count":
            if self.n_electrodes > self.nfilter_high_electrodes_count:
                self.csp = CSP(nfilter = self.nfilter_high_electrodes_count, metric="euclid", log=False)
            else:
                return self
            
        elif self.mode == "low_electrodes_count":
            
            if self.n_electrodes > self.nfilter_high_electrodes_count:
                raise Exception("Number of electrodes is too high. CSP will be slow. Use 'high_electrodes_count' mode instead.")
            else: # <28 electrodes 
                self.csp = CSP(nfilter = self.nfilter_low_electrodes_count, metric = "riemann", log=False)
        else:
            raise Exception("Invalid ADCSP mode")
             
        self.csp.fit(X,y)
        
        return self
    
    def transform(self, X):
        
        if self.n_electrodes <= self.nfilter_low_electrodes_count:
            return X
        
        if self.mode == "high_electrodes_count":
            
            if self.n_electrodes > self.nfilter_high_electrodes_count:
                return self.csp.transform(X)
            else: 
                return X
            
        elif self.mode == "low_electrodes_count":
            
            if self.n_electrodes > self.nfilter_high_electrodes_count:
                raise Exception("Number of electrodes is too high. CSP will be slow. Use 'high_electrodes_count' mode instead.")
            else:
                return self.csp.transform(X)
        else:
             raise Exception("Invalid ADCSP mode")