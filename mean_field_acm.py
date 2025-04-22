import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import SVC as sklearnSVC
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
import warnings

from pyriemann.utils.kernel import kernel
from pyriemann.utils.mean import mean_covariance, mean_power, mean_logeuclid
from pyriemann.utils.distance import distance
from pyriemann.tangentspace import FGDA, TangentSpace
from pyriemann.utils.distance import distance_euclid
from scipy.stats import zscore
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import copy
from enchanced_mdm_mf_tools import mean_power_custom, distance_custom, power_distance, vector_distance
from time import perf_counter_ns,perf_counter
from pyriemann.clustering import Potato

class MeanField_ACM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Riemannian Means Field Classifier
    
    The Riemannian Means Field Classifier (MF) [1]_ is the second version,
    which improves the performance of the Minimum Distance to Mean Field (MDMF) [2]_
    classifier. Classification is performed by defining several power means for
    each class.

    Parameters
    ----------
    power_list : list of float, default=[-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1]
        Exponents of power means.
    method_label : {'lda', 'sum_means', 'inf_means'}, default='lda'
        Method to combine labels:
        * lda: an LDA classfier is trained on the distances to each mean for
          all classes which allows more complex patterns to be learned
        * sum_means: it assigns the covariance to the class whom the sum of
          distances to means of the field is the lowest;
        * inf_means: it assigns the covariance to the class of the closest mean
          of the field.
    metric : string, default="riemann"
        Metric used for distance estimation during prediction.
        For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`.
    power_mean_zeta : float, default=1e-07
        A stopping criterion for the mean calculation. Bigger values help with speed,
        smaller provide better accuracy.
    distance_squared : bool, default = True
        The distances used for classification are squared.  
    distance_strategy : {'default_metric', 'power_distance'}, default='power_distance'
        * default_metric: it uses the metric parameter for the means and distances
          calculations
        * power_mean: the inverse of each power mean is calculated once and then used 
          to calculate all distance to it
    reuse_previous_mean :bool, default = False
        An optimization that allows the calculation of a power mean to be initialized using
        a previously calculated mean from the power_list.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class mean fields and distances in parallel. Depending on 
        the data, the computation of the means and distances can be intensive and 
        in this case the parallel processing can help.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : dict of ``n_powers`` lists of ``n_classes`` ndarrays of shape \
            (n_channels, n_channels)
        Centroids for each power and each class.

    See Also
    --------
    MDM

    Notes
    -----
    .. versionadded:: 0.9

    References
    ----------
    .. [1] `The Riemannian Means Field Classifier for EEG-Based BCI Data'
        <https://www.mdpi.com/1424-8220/25/7/2305>`
        A Anndreev, G Cattan, M Congedo. MDPI Sensors journal, April 2025
    .. [2] `The Riemannian Minimum Distance to Means Field Classifier
        <https://hal.archives-ouvertes.fr/hal-02315131>`_
        M Congedo, PLC Rodrigues, C Jutten. BCI 2019 - 8th International
        Brain-Computer Interface Conference, Sep 2019, Graz, Austria.
    """

    def __init__(self, power_list=[-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1], 
                 method_label='lda',
                 metric="riemann",
                 power_mean_zeta=1e-07,
                 distance_squared=True,
                 distance_strategy = "power_distance",
                 reuse_previous_mean = False,
                 n_jobs=1,
                 
                 euclidean_mean=False, #if True sets LogEuclidian distance for LogEuclidian mean and Euclidian distance for power mean p=1
                 #RPME parameters
                 remove_outliers=True,
                 outliers_th=2.5,
                 outliers_depth=4, #how many times to run the outliers detection on the same data
                 outliers_max_remove_th=30, #default 30%, parameter is percentage
                 outliers_method="zscore",
                 outliers_mean_init=True,
                 outliers_single_zscore=True, #when false more outliers are removed. When True only the outliers further from the mean are removed
                 ):
        """Init."""
        self.power_list = power_list
        self.method_label = method_label
        self.metric = metric
        self.power_mean_zeta = power_mean_zeta
        self.distance_strategy = distance_strategy
        self.reuse_previous_mean = reuse_previous_mean
        self.n_jobs = n_jobs

        self.euclidean_mean = euclidean_mean 
        #RPME
        self.remove_outliers = remove_outliers
        self.outliers_th = outliers_th
        self.outliers_depth = outliers_depth
        self.outliers_max_remove_th = outliers_max_remove_th
        self.outliers_method = outliers_method
        self.outliers_mean_init = outliers_mean_init
        self.distance_squared = distance_squared
        self.outliers_single_zscore = outliers_single_zscore
        
        if distance_strategy not in ["default_metric", "power_distance"]:
            raise Exception()("Invalid distance stategy!")
        
        if (outliers_max_remove_th > 100):
            raise Exception("outliers_max_remove_th is a %, it can not be > 100")
            
        if self.method_label == "lda":
            self.lda = LDA()
    
    def _calculate_mean(self,X, y, p, sample_weight):
        '''
        Calculates mean (and inv. mean) for all classes for specific p

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        sample_weight : TYPE
            DESCRIPTION.

        Returns
        -------
        means_p : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        '''
        means_p   = {} #keys are classes, values are means for this p and class
        inv_means = {} #same as above
        
        if p == 200: #adding an extra mean - this one is logeuclid and not power mean
            #print("euclidean mean")
            for ll in self.classes_:
                means_p[ll] = mean_logeuclid(
                    X[y == ll],
                    sample_weight=sample_weight[y == ll]
                )       
        else:
            for ll in self.classes_:
                
                init = None
                
                #use previous mean for this p
                #usually when calculating the new mean after outliers removal
                if self.outliers_mean_init and p in self.covmeans_:
                    init = self.covmeans_[p][ll] #use previous mean
                    #print("using init mean")
                
                elif self.reuse_previous_mean:
                    pos = self.power_list.index(p)
                    if pos>0:
                        prev_p = self.power_list[pos-1]
                        init = self.covmeans_[prev_p][ll]
                        #print(prev_p)
                        #print("using prev mean from the power list")
                 
                means_p[ll] = mean_power( #original is mean_power_custom
                    X[y == ll],
                    p,
                    sample_weight=sample_weight[y == ll],
                    zeta = self.power_mean_zeta,
                    init = init
                )
            
        if self.distance_strategy == "power_distance":
            inv_means= self.calculate_inv_mean_by_mean(means_p)
            
        return means_p,inv_means #contains means for all classes
    
    def _calcualte_mean_remove_outliers(self,X, y, p, sample_weight):
        '''
        Removes outliers and calculates the power mean p on the rest

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        sample_weight : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        means_p : TYPE
            DESCRIPTION.
        inv_means : TYPE
            DESCRIPTION.

        '''
        X_no_outliers = X.copy() #so that every power mean p start from the same data
        y_no_outliers = y.copy()
        
        count_total_outliers_removed_per_class = np.zeros(len(self.classes_))
        count_total_samples_per_class          = np.zeros(len(self.classes_))
        
        for ll in self.classes_:
            count_total_samples_per_class[ll] = len(y_no_outliers[y_no_outliers==ll])
        
        if self.outliers_method == "iforest":
            iso = IsolationForest(contamination='auto') #0.1
        elif self.outliers_method == "lof":
            lof = LocalOutlierFactor(contamination='auto', n_neighbors=2) #default = 2
        
        early_stop = False
        
        for i in range(self.outliers_depth):
            
            if early_stop:
                #print("Early stop")
                break
            
            #print("\nremove outliers iteration: ",i)
            
            #calculate/update the n means (one for each class)
            means_p,inv_means = self._calculate_mean(X_no_outliers, y_no_outliers, p, sample_weight)
            
            ouliers_per_iteration_count = {}
            
            #outlier removal is per class
            for ll in self.classes_:
                
                samples_before = X_no_outliers.shape[0]
                
                m = [] #each entry contains a distance to the power mean p for class ll
                
                #length includes all classes, not only the ll
                z_scores = np.zeros(len(y_no_outliers),dtype=float)
            
                # Calcualte all the distances only for class ll and power mean p
                for idx, x in enumerate (X_no_outliers[y_no_outliers==ll]):
                    
                    if self.distance_strategy == "power_distance":
                        #dist_p = self._calculate_distance(x, self.covmeans_inv_[p][ll], p)
                        dist_p = self._calculate_distance(x, inv_means[ll], p)
                    else:
                        #dist_p = self._calculate_distance(x, self.covmeans_[p][ll], p)
                        dist_p = self._calculate_distance(x, means_p[ll], p)
                    #dist_p = np.log(dist_p)
                    m.append(dist_p)
                
                m = np.array(m, dtype=float)
                
                if self.outliers_method == "zscore":
                    
                    m = np.log(m)
                    # Calculate Z-scores for each data point for the current ll class
                    # For the non ll the zscore stays 0, so they won't be removed
                    z_scores[y_no_outliers==ll] = zscore(m)
                
                    if self.outliers_single_zscore:
                        outliers = (z_scores > self.outliers_th)
                    else:
                        outliers = (z_scores > self.outliers_th) | (z_scores < -self.outliers_th)
                    
                elif self.outliers_method == "iforest":
                    
                    m1 = [[k] for k in m]
                    z_scores[y_no_outliers==ll] = iso.fit_predict(m1)
                    #outliers is designed to be the size with all classes
                    outliers = z_scores == -1
                    
                elif self.outliers_method == "lof":
                    
                    m1 = [[k] for k in m]
                    z_scores[y_no_outliers==ll] = lof.fit_predict(m1)
                    #outliers is designed to be the size with all classes
                    outliers = z_scores == -1
                    
                else:   
                    raise Exception("Invalid Outlier Removal Method")

                outliers_count = len(outliers[outliers==True])
                
                #check if too many samples are about to be removed
                #case 1 less than self.max_outliers_remove_th are to be removed
                if ((count_total_outliers_removed_per_class[ll] + outliers_count) / count_total_samples_per_class[ll]) * 100 < self.outliers_max_remove_th:
                    #print ("Removed for class ", ll ," ",  len(outliers[outliers==True]), " samples out of ", X_no_outliers.shape[0])
            
                    X_no_outliers = X_no_outliers[~outliers]
                    y_no_outliers = y_no_outliers[~outliers]
                    sample_weight = sample_weight[~outliers]
                
                    if X_no_outliers.shape[0] != (samples_before - outliers_count):
                        raise Exception("Error while removing outliers!")
                    
                    count_total_outliers_removed_per_class[ll] = count_total_outliers_removed_per_class[ll] + outliers_count
                
                else: #case 2 more than self.max_outliers_remove_th are to be removed
                
                    outliers_count = 0 #0 set outliers removed to 0
                    
                    print("WARNING: Skipped full outliers removal because too many samples were about to be removed.")
                
                ouliers_per_iteration_count[ll] = outliers_count
            
            #early stop: if no outliers were removed for both classes then we stop early
            if sum(ouliers_per_iteration_count.values()) == 0:
                early_stop = True
        
        count_total_outliers_removed = count_total_outliers_removed_per_class.sum()

        if count_total_outliers_removed > 0:
           
            #generate the final power mean (after outliers removal)
            means_p,inv_means = self._calculate_mean(X_no_outliers, y_no_outliers, p, sample_weight)
        
            count_outliers_removed_for_single_mean_gt = X.shape[0] - X_no_outliers.shape[0]
            
            if (count_total_outliers_removed != count_outliers_removed_for_single_mean_gt):
                raise Exception("Error outliers removal count!")
            
            #print("Total outliers removed for mean p=",p," is: ",total_outliers_removed, " for all classes")
            
            if (count_outliers_removed_for_single_mean_gt / X.shape[0]) * 100 > self.outliers_max_remove_th:
                raise Exception("Outliers removal algorithm has removed too many samples: ", count_outliers_removed_for_single_mean_gt, " out of ",X.shape[0])
        else: 
            #print("No outliers removed")
            pass
        
        return means_p,inv_means
    
    def _inv_power_distance(self, trial, power_mean_inv, squared=False):
        '''
        A distance that requires inv power mean as second parameter

        Parameters
        ----------
        trial : TYPE
            DESCRIPTION.
        power_mean_inv : TYPE
            DESCRIPTION.
        squared : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        _check_inputs(power_mean_inv, trial)
        d2 = (np.log( np.linalg.eigvals (power_mean_inv @ trial)) **2 ).sum(axis=-1)
        return d2 if squared else np.sqrt(d2)

    def _calculate_all_means(self,X,y,sample_weight):
        
        if self.n_jobs==-1 or self.n_jobs > 1:
            print("parallel means")
            if (self.remove_outliers):
                
                results = Parallel(n_jobs=self.n_jobs)(delayed(self._calcualte_mean_remove_outliers)(X, y, p, sample_weight)
                                      for p in self.power_list
                                  )
            else:
                results = Parallel(n_jobs=-1)(delayed(self._calculate_mean)(X, y, p, sample_weight)
                                        for p in self.power_list
                                    )
        else:
            print("NON parallel means")
            results = [] #per p for all classes
            for p in self.power_list:
                
                if (self.remove_outliers):
                    result_per_p = self._calcualte_mean_remove_outliers(X, y, p, sample_weight)
                else:
                    result_per_p = self._calculate_mean(X, y, p, sample_weight)
                results.append(result_per_p)
        
        for i, p in enumerate(self.power_list):
            self.covmeans_[p]     = results[i][0]
            self.covmeans_inv_[p] = results[i][1]
                
    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids. Calculates the power means.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : MeanField instance
            The MeanField instance.
        """
        
        # example on how to add non power means
        # p>= 200 are handled separately from the power means [-1 .. 1]
        if self.euclidean_mean:
            self.power_list.append(200)
            
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        
        # keys are p, each value is another dictionary over the classes and
        # values of this dictionary are the means for this p and a class
        self.covmeans_ = {}
        self.covmeans_inv_ = {}
        
        self._calculate_all_means(X,y,sample_weight)
        
        if len(self.power_list) != len(self.covmeans_.keys()):
            raise Exception("Problem with number of calculated means!",len(self.power_list),len(self.covmeans_.keys()))
            
        if self.distance_strategy == "power_distance" and len(self.covmeans_.keys()) != len(self.covmeans_inv_.keys()):
            raise Exception("Problem with the number of inverse matrices")
        
        if self.method_label == "lda":
            dists = self._predict_distances(X)
            self.lda.fit(dists,y)

        return self
    
    def calculate_inv_mean_by_mean(self,cov): #for all classes
        '''
        Calculates the inverse mean of a mean covariance matrix.

        Parameters
        ----------
        cov : TYPE
            DESCRIPTION.

        Returns
        -------
        inv_means_p : TYPE
            DESCRIPTION.

        '''
        inv_means_p = {}
        for ll in self.classes_:
            inv_means_p[ll] = np.linalg.inv(cov[ll])
        
        return inv_means_p           

    def _get_label(self, x, labs_unique):
        
        m = np.zeros((len(self.power_list), len(labs_unique)))
        
        for ip, p in enumerate(self.power_list):
            for ill, ll in enumerate(labs_unique):
                 m[ip, ill] = self._calculate_distance(x,self.covmeans_[p][ll],p)

        if self.method_label == 'sum_means':
            ipmin = np.argmin(np.sum(m, axis=1))
        elif self.method_label == 'inf_means':
            ipmin = np.where(m == np.min(m))[0][0]
        else:
            raise TypeError('method_label must be sum_means or inf_means')

        y = labs_unique[np.argmin(m[ipmin])]
        return y

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest means field.
        """
        
        #print("In predict")
        if self.method_label == "lda":

            dists = self._predict_distances(X)
            
            pred  = self.lda.predict(dists)
            
            return np.array(pred)
            
        else:
            
            labs_unique = sorted(self.covmeans_[self.power_list[0]].keys())
    
            pred = Parallel(n_jobs=self.n_jobs)(delayed(self._get_label)(x, labs_unique)
                 for x in X
                )
            
            return np.array(pred)

    def _calculate_distance(self,A,B,p):

        squared = self.distance_squared
        
        if len(A.shape) == 2:
        
            if self.distance_strategy == "default_metric":
                
                dist = distance(
                        A,
                        B,
                        metric=self.metric,
                        squared = squared,
                    )
            
            # similar to "default_metric", but uses inverse mean
            elif self.distance_strategy == "power_distance":
                
                dist = self._inv_power_distance(
                        A, #trial
                        B, #mean inverted
                        squared = squared,
                    )
            else:
                raise Exception("Invalid distance strategy")
                    
        else:
            raise Exception("Error size of input, not matrices?")
            
        return dist
    
    def _calucalte_distances_for_all_means(self,x):
        '''
        Calculates the distances to all power means 

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        combined : TYPE
            DESCRIPTION.

        '''
        m = {} #contains a distance to a power mean
        
        for p in self.power_list:
            m[p] = []
            
            for ll in self.classes_: #add all distances (1 per class) for m[p] power mean
                
                if self.distance_strategy == "power_distance":
                    dist_p = self._calculate_distance(x, self.covmeans_inv_[p][ll], p)
                else:
                    dist_p = self._calculate_distance(x, self.covmeans_[p][ll], p)
                
                m[p].append(dist_p)
                
        combined = [] #combined for all classes
        for v in m.values():
            combined.extend(v)
        
        #check combned = (number of classes) x (number of power means)
        if len(combined) != (len(self.power_list) * len(self.classes_)) :
            raise Exception("Not enough calculated distances!", len(combined),(len(self.power_list) * 2))
            
        return combined
        
    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""
        
        #print("predict distances")
           
        if (self.n_jobs == 1):
            distances = []
            for x in X:
                distances_per_mean = self._calucalte_distances_for_all_means(x)
                distances.append(distances_per_mean)
        else:
            distances = Parallel(n_jobs=self.n_jobs)(delayed(self._calucalte_distances_for_all_means)(x)
                 for x in X
                )
            
        distances = np.array(distances)
        
        return distances

    def transform(self, X,):
        """Get the distance to each means field.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each means field according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax of negative squared distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        if self.method_label == "lda":
            
            dists = self._predict_distances(X)
            
            return self.lda.predict_proba(dists)
            
        else:
            return softmax(-self._predict_distances(X) ** 2)