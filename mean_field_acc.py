"""
Contains the python class for the article:
    "The Riemannian Means Field Classifier for EEG-Based BCI Data"

Available at:
    Sensors: https://www.mdpi.com/1424-8220/25/7/2305
    HAL:     (in preparation)

The algorithm is abbreviated "MF" in the article.
The Python class is called "MFACC".
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from joblib import Parallel, delayed
from pyriemann.utils.mean import mean_power, mean_logeuclid
from scipy.stats import zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.ensemble import IsolationForest
#from sklearn.neighbors import LocalOutlierFactor
from pyriemann.utils.distance import distance

class MFACC(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    This class is an iplementaion of classification by 
    Riemannian Means Field Classifier (MF) [1]_

    The MF is the second version, which improves the performance of the Minimum
    Distance to Mean Field (MDMF) [2]_ classifier. Classification is performed by
    defining a list of power means for each class. The authors of the MF have
    shown that using an LDA classifier on the distances improves the performance. 
    The MF approaches the performance of the TS+LR. The MF, provided here, also supports 
    Robust Power Mean Estimation (RPME) and shows how other non power means can also 
    be used.
    
    This class is an enhanced version incorporating our improvements to the original
    version published in the pyRiemann library.
    
    Parameters
    ----------
    power_list : list of float, default=[-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1,
                                         0.25, 0.5, 0.75, 1]
        Exponents of power means.
    method_label : {'lda', 'sum_means', 'inf_means'}, default='lda'
        Method to combine labels:
        * lda: an LDA classifier is trained on the distances to each mean for
          all classes, which allows more complex patterns to be learned
        * sum_means: it assigns the covariance to the class whom the sum of
          distances to means of the field is the lowest;
        * inf_means: it assigns the covariance to the class of the closest mean
          of the field.
    metric : string, default="riemann"
        Metric used for distance estimation during prediction.
        For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`.
    power_mean_zeta : float, default=1e-07
        A stopping criterion for the mean calculation: bigger values improve
        speed, while smaller provide better accuracy.
    power_mean_maxiter : int, default=150
        Sets the maximum mumber of iterations for the power mean calculation.
    distance_squared : bool, default = True
        The distances used for classification are squared.
    reuse_previous_mean :bool, default = False
        An optimization that allows the calculation of a power mean to be
        initialized using a previously calculated mean from the power_list.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class mean fields and distances in parallel. Depending on
        the data, the computation of the means and distances can be intensive
        and in this case the parallel processing can help.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    euclidean_mean : bool, default = False
        An example on how to add a non-power mean. All such means can be 
        encoded with numbers out of the range [-1..1]. In this case 200 has
        been selected for the euclidean_mean. These non power means are handled
        separately in the code and can even have their proper distance.
    rpme_enabled : bool, default = False
        Enables the RPME which removes outliers and improves the power mean
        generation. This is at the cost of speed. It alo depends on the data,
        the amount of noise in the data and it is more effective when there is
        sufficent data per class.
    rpme_th : int, default = 2.5
        The default valu is for a zscore threshold.
    rpme_depth : int, default = 4
        For how many iterations the RPME is applied. The data after the first
        outliers removal is used for the next outliers removal and this is
        repeated rpme_depth times.
    rpme_max_remove_th : int, default = 30
        A threshold represented by a percentage. The parameter guards that no
        more than rpme_max_remove_th % of the data is removed per class. This
        prevents removing too much data as outliers.
    rpme_method : str, default = "zscore"
        Selects which outliers detection method to be used:
            * zscore: Z score
        It is designed so that other outliers method can be added.
    rpme_mean_init : bool, default = True
        The RPME increases the number of power means that need to be
        calculated. This parameter enables an optimization that helps power
        means to be calculated faster, by using another mean as a starting 
        point.
    rpme_single_zscore: bool, default = True
         When True only the outliers further from the mean are removed. When 
         False more outliers are removed. This is because, when using the 
         z-score method, data points with z-scores greater than 2.5 or less 
         than -2.5 are considered outliers and removed.
    
    Notes:
        If you want to use MFACC as a transformer then the distances produced
        depend on the parameter method_label:
        * 'sum_means', 'inf_means' will produce 1 distance per class (no matter the number power means)
        * 'lda' will produce a distance of size (number of power means) x (number of classes), use this
          if you want to change the classifier using the Transform() method.
          
        Use method_label="lda" to produce all distances even if the class is used as
        a scikit-learn transformer and not classifier.


    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : dict of ``n_powers`` lists of ``n_classes`` ndarrays of shape \
            (n_channels, n_channels)
        Centroids for each power and each class.
    
    References
    ----------
        .. [1] `Multiclass Brain-Computer Interface Classification by Riemannian
            Geometry
            <https://hal.archives-ouvertes.fr/hal-00681328>`_
            A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. IEEE Transactions
            on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.
        .. [2] `Riemannian geometry applied to BCI classification
            <https://hal.archives-ouvertes.fr/hal-00602700/>`_
            A. Barachant, S. Bonnet, M. Congedo and C. Jutten. 9th International
            Conference Latent Variable Analysis and Signal Separation
            (LVA/ICA 2010), LNCS vol. 6365, 2010, p. 629-636.

    """

    def __init__(
            
        self,
        power_list=[-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1],
        method_label='lda',
        metric="riemann",
        power_mean_zeta=1e-07,
        power_mean_maxiter=150,
        distance_squared=True,
        reuse_previous_mean=False,
        n_jobs=1,
        euclidean_mean=False,
        #RPME parameters
        rpme_enabled=False,
        rpme_th=2.5,
        rpme_depth=4,
        rpme_max_remove_th=30,
        rpme_method="zscore",
        rpme_mean_init=True,
        rpme_single_zscore=True, 
    ):
        """Init."""
        self.power_list = power_list
        self.method_label = method_label
        self.metric = metric
        self.power_mean_zeta = power_mean_zeta
        self.power_mean_maxiter = power_mean_maxiter
        self.distance_squared = distance_squared
        self.reuse_previous_mean = reuse_previous_mean
        self.n_jobs = n_jobs
        self.euclidean_mean = euclidean_mean 
        #RPME
        self.rpme_enabled = rpme_enabled
        self.rpme_th = rpme_th
        self.rpme_depth = rpme_depth
        self.rpme_max_remove_th = rpme_max_remove_th
        self.rpme_method = rpme_method
        self.rpme_mean_init = rpme_mean_init
        self.rpme_single_zscore = rpme_single_zscore
        
        if self.reuse_previous_mean and self.n_jobs != 1:
            raise Exception(("Currently reuse_previous_mean is not supported "
                            "when combined with parallel mean calculation."))

        if method_label not in {'lda', 'sum_means', 'inf_means'}:
            raise TypeError("Parameter method_label is incorrect.")
            
        if (rpme_max_remove_th > 100):
            raise Exception("outliers_max_remove_th is a %, it can not be > 100")
            
        if self.method_label == "lda":
            self.lda = LDA()
    
    def _calculate_mean(self,X, y, p, sample_weight):
        '''
        Calculates power means for all classes for specific p.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        p : float
            Exponent of a power mean.
        sample_weight : None | ndarray shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        means_p : dict
            Contains the power means for all classes for this p.

        '''
        means_p   = {} #keys are classes, values are means for this p and class
        
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

                if self.reuse_previous_mean and self.n_jobs == 1:
                    pos = self.power_list.index(p)
                    if pos > 0:
                        prev_p = self.power_list[pos-1]
                        if prev_p in self.covmeans_:
                            init = self.covmeans_[prev_p][ll]
                        else:
                            raise Exception("No previous mean.")

                means_p[ll] = mean_power(
                    X[y == ll],
                    p,
                    sample_weight=sample_weight[y == ll],
                    zeta=self.power_mean_zeta,
                    init=init,
                    maxiter=self.power_mean_maxiter
                )
            
        return means_p
    
    def _calculate_mean_remove_outliers(self,X, y, p, sample_weight):
        '''
        Robust Power Mean Estimation (RPME)
        It is an algorithm that removes outliers during 1 or more iterations 
        and calculates the power mean p on the rest.
        '''
        X_no_outliers = X.copy() #so that every power mean p start from the same data
        y_no_outliers = y.copy()
        
        count_total_outliers_removed_per_class = np.zeros(len(self.classes_))
        count_total_samples_per_class          = np.zeros(len(self.classes_))
        
        for ll in self.classes_:
            count_total_samples_per_class[ll] = len(y_no_outliers[y_no_outliers==ll])
        
        early_stop = False
        
        for i in range(self.rpme_depth):
            
            if early_stop:
                #print("Early stop")
                break
            
            #print("\nremove outliers iteration: ",i)
            
            #calculate/update the n means (one for each class)
            means_p = self._calculate_mean(X_no_outliers, y_no_outliers, p, sample_weight)
            
            ouliers_per_iteration_count = {}
            
            #outlier removal is per class
            for ll in self.classes_:
                
                samples_before = X_no_outliers.shape[0]
                
                m = [] #each entry contains a distance to the power mean p for class ll
                
                #length includes all classes, not only the ll
                z_scores = np.zeros(len(y_no_outliers),dtype=float)
            
                # Calcualte all the distances only for class ll and power mean p
                for idx, x in enumerate (X_no_outliers[y_no_outliers==ll]):
                    dist_p = self._calculate_distance(x, means_p[ll], p)
                    m.append(dist_p)
                
                m = np.array(m, dtype=float)
                
                if self.rpme_method == "zscore":
                    
                    m = np.log(m)
                    # Calculate Z-scores for each data point for the current ll class
                    # For the non ll the zscore stays 0, so they won't be removed
                    z_scores[y_no_outliers==ll] = zscore(m)
                
                    if self.rpme_single_zscore:
                        outliers = (z_scores > self.rpme_th)
                    else:
                        outliers = (z_scores > self.rpme_th) | (z_scores < -self.rpme_th)
                
                # You can add your outliers removal strategies here.
                # Note that they will probably require different default parameters compared to z-score.
                    
                else:   
                    raise Exception("Invalid outliers removal method for RPME.")

                outliers_count = len(outliers[outliers==True])
                
                #check if too many samples are about to be removed
                #case 1 less than self.max_outliers_remove_th are to be removed
                if ((count_total_outliers_removed_per_class[ll] + outliers_count) / count_total_samples_per_class[ll]) * 100 < self.rpme_max_remove_th:
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
            means_p = self._calculate_mean(X_no_outliers, y_no_outliers, p, sample_weight)
        
            count_outliers_removed_for_single_mean_gt = X.shape[0] - X_no_outliers.shape[0]
            
            if (count_total_outliers_removed != count_outliers_removed_for_single_mean_gt):
                raise Exception("Error outliers removal count!")
            
            #print("Total outliers removed for mean p=",p," is: ",total_outliers_removed, " for all classes")
            
            if (count_outliers_removed_for_single_mean_gt / X.shape[0]) * 100 > self.rpme_max_remove_th:
                raise Exception("Outliers removal algorithm has removed too many samples: ", count_outliers_removed_for_single_mean_gt, " out of ",X.shape[0])
        else: 
            #print("No outliers removed")
            pass
        
        return means_p

    def _calculate_all_means(self,X,y,sample_weight):
        
        if (self.rpme_enabled):
            calculate_mean = self._calculate_mean_remove_outliers
        else:
            calculate_mean = self._calculate_mean

        if self.reuse_previous_mean:
            # this non-parallel version is needed when the parameter reuse_previous_mean=True
            for p in self.power_list:
                result_per_p = calculate_mean(X, y, p, sample_weight)
                self.covmeans_[p] = result_per_p
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(calculate_mean)(X, y, p, sample_weight)
                for p in self.power_list
            )

            for i, p in enumerate(self.power_list):
                self.covmeans_[p] = results[i]
                
    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids. Calculates the power means.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
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
        
        self._calculate_all_means(X,y,sample_weight)
        
        if len(self.power_list) != len(self.covmeans_.keys()):
            raise Exception("Problem with number of calculated means!",len(self.power_list),len(self.covmeans_.keys()))
        
        if self.method_label == "lda":
            dists = self._predict_distances(X)
            self.lda.fit(dists,y)

        return self          

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
        if len(A.shape) == 2:
            dist = distance(
                    A,
                    B,
                    metric=self.metric,
                    squared=self.distance_squared,
                )
        else:
            raise Exception("Error size of input, not matrices?")

        return dist
    
    def _calculate_distances_for_all_means(self,x):
        '''
        Calculates the distances to all power means and all classes.

        Parameters
        ----------
        x : ndarray, shape (n_channels, n_channels)
            An SPD matrix.

        Raises
        ------
        Exception
            If (number of classes) x (number of power means) != (total number
            of calculated distances).

        Returns
        -------
        combined : list
            A list of all distances to all power means for all classes.

        '''
        m = {}  # keys are p exponents and values are distances for each class

        # store in m all distances (1 per class) for each p
        for p in self.power_list:
            m[p] = []

            for ll in self.classes_:
                dist_p = self._calculate_distance(x, self.covmeans_[p][ll], p)
                m[p].append(dist_p)

        combined = []  # combined for all classes
        for v in m.values():
            combined.extend(v)

        if len(combined) != (len(self.power_list) * len(self.classes_)):
            raise Exception("Not enough calculated distances!", len(combined),
                            (len(self.power_list) * 2))

        return combined
        
    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""
        if self.method_label in ("sum_means", "inf_means"):
            dist = []
            for x in X:
                m = {}
                for p in self.power_list:
                    m[p] = []
                    for c in self.classes_:
                        m[p].append(
                            distance(
                                x,
                                self.covmeans_[p][c],
                                metric=self.metric,
                            )
                        )
                pmin = min(m.items(), key=lambda x: np.sum(x[1]))[0]
                dist.append(np.array(m[pmin]))
            return np.stack(dist)
        else: # lda or another classifier
            distances = Parallel(n_jobs=self.n_jobs)(
                delayed(self._calculate_distances_for_all_means)(x)
                for x in X
            )

            distances = np.array(distances)
            return distances

    def transform(self, X):
        """Get the distance to each means field.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each means field according to the metric.
        """
        return self._predict_distances(X)

    def fit_transform(self, X, y, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each means field according to the metric.
        """
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def predict_proba(self, X):
        """Predict proba using softmax of negative squared distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.

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