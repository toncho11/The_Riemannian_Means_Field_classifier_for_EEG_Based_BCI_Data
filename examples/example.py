"""

This is an example on how to use the code for the article:
    
"The Riemannian Means Field Classifier for EEG-Based BCI Data"

Available at:
    Sensors: https://www.mdpi.com/1424-8220/25/7/2305
    HAL:     https://hal.science/hal-05043032

"""

from pyriemann.estimation import XdawnCovariances, Covariances
from sklearn.pipeline import make_pipeline
#from mean_field_acc import MFACC
#from mean_field_acc_utils import ADCSP
from mfacc import MFACC
from mfacc import ADCSP
from pyriemann.classification import TangentSpace
from sklearn.linear_model import LogisticRegression

from moabb.datasets import (
    BNCI2014_001,
    Zhou2016,
    BNCI2015_001,
    BNCI2014_002,
    BNCI2014_004,
    AlexMI,
    Weibo2014,
    Cho2017,
    GrosseWentrup2009,
    PhysionetMI,
    Shin2017A,
    Lee2019_MI,
    Schirrmeister2017
)
from moabb.evaluations import (
    WithinSessionEvaluation,
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
)
from moabb.paradigms import P300, MotorImagery, LeftRightImagery

pipelines = {}

pipelines["ADCSP+MF"] = make_pipeline(
    Covariances("oas"),
    ADCSP(mode="high_electrodes_count"),
    ADCSP(mode="low_electrodes_count"),
    MFACC(
           method_label="lda",
           n_jobs=1,
           rpme_enabled = False,
         ),   
    )

# motor imagery datasets
datasets = [Cho2017(), Shin2017A(), Schirrmeister2017()]

subj = [1, 2, 3] #select first 3 subjects
for d in datasets:
    d.subject_list = subj
    
paradigm = LeftRightImagery()

evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets, overwrite=True
)

results = evaluation.process(pipelines)

print(results.groupby("pipeline").mean("score")[["score", "time"]])