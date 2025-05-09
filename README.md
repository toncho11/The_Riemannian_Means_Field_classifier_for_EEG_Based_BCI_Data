This Git repository contains the code for the article "The Riemannian Means Field Classifier for EEG-Based BCI Data" by Anton Andreev, Gregoire Cattan and Marco Congedo, Sensors journal MDPI. The code has been refactored, streamlined and documented in order to be used by other researchers or Machine Learning practitioners in the domain. While it targets mainly EEG data, it has been also tested on non EEG data as well.

* MDPI Sensors: https://www.mdpi.com/1424-8220/25/7/2305
* HAL: https://hal.science/hal-05043032

If you use our algorithm, please cite this publication: https://www.mdpi.com/1424-8220/25/7/2305

Abstract: A substantial amount of research has demonstrated the robustness and accuracy of the Riemannian minimum distance to mean (MDM) classifier for all kinds of EEG-based brain–computer interfaces (BCIs). This classifier is simple, fully deterministic, robust to noise, computationally efficient, and prone to transfer learning. Its training is very simple, requiring just the computation of a geometric mean of a symmetric positive-definite (SPD) matrix per class. We propose an improvement of the MDM involving a number of power means of SPD matrices instead of the sole geometric mean. By the analysis of 20 public databases, 10 for the motor-imagery BCI paradigm and 10 for the P300 BCI paradigm, comprising 587 individuals in total, we show that the proposed classifier clearly outperforms the MDM, approaching the state-of-the art in terms of performance while retaining the simplicity and the deterministic behavior. In order to promote reproducible research, our code will be released as open source.

Installation: 
* `pip install moabb` (version 1.2 or above)
* `pip install pyriemann` (use current pyRiemann from Git 0.9.dev0, 0.9 when released, or a version after 0.9)
* Place `mean_field_acc` and `mean_field_acc_utils` in the same folder as your script. 

Install pyRiemann after MOABB. MOABB can override the version of pyRiemann you want to use.

The [example.py](https://github.com/toncho11/The_Riemannian_Means_Field_classifier_for_EEG_Based_BCI_Data/blob/main/example.py) script contains the pipeline we recommend.

```
from mean_field_acc import MFACC
from mean_field_acc_utils import ADCSP

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
```
Please check the documentation for each parameter of MFACC for more information. ADCSP is an Adaptive Double CSP procedure we apply. The Robust Power Mean Estimation (RPME) is an outliers removal procedure that can be applied while calculating the power means. Check the `rpme` parameters on how to configure. It can help better estimate the power means at the cost of speed. It also depends on your data. There is also an example using the euclidean mean that shows how to add non-power means.  

Authors:
* [Anton Andreev](https://scholar.google.com/citations?user=NFtzWMAAAAAJ&hl=en)
* [Gregoire Cattan](https://scholar.google.com/citations?user=SYe1u-kAAAAJ&hl=en)
* [Marco Congedo](https://scholar.google.com/citations?user=f9a1rO0AAAAJ&hl=en)
