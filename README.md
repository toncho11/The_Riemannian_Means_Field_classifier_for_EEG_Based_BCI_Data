Here will be published an updated version of the MeansField classifier used in the article "The Riemannian Means Field Classifier for EEG-Based BCI Data" by Anton Andreev, Gregoire Cattan and Marco Congedo, Sensors journal MDPI

* MDPI Sensors: https://www.mdpi.com/1424-8220/25/7/2305
* HAL: not yet published

Abstract: A substantial amount of research has demonstrated the robustness and accuracy of the Riemannian minimum distance to mean (MDM) classifier for all kinds of EEG-based brain–computer interfaces (BCIs). This classifier is simple, fully deterministic, robust to noise, computationally efficient, and prone to transfer learning. Its training is very simple, requiring just the computation of a geometric mean of a symmetric positive-definite (SPD) matrix per class. We propose an improvement of the MDM involving a number of power means of SPD matrices instead of the sole geometric mean. By the analysis of 20 public databases, 10 for the motor-imagery BCI paradigm and 10 for the P300 BCI paradigm, comprising 587 individuals in total, we show that the proposed classifier clearly outperforms the MDM, approaching the state-of-the art in terms of performance while retaining the simplicity and the deterministic behavior. In order to promote reproducible research, our code will be released as open source.

Installation: 
* moabb
* pyRiemann

Usage:
* example.py

Authors:
* [Anton Andreev](https://scholar.google.com/citations?user=NFtzWMAAAAAJ&hl=en)
* [Gregoire Cattan](https://scholar.google.com/citations?user=SYe1u-kAAAAJ&hl=en)
* [Marco Congedo](https://scholar.google.com/citations?user=f9a1rO0AAAAJ&hl=en)
