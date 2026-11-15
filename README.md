# Domain-adaptation Machine Learning Approach for Correlation of Full-sized Charpy Properties From Sub-sized Tests
**Authors: T. Wang, J.W. Merickel, Y. Tang, F. Xu, A. Vakanski, R. Song**

Reliable assessment of nuclear structural materials requires a consistent interpretation of impact toughness across specimen sizes. However, size effects vary across specimens and change both absorbed energy and ductile-to-brittle transition behavior, complicating the prediction of full-sized properties from sub-sized Charpy tests. Sparse measurements and uneven temperature coverage further limit the development of predictive correlations. 

This study develops a three-step framework integrating uncertainty-aware dual sampling (UADS), sub-to-full label projection, and a domain-specific AdaBN (DS-AdaBN) regression network. UADS combines Gaussian-Local Sampling and Envelope Sampling to enhance temperature coverage and represent local variability through temperature-dependent sampling bands. A specimen-specific projection converts the sampled responses into full-sized-equivalent training labels, while DS-AdaBN enables learning from sampled and measured data with different distributions. 

Evaluation across A533B transition curves yields final R² values of 0.9569 for upper-shelf energy (USE) and 0.9628 for ductile-to-brittle transition temperature (DBTT), with corresponding root-mean-square errors of 18.84 J and 12.02 °C. 

The framework provides a basis for using sub-sized impact tests in full-sized property estimation and for extending size correlation based on domain adaptation machine learning to additional materials and specimen geometries.

## 📂 Repository Organization
- [**Charpy Dataset**](Charpy_Dataset/): Contains the Charpy impact test dataset and the data files used in this study.
- [**Domain Adaptation**](Domain_Adaptation/): Contains the code and outputs for the main results of the study, including the UADS sampling, sub-to-full label projection, DS-AdaBN regression, and physics-guided and uncertainty-adaptive correction for USE and DBTT prediction.
- [**Tables and Figures**](Tables_and_Figures/): Contains notebooks and exported files for the comparison with other domain-adaptation methods and for generating the additional tables and figures reported in the study.
- [**Ablation Scripts**](Ablation_Scripts/): Contains scripts for the ablation studies evaluating the contribution of the sampling strategies, real-domain augmentation, and the number of synthetic points per curve.

## 📊 Data
- [**Dataset**](Charpy_Dataset/Charpy_Impact_Test_Dataset.xlsx): The dataset of Charpy impact properties was collected by our team and is described in [Comprehensive Toughness Dataset of Nuclear Reactor Structural Materials using Charpy V-Notch Impact Testing](https://www.nature.com/articles/s41597-025-04823-1) and archived in the [Materials Cloud Archive](https://doi.org/10.24435/materialscloud:p6-5y). The full-sized curves from this dataset serve as references for label projection and as ground truth for evaluation.
- [**Preprocessed data**](Charpy_Dataset/Preprocessed_Data.xlsx): The sub-sized Charpy test data from References 52, 53, and 59 with encoded material, specimen, and test features, used as the model input.
- [**Transformed data**](Charpy_Dataset/Transformed_Data.xlsx): The precomputed full-sized-equivalent temperature and absorbed energy labels of the measured sub-sized data, used as the training labels.

## 🔨 Requirements
- Python 3.12
- pandas  
- numpy  
- scipy  
- scikit-learn  
- torch  
- matplotlib
- openpyxl
- jupyter

## ▶️ Use
Download the `Charpy_Dataset` folder and set `ROOT` at the top of each script or notebook to your own path of this folder. Run the notebook in `Domain Adaptation` to reproduce the main results, and run each Python script in `Ablation Scripts` (e.g., `python ablation_gl_only.py`) to reproduce the ablation studies. Then run the notebooks in `Tables and Figures` to generate the additional tables and figures, starting with `DSAdaBN_Different_Methods.ipynb` and followed by `DSAdaBN_Sampling_Figure_Table.ipynb`.

## 🚩 License
<a href="LICENSE">MIT License</a>

## 👏 Acknowledgments
This work was supported through the INL Laboratory Directed Research & Development (LDRD) Program under DOE Idaho Operations Office Contract DE-AC07-05ID14517 (project tracking number 24A1081-149). Accordingly, the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this manuscript or allow others to do so, for U.S. Government purposes.

## ✉️ Contact or Questions
A. Vakanski, e-mail: vakanski@uidaho.edu  
R. Song, e-mail: Rongjie.song@inl.gov
