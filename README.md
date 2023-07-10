![Python](https://img.shields.io/badge/python-v3.11-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Status](https://img.shields.io/badge/status-up-brightgreen)

<h4 align="center">Feature Extraction Techniques based on Deep Learning Framework
for Enhanced Classification of Non-coding RNA</h4>

<p align="center">
  <a href="https://github.com/brenoslivio/DL-RNAFeatExtraction">Home</a> •
  <a href="#installing-dependencies-and-package">Installing</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> 
</p>

## Abstract

The accurate identification and classification of non-coding RNA (ncRNA) are crucial for unravelling their functions and regulatory mechanisms in various biological processes, where understanding ncRNA roles can shed light on complex microbial interactions and their impact on ecosystems. Traditional machine-learning approaches have been employed for distinguishing ncRNA. However, these methods often necessitate extensive feature engineering and may be constrained by the accuracy of the selected features. Recently, deep learning techniques have demonstrated considerable potential in improving classification performance for ncRNA.
This study introduces a robust hybrid deep learning framework that integrates Convolutional Neural Networks (CNN) or Bidirectional Long Short-Term Memory (BiLSTM) networks with external features to enhance classification accuracy. The framework employs a combination of $k$-mer one-hot, $k$-mer dictionary, and advanced feature extraction techniques to represent the input sequences. The extracted features are subsequently incorporated into the deep learning architecture, enabling the model to exploit the spatial and sequential information inherent in the ncRNA sequences. The framework further benefits from advanced training strategies, such as dropout and batch normalization, to mitigate overfitting and improve generalization.
To evaluate our proposed framework's performance, we used benchmark datasets and real-world laboratory RNA samples extracted with the Infernal tool. The results demonstrated a high level of accuracy in classifying ncRNA. This indicates the effectiveness and robustness of our hybrid deep learning framework in handling complex ncRNA sequence data.
Our proposed framework paves the way for further investigation in the field of ncRNA classification and may contribute to a deeper understanding of ncRNA's biological roles and functions. The successful integration of CNN or BiLSTM with external features offers a promising avenue for future research in developing advanced models for ncRNA classification, ultimately enhancing our knowledge of the diverse roles that ncRNAs play in cellular processes and disease states.

## Authors

* Anderson Paulo Avila Santos, Breno Lívio Silva de Almeida, Robson Parmezan Bonidia, Peter F. Stadler, Ulisses Nunes da Rocha, Danilo Sipoli Sanches, and André Carlos Ponce de Leon Ferreira de Carvalho.

* **Correspondence:** anderson.avila@usp.br

## Publication

Abstract accepted and presented at the 16th symposium on Genetics and Bacterial Ecology 2023 (BAGECO 2023), with the title:  **Feature Extraction Techniques based on Deep Learning Framework for Enhanced Classification of Non-coding RNA**.

Full paper: In submission.

## Installing dependencies and package

## Conda - Terminal

Installing the application using miniconda, e.g.:

```sh
$ git clone https://github.com/brenoslivio/DL-RNAFeatExtraction.git DL-RNAFeatExtraction

$ cd DL-RNAFeatExtraction

$ git submodule init

$ git submodule update
```

**1 - Install Miniconda:** 


See documentation: https://docs.conda.io/en/latest/miniconda.html

```sh
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

$ chmod +x Miniconda3-latest-Linux-x86_64.sh

$ ./Miniconda3-latest-Linux-x86_64.sh

$ export PATH=~/miniconda3/bin:$PATH
```

**2 - Create environment:**

```sh
conda env create -f environment.yml -n dl-featextraction
```

**3 - Activate environment:**

```sh
conda activate dl-featextraction
```

**4 - You can deactivate the environment, using:**

```sh
conda deactivate
```

## How to use



## Citation

If you use this code in a scientific publication, we would appreciate citations to the following conference abstract:

 Anderson P Avila Santos, Breno L S de Almeida, Robson P Bonidia, Peter F Stadler, Ulisses N da Rocha, Danilo S Sanches, André C P L F de Carvalho, Feature Extraction Techniques based on Deep Learning Framework for Enhanced Classification of Non-coding RNA. In: 16th symposium on Genetics and Bacterial Ecology 2023 (BAGECO 2023), 2023, Copenhagen. New approaches/technologies in microbial ecology, 2023.

```sh
@conference{bageco2023,
    title        = "{Feature Extraction Techniques based on Deep Learning Framework for Enhanced Classification of Non-coding RNA}",
    author       = {Santos, Anderson P Avila and de Almeida, Breno L S and Bonidia, Robson P  and Stadler, Peter F and da Rocha, Ulisses N and Sanches, Danilo S and de Carvalho, André C P L F},
    year         = 2023,
    month        = {06},
    booktitle    = {New approaches/technologies in microbial ecology},
    address      = {Copenhagen, Denmark},
    pages        = {235},
    organization = {CAP Partner},
    url = {https://bageco2023.org/wp-content/uploads/2023/06/CAP-Partner_Bageco2023_programme_A5_abstract-korr10.pdf}
}
```
