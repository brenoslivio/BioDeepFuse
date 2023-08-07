![Python](https://img.shields.io/badge/python-v3.11-blue)
![Status](https://img.shields.io/badge/status-up-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<h4 align="center">BioDeepFuse: A Hybrid Deep Learning Approach with Integrated Feature Extraction Techniques for Enhanced Non-coding RNA Classification</h4>

<p align="center">
  <a href="https://github.com/brenoslivio/BioDeepFuse">Home</a> •
  <a href="#installing-dependencies-and-package">Installing</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> 
</p>

## Abstract

The accurate classification of non-coding RNA (ncRNA) sequences is crucial for unravelling their functions and regulatory mechanisms in various biological processes where understanding ncRNA roles can shed light on complex microbial interactions and their impact on ecosystems. Although traditional machine learning based approaches have been employed for distinguishing ncRNA, they often need extensive feature engineering. Recently, deep learning algorithms have been successfully used to improve ncRNA classification. This study introduces a hybrid deep learning framework that integrates convolutional neural networks (CNN) or bidirectional long short-term memory (BiLSTM) networks with external features to enhance classification accuracy. The framework employs a combination of $k$-mer one-hot, $k$-mer dictionary, and feature extraction techniques to represent the input sequences. The extracted features are subsequently incorporated into the deep network architecture, enabling the exploitation of the spatial and sequential information in ncRNA sequences. We used benchmark datasets and real-world laboratory RNA samples to evaluate our proposed framework's performance. The results show high predictive accuracy in classifying ncRNA, confirming the effectiveness and robustness of our framework in handling complex ncRNA sequence data. Our framework paves the way for further investigation in ncRNA classification and may contribute to a deeper understanding of ncRNA's biological roles and functions. The successful integration of CNN or BiLSTM with external features offers a promising avenue for future research, mainly for developing advanced ncRNA classifiers and enhancing current knowledge of ncRNAs' diverse roles in cellular processes and disease states.

## Authors

* Anderson Paulo Avila Santos, Breno Lívio Silva de Almeida, Robson Parmezan Bonidia, Peter F. Stadler, Ulisses Nunes da Rocha, Danilo Sipoli Sanches, and André Carlos Ponce de Leon Ferreira de Carvalho.

* **Correspondence:** Ulisses Nunes da Rocha. Email: ulisses.rocha@ufz.de

## Publication

Abstract accepted and presented at the 16th symposium on Genetics and Bacterial Ecology 2023 (BAGECO 2023), with the title:  **Feature Extraction Techniques based on Deep Learning Framework for Enhanced Classification of Non-coding RNA**.

Full paper: In submission.

## Installing dependencies and package

## Conda - Terminal

Installing the application using miniconda, e.g.:

```sh
$ git clone https://github.com/brenoslivio/BioDeepFuse.git BioDeepFuse

$ cd BioDeepFuse

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
conda env create -f environment.yml -n biodeepfuse
```

**3 - Activate environment:**

```sh
conda activate biodeepfuse
```

**4 - You can deactivate the environment, using:**

```sh
conda deactivate
```

## How to use

When running the application, it is possible to use diverse arguments. They are shown using the `--help` argument. The options available are:

```bash
  -train TRAIN, --train TRAIN
                        Folder with FASTA training files

  -test TEST, --test TEST
                        Folder with FASTA testing files

  -epochs EPOCHS, --epochs EPOCHS
                        Number of epochs to train

  -patience PATIENCE, --patience PATIENCE
                        Epochs to stop training after loss plateau

  -encoding ENCODING, --encoding ENCODING
                        Encoding - 0: One-hot encoding, 1: K-mer embedding, 2: No encoding (only feature extraction), 3: All encodings (without feature extraction)

  -k K, --k K           Length of k-mers

  -concat CONCAT, --concat CONCAT
                        Concatenation type - 1: Directly, 2: Using dense layer before concatenation

  -feat_extraction FEAT_EXTRACTION [FEAT_EXTRACTION ...], --feat_extraction FEAT_EXTRACTION [FEAT_EXTRACTION ...]
                        Features to be extracted, e.g., 1 2 3 4 5 6. 1 = NAC, 2 = DNC, 3 = TNC, 4 = kGap, 5 = ORF, 6 = Fickett Score

  -features_exist FEATURES_EXIST, --features_exist FEATURES_EXIST
                        Features extracted previously - 0: False, 1: True; Default: False

  -algorithm ALGORITHM, --algorithm ALGORITHM
                        Algorithm - 0: Support Vector Machines (SVM), 1: Extreme Gradient Boosting (XGBoost), 2: Deep Learning

  -num_convs NUM_CONVS, --num_convs NUM_CONVS
                        Number of convolutional layers

  -activation ACTIVATION, --activation ACTIVATION
                        Activation to use - 0: ReLU, 1: Leaky ReLU; Default: ReLU

  -batch_norm BATCH_NORM, --batch_norm BATCH_NORM
                        Use Batch Normalization for Convolutional Layers - 0: False, 1: True; Default: False

  -cnn_dropout CNN_DROPOUT, --cnn_dropout CNN_DROPOUT
                        Dropout rate between Convolutional layers - 0 to 1

  -num_lstm NUM_LSTM, --num_lstm NUM_LSTM
                        Number of LSTM layers

  -bidirectional BIDIRECTIONAL, --bidirectional BIDIRECTIONAL
                        Use Bidirectional LSTM - 0: False, 1: True; Default: False

  -lstm_dropout LSTM_DROPOUT, --lstm_dropout LSTM_DROPOUT
                        Dropout rate between LSTM layers - 0 to 1

  -output OUTPUT, --output OUTPUT
                        Output folder for classification reports.
```

Alternatively, you can run the experiments used for the paper in `run_experiments.sh`.

## Citation

If you use this application in a scientific publication, we would appreciate citations to the following conference abstract:

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
