# Attack On Sequential Embedding-Driven Attentive Malware Classification

## Setup
Clone the repo. Download the datasets and models that could not be uploaded to GitHub due to upload limit and place them into appropriate folder.

## Prerequisites
Ensure you all the necessary libraries installed
- numpy
- pandas
- os
- time
- gc (garbage collector)
- sklearn (scikit-learn)
- lightgbm
- random
- warnings
- tqdm
- keras
- torch
- xgboost
- RandomForestClassifier
- metrics (from scikit-learn)
- LogisticRegression
- SVC (Support Vector Classification)
- confusion_matrix (from scikit-learn)
- seaborn
- matplotlib.pyplot
- pickle
- FastText (from gensim.models)
- PCA (Principal Component Analysis from scikit-learn)
- TSNE (t-Distributed Stochastic Neighbor Embedding from scikit-learn)

## How to Run
  1. Once you have all the necessary files downloaded and libraries installed you can simply run the notebooks and python files using and IDE of your choice.

## System Requirements
Google Pro High Memory Ram and GPU
    
## Dataset
The project utilizes the "Microsoft BIG 15" benchmark dataset. You cn download it from the link below:

https://www.kaggle.com/c/malware-classification

Warning: this dataset is almost half a terabyte uncompressed! They have compressed the data using 7zip to achieve the smallest file size possible.
They provided a set of known malware files representing a mix of 9 different families. Each malware file has an Id, a 20 character hash value uniquely identifying the file, and a Class, an integer representing one of 9 family names to which the malware may belong:

Ramnit
Lollipop
Kelihos_ver3
Vundo
Simda
Tracur
Kelihos_ver1
Obfuscator.ACY
Gatak

For each file, the raw data contains the hexadecimal representation of the file's binary content, without the PE header (to ensure sterility).  You are also provided a metadata manifest, which is a log containing various metadata information extracted from the binary, such as function calls, strings, etc. This was generated using the IDA disassembler tool.

## Model Architecture
![Image](/content/model.png)
