# AMMEPP
This repository contains the implementation of AMMEPP, a personalized news recommendation model that adaptively models users’ multiple interests and enhances news popularity prediction to address cold-start problems.

## 1 Download
### 1-1 Datasets
Download the datasets (MIND-small and MIND-large) from https://msnews.github.io/.

### 1-2 Pre-trained Word Vectors
Download the pre-trained word vectors (glove.840B.300d.zip) from https://github.com/stanfordnlp/GloVe.

## 2 Preprocessing
### 2.1 Data Preprocessing
<pre><code>python data_preprocess.py --dataset MIND-small</code></pre>
<pre><code>python data_preprocess.py --dataset MIND-large</code></pre>

### 2.2 Popularity Preprocessing
<pre><code>python popularity_preprocess.py --dataset MIND-small</code></pre>
<pre><code>python popularity_preprocess.py --dataset MIND-large</code></pre>

## 3 Training, Validation, and Test
<pre><code>python main.py --dataset MIND-small</code></pre>
<pre><code>python main.py --dataset MIND-large</code></pre>
