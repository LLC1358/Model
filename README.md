# AMMEPP
This repository contains the implementation of AMMEPP, a personalized news recommendation model that adaptively models users’ multiple interests and enhances news popularity prediction to address cold-start problems. The model is evaluated on the MIND-small and MIND-large datasets.

## 1. Datasets
Download the benchmark datasets MIND-small and MIND-large from the official Microsoft News dataset page: from https://msnews.github.io/.

## 2. Preprocessing
### 2.1 Data Preprocessing
<pre><code>python data_preprocess.py --dataset MIND-small</code></pre>
<pre><code>python data_preprocess.py --dataset MIND-large</code></pre>

### 2.2 Popularity Preprocessing
<pre><code>python popularity_preprocess.py --dataset MIND-small</code></pre>
<pre><code>python popularity_preprocess.py --dataset MIND-large</code></pre>

## 3. Training, Validation, and Test
<pre><code>python main.py --dataset MIND-small</code></pre>
<pre><code>python main.py --dataset MIND-large</code></pre>
