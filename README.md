# AMMEPP

## Datasets
Download the datasets (MIND-small and MIND-large) from https://msnews.github.io/.

## Environment


## Preprocessing
### Data Preprocessing
<pre><code>python data_preprocess.py --dataset MIND-small</code></pre>
<pre><code>python data_preprocess.py --dataset MIND-large</code></pre>

### Popularity Preprocessing
<pre><code>python popularity_preprocess.py --dataset MIND-small</code></pre>
<pre><code>python popularity_preprocess.py --dataset MIND-large</code></pre>

## Training, Validation, and Test
<pre><code>python main.py --dataset MIND-small</code></pre>
<pre><code>python main.py --dataset MIND-large</code></pre>
