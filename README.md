# ADEPT

Source code and data for [*ADEPT: A DEbiasing PrompT Framework*](https://arxiv.org/abs/2211.05414) (**AAAI-23**).

An illustration of how debiasing works using **ADEPT** and for downstream tasks:

![](figures/Figure1-a.png)

(a) While debiasing, **ADEPT** only trains the prompt parameters and keeps the base model frozen.

![](figures/Figure1-b.png)

(b) When performing downstream tasks, **ADEPT** conditions the base model or both the prompt and the base model.

## Replication

We conduct experiments on the [**bert-large-uncased**](https://huggingface.co/bert-large-uncased) pre-trained model from [HuggingFace](https://huggingface.co/). By using **ADEPT**, we need only train 1.97M parameters when prompt-tuning with 40 prompt tokens, orders of magnitude smaller than the 335M parameters required for finetuning.

We provide bash scripts and codes to replicate our findings. Our environment is:

* Ubuntu servers with NVIDIA GeForce RTX 3090 (24G) GPUs
* cuda 11.1
* packages with certain versions

### Environment Setup

Create environment:

```bash
conda create -n debias python=3.8.5
conda activate debias
```

Install pytorch and python packages:

```bash
conda install -n pt2 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
cd ADEPT
pip install -r requirements.txt
```

### Data

We've already included word lists for attributes in the `./data` folder, so there is no need to acquire them from other resources. As for larger corpora, you can download News-Commentary v15 [here](https://data.statmt.org/news-commentary/v15/documents.tgz) and Hugging Face's BookCorpus replica [here](https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2). New-Commentary alone can support gender debiasing. You may need to create a new text file by combining the two corpora mentioned above so that there are sufficient sentences for religion debiasing. 

### Experiments

Collect sentences:

```bash
cd script
bash ./collect_sentences.sh bert [corpus_path] gender final
```

Debias:

```bash
bash ./debias.sh bert 0 ADEPT gender # for ADEPT
bash ./debias.sh bert 0 DPCE gender # for DPCE
```

Preprocess corpus for plotting words' correlation (for better visualizing pairwise words' correlation, we highly suggest that you choose a large corpus, like a subset of **BookCorpus** sampled with function `sample_sentences_from_bookcorpus` in `utils.py`, because we have set the minimum threshold for $len(S_m^{a(i)})$ if word $w_m^{a(i)}$ is to be plotted):

```bash
bash ./preprocess_plot_word_correlation.sh bert 0 gender [corpus_path]
```

Plot words' correlation:

```bash
bash ./plot_word_correlation.sh bert 0 ADEPT gender [model_name_or_path] # for ADEPT
bash ./plot_word_correlation.sh bert 0 DPCE gender [model_name_or_path] # for DPCE
```

### Visualization:

#### Gender:

*We color neutral words beige, male words blue, and female words red.*

(a) [**original**](https://huggingface.co/bert-large-uncased):

![](figures/Figure2-a.png)

(b) [**DPCE**](https://arxiv.org/abs/2101.09523):

![](figures/Figure2-b.png)

(c) **ADEPT-finetuning**:

![](figures/Figure2-c.png)

(d) **ADEPT**:

![](figures/Figure2-d.png)



#### Religion:

*We color neutral words grey, Judaism words yellow, Christianity words blue, and Islam words green.*

(a) [**original**](https://huggingface.co/bert-large-uncased):

![](figures/Figure3-a.png)

(b) **ADEPT**:

![](figures/Figure3-b.png)
