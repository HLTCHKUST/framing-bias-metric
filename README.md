# NeuS: Neutral Multi-News Summarization for Mitigating Framing Bias

 <img src="img/pytorch-logo-dark.png" width="12%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Code for "NeuS: Neutral Multi-News Summarization for Mitigating Framing Bias", NAACL2022 [\[PDF\]](https://arxiv.org/pdf/2204.04902.pdf)

<img align="right" src="img/HKUST.jpeg" width="12%">

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@inproceedings{lee2022neus,
  title={NeuS: Neutral Multi-News Summarization for Mitigating Framing Bias},
  author={Lee, Nayeon and Bang, Yejin and Yu, Tiezheng and Madotto, Andrea and Fung, Pascale},
  journal={Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  year={2022}
}
</pre>

## 0. Setup
Run `pip install -r requirement.txt`

Acknowledgement: Our codes are based on Huggingface's [Transformer](https://github.com/huggingface/transformers) codebase. 

## 1. Dataset & Pre-processing
### How?
Run `data_preprocessing.ipynb`. 
* To replicate all our data preprocessing steps, run all codes. 
* To just prepare testset of your own to run with our pretrained model, run the `create_source_target_with_processing_probe_format(article_test, 'test')` function with your own `article_test` list. 

### Additional Details:
Datasets are inside `data` folder:
* `raw_crawled/` : contains crawled data from Allsides.com until 2021-10-19 (with all the meta dataset)
* `naacl2022_filtered_allsides_article.json` : filtered & preprocessed verion from `raw_crawled`. 
* `naacl2022_lrc_roundup_random_order_probe` : contains final train/val/test files used in our NeuS-Title model. 

Full article version (smaller subset): Not directly used in our paper, but releasing to help the community :blush:
* BASIL (cite) extended (neutral): **TODO**
* AllSides articles: **TODO**

## 2. Train Model
Run `bash finetune_neus_script.sh` with desired hyper-parameters \& correct configurations.

## 3. Generate neutral summary
1. Download our checkpoint from [here](https://drive.google.com/file/d/1z6tlfE0vxLN5x3UBYGHmO7EUDprBbUmW/view?usp=sharing) into `save/` directory
2. Run `bash generate_script.sh` with correct configurations (i.e. `PROJECT_DIR`, `OUTPUT_DIR`, `DATA_DIR`).

If you want to generate using your own model checkpoint, change the `OUTPUT_DIR` to be the directory containing your model checkpoint. 

## 4. Evaluate (Metric)
### Lexicon Resources:
All the lexicons are inside `data/lexicons`:
* `NRC-VAD-Lexicon-Aug2018Release/` : VAD lexicon used in our paper. Refer to the README inside the folder for citation details.
* `bias-lexicon/` : other lexicons that the community could find useful (not used in our work). Refer to README inside the folder for citation details.
* `bias_related_lexicons/` : other lexicons that the community could find useful (not used in our work). Refer to README inside the folder for citation details.

### Running the evaluation:
Run the codes in `metric_code.ipynb`. 
* Make sure you have `data/lexicons/NRC-VAD-Lexicon-Aug2018Release/`.
* Also, make sure you correctly load test generation texts and target texts into `gens` and `test_tgts` variables. 
