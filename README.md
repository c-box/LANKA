# LANKA

This is the source code for paper: **Knowledgeable or Educated Guess? Revisiting Language Models as Knowledge Bases** (ACL 2021, long paper)

## Reference

If this repository helps you, please kindly cite the following bibtext:

```
@article{cao2021knowledgeable,
  title={Knowledgeable or Educated Guess? Revisiting Language Models as Knowledge Bases},
  author={Cao, Boxi and Lin, Hongyu and Han, Xianpei and Sun, Le and Yan, Lingyong and Liao, Meng and Xue, Tong and Xu, Jin},
  journal={arXiv preprint arXiv:2106.09231},
  year={2021}
}
```

## Usage

To reproduce our results:

### 1. Create conda environment and install requirements

```shell
git clone https://github.com/c-box/LANKA.git
cd LANKA
conda create --name lanka python=3.7
conda activate lanka
pip install -r requirements.txt
```

## 2. Download the data

* Download the data using terminal

  ```shell
  pip install gdown
  gdown https://drive.google.com/uc?id=1oQ7TXrZ7aQXpZnENu2Sytc8A0D3yvqkP
  unzip data.zip
  rm data.zip
  ```

* Or you can acquire the data using the following Google Drive link.

  https://drive.google.com/file/d/1oQ7TXrZ7aQXpZnENu2Sytc8A0D3yvqkP/view?usp=sharing

## 3. Run the experiments
If your GPU is smaller than 24G, please adjust batch size using "--batch-size" parameter.
#### 3.1 Prompt-based Retrieval

* Evaluate the precision on LAMA and WIKI-UNI using different prompts:

  * Manually prompts created by Petroni et al. (2019)

    ```shell
    python -m scripts.run_prompt_based --relation-type lama_original --model-name bert-large-cased --method evaluation --cuda-device [device] --batch-size [batch_size]
    ```

  * Mining-based prompts by Jiang et al. (2020b)

    ```shell
    python -m scripts.run_prompt_based --relation-type lama_mine --model-name bert-large-cased --method evaluation --cuda-device [device]
    ```

  * Automatically searched prompts from Shin et al. (2020)

    ```shell
    python -m scripts.run_prompt_based --relation-type lama_auto --model-name bert-large-cased --method evaluation --cuda-device [device]
    ```

* **Store various distributions needed for subsequent experiments:**

  ```shell
  python -m scripts.run_prompt_based --model-name bert-large-cased --method store_all_distribution --cuda-device [device]
  ```
  
* Calculate the average percentage of instances being covered by top-k answers or predictions (Table 1):

  ```shell
  python -m scripts.run_prompt_based --model-name bert-large-cased --method topk_cover --cuda-device [device]
  ```

* Calculate the Pearson correlations of the prediction distributions on LAMA and WIKI-UNI (Figure 3, the figures will be stored in the 'pics' folder):

  ```shell
  python -m scripts.run_prompt_based --model-name bert-large-cased --method prediction_corr --cuda-device [device]
  ```

* Calculate the Pearson correlations between the prompt-only distribution and prediction distribution on WIKI-UNI (Figure 4):

  ```shell
  python -m scripts.run_prompt_based --model-name bert-large-cased --method prompt_only_corr --cuda-device [device]
  ```

* Calculate the KL divergence between the prompt-only distribution and golden answer distribution of LAMA (Table 2):

  ```shell
  python -m scripts.run_prompt_based --relation-type [relation_type] --model-name bert-large-cased --method cal_prompt_only_div --cuda-device [device]
  ```

#### 3.2 Case-based Analogy

* Evaluate case-based paradigm:

  ```shell
  python -m scripts.run_case_based --model-name bert-large-cased --task evaluate_analogy_reasoning --cuda-device [device]
  ```

* Detailed comparison for prompt-based and case-based  paradigms (precision, type precision, type change, etc.) (Table 4):

  ```shell
  python -m scripts.run_case_based --model-name bert-large-cased --task type_precision --cuda-device [device]
  ```

* Calculate the in-type rank change (Figure 6):

  ```shell
  python -m scripts.run_case_based --model-name bert-large-cased --task type_rank_change --cuda-device [device]
  ```

#### 3.3 Context-based Inference

* For explicit answer leakage (Table 5 and 6):

  ```shell
  python -m scripts.run_context_based --model-name bert-large-cased --method explicit_leak --cuda-device [device]
  ```

* For implicit answer leakage (Table 7):

  ```shell
  python -m scripts.run_context_based --model-name bert-large-cased --method implicit_leak --cuda-device [device]
  ```
