# LANKA

This is the source code for paper: **Knowledgeable or Educated Guess? Revisiting Language Models as Knowledge Bases** (ACL 2021, long paper)

## Reference

If this repository helps you, please kindly cite the following bibtext:

```

```

## Usage

To reproduce our results:

### 1. Create conda environment and install requirements

```shell
conda create --name lanka python=3.7
conda activate lanka
pip install -r requirements.txt
```

## 2. Download the data

* Download the data using wget

  ```shell
  wget https://drive.google.com/uc?export=download&id=1oQ7TXrZ7aQXpZnENu2Sytc8A0D3yvqkP
  unzip data.zip
  rm data.zip
  ```

* Or you can acquire the data using the following Google Drive link.

  https://drive.google.com/file/d/1oQ7TXrZ7aQXpZnENu2Sytc8A0D3yvqkP/view?usp=sharing

## 3. Run the experiments

#### 3.1 Prompt-based Retrieval

* Evaluate the precision on LAMA and WIKI-UNI using different prompts:

  * $T_{man}$

    ```shell
    python -m scripts.run_prompt_based --relation-type lama_original --model-name bert-large-cased --method evaluation --cuda-device [device]
    ```

  * $T_{mine}$

    ```shell
    python -m scripts.run_prompt_based --relation-type lama_mine --model-name bert-large-cased --method evaluation --cuda-device [device]
    ```

  * $T_{auto}$

    ```shell
    python -m scripts.run_prompt_based --relation-type lama_auto --model-name bert-large-cased --method evaluation --cuda-device [device]
    ```

* Calculate the average percentage of instances being covered by top-k answers or predictions:

  ```shell
  python -m scripts.run_prompt_based --model-name bert-large-cased --method topk_cover --cuda-device [device]
  ```

* Store various distributions needed for subsequent experiments:

  ```shell
  python -m scripts.run_prompt_based --model-name bert-large-cased --method store_all_distribution --cuda-device [device]
  ```

* Calculate the Pearson correlations of the prediction distributions on LAMA and WIKI-UNI (Figure 3):

  ```shell
  python -m scripts.run_prompt_based --model-name bert-large-cased --method prediction_corr --cuda-device [device]
  ```

* Calculate the Pearson correlations between the prompt-only distribution and prediction distribution on WIKI-UNI (Figure 4):

  ```shell
  python -m scripts.run_prompt_based --model-name bert-large-cased --method prompt_only_corr --cuda-device [device]
  ```

* Calculate the KL divergence between the prompt-only distribution and golden answer distribution of LAMA (Table 2):

  ```shell
  python -m scripts.run_prompt_based --relation-type [relation_type] --model-name bert-large-cased --method evaluation --cuda-device [device]
  ```

####3.2 Case-based Analogy

* Evaluate case-based paradigm:

  ```shell
  python -m scripts.run_case_based --model-name bert-large-cased --task evaluate_analogy_reasoning --cuda-device [device]
  ```

* Detailed comparison for prompt-based and case-based  paradigms (precision $\Delta$, type precision $\Delta$, type change, etc.):

  ```shell
  python -m scripts.run_case_based --model-name bert-large-cased --task type_precision --cuda-device [device]
  ```

* Calculate the in-type rank change:

  ```shell
  python -m scripts.run_case_based --model-name bert-large-cased --task type_rank_change --cuda-device [device]
  ```

#### 3.3 Context-based Inference

* For explicit answer leakage:

  ```shell
  python -m scripts.run_context_based --model-name bert-large-cased --method explicit_leak --cuda-device [device]
  ```

* For implicit answer leakage:

  ```shell
  python -m scripts.run_context_based --model-name bert-large-cased --method implicit_leak --cuda-device [device]
  ```
