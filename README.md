# Qwen2Vec: Adapting Qwen for Dense Retrieval

## Instructions for Training the EBAR and EBAE Models

Follow these steps to train Qwen-2.5-0.5B with the EBAR and EBAE training method from the following paper:
'Llama2Vec: Unsupervised Adaptation of Large Language Models for Dense Retrieval'.

The idea is to use continuous pretraining on Wikipedia to train the model to compress the meaning of the entire input text into a special <EOS> token, just as Bert-like encoder model do with the CLS token. So the goal is to use the abilities of the autoregressive LLM to encode the meaning of the input text. The training loss consists of having the model predict all tokens of the input text.

---

## Step 1: Download Spanish Wikipedia Articles
Run the following notebook to extract the first 1000 articles from Spanish Wikipedia and save them as a pickle file:

```bash
get_spanish_wiki.ipynb
```

This script will handle data extraction and save the articles for further processing.

---

## Step 2: Prepare the Dataset
Use this notebook to preprocess the Wikipedia data and prepare it for EBAR and EBAE training:

```bash
prepare_dataset_for_ebar_ebae.ipynb
```

The preprocessing includes:
- Tokenizing the articles.
- Preparing chunks for input prompts and next sentences.

---

## Step 3: Train the Model
Finally, run the main training notebook to adapt the model with EBAR and EBAE methods:

```bash
llama_2_vec.ipynb
```

This notebook will:
- Load the preprocessed data.
- Train the model using the specified loss functions for EBAR and EBAE.
- Save the trained model for further use.

---

## Notes
- Ensure all dependencies are installed before running the notebooks.
- For detailed explanations of the training procedure, refer to the documentation in each notebook.

---

Happy training!
