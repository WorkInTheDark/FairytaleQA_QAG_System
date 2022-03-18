# FairytaleQA_QAG_System

For paper ```It is AI’s Turn to Ask Humans a Question: Question-Answer Pair Generation for Children’s Story Books``` [accepted to ACL 2022]

We have a separate Repository for the FairytaleQA Dataset here: https://github.com/uci-soe/FairytaleQAData

## To-do List
* [x] Add notebook to fine-tune BART QG model
* [x] Add notebook for end-2-end QA-pair Generation
* [x] Add notebook for Ranking module 
* [x] Add model weights
* [x] Add instruction

## What is this repo for
We design an automated QA-pair generation (QAG) system for an education scenario: given a story book at the kindergarten to eighth-grade level as input, our system can automatically generate QA-pairs that are capable of testing a variety of dimensions of a student's comprehension skills. We are using a new expert-annotated FairytaleQA dataset, which has 278 child-friendly storybooks with 10,580 QA pairs.

There are three sub-modules in our QAG pipeline: a heuristics-based answer generation module (AG), followed by a BART-based question generation module (QG) module fine-tuned on FairytaleQA dataset, and a DistilBERT-based ranking module fine-tuned on FairytaleQA dataset to rank and select top N QA-pairs for each input section.

For the fine-tune process and the end-2-end generation pipeline, We've been using the same version of transformers since we started the project to avoid version conflicts and it is included in this repo. You may find the latest version here: https://github.com/huggingface/transformers

## What's here
We provide separate Jupyter Notebooks for the following task: 

* ```1_Train_BART_model.ipynb``` --> fine-tune a BART QG model
* ```2_Generate_QA_pairs_with_our_QAG_system.ipynb``` --> end-to-end QAG
* ```3_RANK_QA_on_test_val.ipynb``` --> Ranking module after generating QA-pairs with the previous Notebook 

We also provide a Jupyter Notebook (```0_Pre_processing_the_original_data.ipynb```) for preprocessing the original story dataset into desired training format. You may acquite the original story dataset from the repo shared above. 

To make things easy, we have pre-processed the original storys for QAG and stored them under ```./QAG_Generation_E2E/data/input_for_QAG```, so you can directly run ```2_Generate_QA_pairs_with_our_QAG_system.ipynb``` without the need to pre-process original story books. (But you still need to get the model checkpoint below)

Here are the model checkpoints that being used in the end-to-end QAG Notebook and the Ranking Module Notebook: 
* BART QG model: https://drive.google.com/file/d/16z6yOBv6JNm5eX5wmPTGSHqKf3NGMFDI/view?usp=sharing

  Load this model in ```2_Generate_QA_pairs_with_our_QAG_system.ipynb``` cell 9

* DistilBERT Ranking model: https://drive.google.com/drive/folders/1Mjg1ZQ0vltzy9WthOw7s9gRvB3g99pDI?usp=sharing

  Load this model in ```3_RANK_QA_on_test_val.ipynb``` cell 5
 
## Tips
* We would suggest using Google Colab so that you can copy the model to your drive and mount it to the Colab instance directly, since it'll be quite slow to download such large BART model from Google Drive.
* To run ```2_Generate_QA_pairs_with_our_QAG_system.ipynb```, you need to have a system with more than 16G RAM, and preferrably with GPU support.

