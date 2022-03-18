# FairytaleQA_QAG_System

For paper ```It is AI’s Turn to Ask Humans a Question: Question-Answer Pair Generation for Children’s Story Books``` [accepted to ACL 2022]

We have a separate Repository for the FairytaleQA Dataset here: https://github.com/uci-soe/FairytaleQAData

## To-do List
* [ ] Add notebook for fine-tune BART QG model
* [x] Add notebook for end-2-end QA-pair Generation
* [x] Add notebook for Ranking module 
* [ ] Add model weights
* [ ] Add instruction

## Instruction
We design an automated question-answer generation (QAG) system for an education scenario: given a story book at the kindergarten to eighth-grade level as input, our system can automatically generate QA pairs that are capable of testing a variety of dimensions of a student's comprehension skills. We are using a new expert-annotated FairytaleQA dataset, which has 278 child-friendly storybooks with 10,580 QA pairs.

There are three sub-modules in our QAG pipeline: a heuristics-based answer generation module (AG), followed by a BART-based question generation module (QG) module fine-tuned on FairytaleQA dataset, and a DistilBERT-based ranking module fine-tuned on FairytaleQA dataset to rank and select top N QA-pairs for each input section.

## What's here
We provide separate Jupyter Notebooks for the following task: 1). fine-tune a BART QG model, 2). end-to-end QAG, and 3). ranking module after generating QA-pairs. 
We also provide a Jupyter Notebook for preprocessing the original story dataset into desired training format. 

Here are the model checkpoints that being used in the end-to-end QAG Notebook and the Ranking Module Notebook: (We would suggest using Google Colab so that you can copy the model to your drive and mount it to the instance directly, since BART model is quite big

* BART QG model: https://drive.google.com/file/d/16z6yOBv6JNm5eX5wmPTGSHqKf3NGMFDI/view?usp=sharing

  Load this model in ```2_Generate_QA_pairs_with_our_QAG_system.ipynb``` cell 9

* DistilBERT Ranking model: https://drive.google.com/drive/folders/1Mjg1ZQ0vltzy9WthOw7s9gRvB3g99pDI?usp=sharing

  Load this model in ```3_RANK_QA_on_test_val.ipynb``` cell 5
