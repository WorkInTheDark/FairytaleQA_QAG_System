# FairytaleQA_QAG_System

For paper [```It is AI’s Turn to Ask Humans a Question: Question-Answer Pair Generation for Children’s Story Books```](https://arxiv.org/abs/2109.03423/)  [accepted to ACL 2022]

We have a separate repository for the FairytaleQA Dataset here: 

https://github.com/uci-soe/FairytaleQAData (this repo is not available now)

https://github.com/WorkInTheDark/FairytaleQA_Dataset

## What this repo is for
We developed an automated QA-pair generation (QAG) system for an education scenario: given a story book as input, our system can automatically generate QA-pairs that are capable of testing a variety of dimensions of a student's comprehension skills. We are using a new expert-annotated FairytaleQA dataset, which focuses on narrative comprehension for elementary to middle school students and contains 10,580 QA-pairs labeled by education experts from 278 classic fairytales.

For the fine-tune process and the end-2-end generation pipeline, We've been using the same version of transformers since we started the project to avoid version conflicts and it is included in this repo. You may find the latest version here: https://github.com/huggingface/transformers

## QA-pair Generation System Diagram

<!-- ![](/QAG2.png "QA-pair Generation System Diagram") -->
<p align="middle">
  <img src="/QAG2.png" alt="QA-pair Generation System Diagram" width=300/>
</p>
  
There are three sub-modules in our QAG pipeline: 
1. An answer generation(AG) module that leverages Spacy English model to extract named entities and noun chunks and Propbank’s Semantic Role Labeler to extract action events’ descriptions as candidate answers
2. A BART-based question generation(QG) module fine-tuned on FairytaleQA dataset 
3. A ranking module to rank and select top-<em>N</em> QA-pairs. We fine-tune a DistilBERT model on a classification task between QA-pairs generated with our QAG system and ground-truth from FairytaleQA Dataset


## What's here
We provide separate Jupyter Notebooks for the following task: 

* ```0_Pre_processing_the_original_data.ipynb``` --> Pre-processing the original story dataset into desired fine-tuning format. You may acquire the original story dataset from https://github.com/uci-soe/FairytaleQAData. Remember to put question and story files into one folder before using this notebook, so that the script can directly find the story file and question file for the same story.
* ```1_Train_BART_model.ipynb``` --> fine-tune a BART QG model
* ```2_Generate_QA_pairs_with_our_QAG_system.ipynb``` --> end-to-end QAG
* ```3_RANK_QA_on_test_val.ipynb``` --> Ranking module after generating QA-pairs with the previous Notebook 


To make things easy, we have pre-processed the original storys from FairytaleQA Dataset for QAG and stored them under ```./QAG_Generation_E2E/data/input_for_QAG```. In each pre-processed story file, each line is a section of the story. (A section is determined by human coders which contains multiple paragraphs) 

Thus, you may directly run ```2_Generate_QA_pairs_with_our_QAG_system.ipynb``` without the need to pre-process original story books by yourself if you just wish to view the generation results on FairytaleQA Dataset. (But you still need to get the model checkpoint below). Also, you may directly use the pre-processed story data to test your own QAG systems. 

Here are the model checkpoints that being used in the end-to-end QAG Notebook and the Ranking Module Notebook: 
* BART QG model: https://drive.google.com/file/d/16z6yOBv6JNm5eX5wmPTGSHqKf3NGMFDI/view?usp=sharing

  Load this model in ```2_Generate_QA_pairs_with_our_QAG_system.ipynb``` cell 9

* DistilBERT Ranking model: https://drive.google.com/drive/folders/1Mjg1ZQ0vltzy9WthOw7s9gRvB3g99pDI?usp=sharing

  Load this model in ```3_RANK_QA_on_test_val.ipynb``` cell 5
 
## Tips
* We would suggest using Google Colab so that you can copy the model to your drive and mount it to the Colab instance directly, since it'll be quite slow to download such large BART model from Google Drive.
* To run ```2_Generate_QA_pairs_with_our_QAG_system.ipynb```, you need to have a system with more than 16G RAM, and preferrably with GPU support.
* If you are using Google Colab, remember to restart the runtime after installing the dependencies (Colab will have an automatic prompt as well).

## Citation
Our Dataset Paper is accepted to ACL 2022, you may cite:
```
@inproceedings{yao2022storybookqag,
    author = {Yao, Bingsheng and Wang, Dakuo and Wu, Tongshuang and Zhang, Zheng and Li, Toby Jia-Jun and Yu, Mo and Xu, Ying},
    title = {It is AI's Turn to Ask Humans a Question: Question-Answer Pair Generation for Children's Story books},
    publisher = {Association for Computational Linguistics},
    year = {2022}
}
```
