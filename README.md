# ONCLUSIVE ML CHALLENGE
Task: Build an ML system to verify the veracity of claims.

**Dataset:** PUBHEALTH is a comprehensive dataset for explainable automated fact-checking of public health claims. Each instance in the PUBHEALTH dataset has an associated veracity label (true, false, unproven, mixture). Furthermore each instance in the dataset has an explanation text field. The explanation is a justification for which the claim has been assigned a particular veracity label.

Dataset link: https://huggingface.co/datasets/health_fact

Pretrained huggingface model used: : https://huggingface.co/yikuan8/Clinical-Longformer

**Solution:** Modelled veracity verification as a multi-class classification problem. Given a pair of 'claim' and 'source or evidence' in natural language, one of the four veracity classes (true, false, unproven, mixture) is predicted. This is a natural language inference task where a claim is verified against evidence for veracity. Following steps were taken to build the ML system:

The dataset was downloaded as train, validation and test splits using huggingface's datasets library.
Data was preprocessed by combining 'claim' and 'main_text' columns and oversampling the minority class.
A pretrained tokenizer was used to tokenize and convert input text into indices and attention masks.
Since the data consists of very long text instances about health related claims, the **Clinical-Longformer** was used with a sequence classification head to train a multi-class classification network.
## Clinical-Longformer is a clinical knowledge enriched version of Longformer that was further pre-trained using MIMIC-III clinical notes. It allows up to 4,096 tokens as the model input.

Distributed training was performed on a 4 gpu machine using huggingface's accelerate library.

# Performance of the final model:

## Validation set:

Loss: 0.58121

Micro F1: 0.73229

Macro F1: 0.636882

False F1: 0.67

Mixture F1: 0.46

True F1: 0.90

Unproven F1: 0.52

## Test set:

Loss: 0.646326

Micro F1: 0.729116

Macro F1: 0.64037

False F1: 0.71

Mixture F1: 0.51

True F1: 0.87

Unproven F1: 0.47


**Note:** Due to github size limits, the model .bin file was not uploaded to models/2 directory. Please download the model file from https://drive.google.com/drive/folders/17-HGOm59_YwBKD8OmQNgyxhzojcjZgjL?usp=sharing and place in the models/2 directory to run the evaluation code.
