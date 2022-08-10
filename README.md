# [KoSAIM 2022 Summer School] RNN Hands-on session. 

## Handson session.
### Google colab
https://colab.research.google.com/drive/1GwLQd2ij4EShd7H7cVtxU2tqAZVMAS1N?usp=sharing


# ICU Mortality Prediction

## Objective
Given ICU records where the length-of-stay is between 24 hours and 48 hours (i.e. 1 day <= length-of-stay <= 2 days), use the information from the first 3 hours to predict whether the patient will die during the ICU stay.
Use the Logistic Regression and the Gated Recurrent Unit (GRU) to compare performance.

## Dataset
- Tables & Features
  Each patient has one or more ICU admission records, where each ICU admission has a unique ID denoted by ICUSTAY_ID. Use the following columns from each table to form a patient representation vector. 

## IMPORTANT! 
- You must use CHARTTIME to form a sequence of events, so that you can use RNN.
  - CHARTEVENTS
    - ICUSTAY_ID
      - If icustay_id is NULL, disregard that row.
    - ITEMID
      - If you are curious about what the itemid represents, refer to D_ITEMS.
    - VALUENUM
      - If valuenum is NULL, disregard that row.
    - CHARTTIME
- Do not use features that are outside the first 3 hours. 
- Keep the maximum sequence length of each sample to 100!! This is to make the RNN training more manageable. If some ICU stay has more than 100 events within the first 3 hours, remove the extra events.


* You can also use other tables for additional features.
  E.g. GENDER from PATIENTS, ETHNICITY from ADMISSIONS

 
## Things to consider:
- Continuous values
Each discrete ITEMID comes with a continuous value. How would you use these two types of information to form a sequence input?

- How to use time?
Timestamps are continuous values. How would you use them when using RNN?

- What if two events happen at the same time?
There can be two measurements at the same CHARTTIME. Then how would you treat them when you create a sequence input?

- Unknown feature values
For example, if your training data does not include some ITEMID A, but your test data does, then how are you going to handle that?


## Labels
You don’t need to care about SUBJECT_ID. Each ICU admission will be treated independently, and therefore you only need to care about ICUSTAY_ID.

First, remove all ICU stays whose duration is either less than 24 hours or more than 48 hours. Then use the events from the first 3 hours only. This is to make the prediction task more challenging. (If you use the information from the entire ICU stay, predicting patient death would be quite easy). As for the mortality label, use the deathtime column of the ADMISSIONS table. 


## Data Split
ICU admission records whose ICUSTAY_ID ends with the digit 8 & 9 should be left out as a test set. Use the rest of the samples as the training set.


## Data Format
You can preprocess MIMIC-III so that your preprocessing code produces:
 - X_train.npy, X_test.npy
     - A text file of N rows, where N is the number of train/test samples
     - Each row contains a sequence of “timestamp:itemid:value”.
     - Convert CHARTTIME to timestamps such that the first timestamps are always 0, and the increments are by minutes. (CHARTTIME does not use seconds anyway)
     - Example: “0.0:223834:15.0 0.0:223835:100.0 30.0:224665:1.11 60.0:220179:137.0 60.0:220180:72.0 …”
     - (This is only an example. You can use other features, and storen in any format)


 - y_train.npy, y_test.npy
     - A Numpy vector (i.e. 1-dimensional matrix) whose values are binary (1: death during ICU, 0: no death during ICU)
Must use Numpy float format.
     - In order to confirm that you’ve pre-processed the dataset correctly, we will check the number of positive/negative labels in both ‘y_train.npy’ and ‘y_test.npy’. Make sure you store them in Numpy float array.


## Modeling & Evaluation
 - Use PyTorch LSTM to train an RNN classifier.
 - Use both AUROC and AUPRC as the evaluation metric.

