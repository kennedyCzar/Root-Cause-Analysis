# ROOTCAUSEANALYSIS
Git to work the week on that project

Project realized on Python

* Logistic_regression -> LR on the first events on we print the coefficients to analyze the global trend
* CRF-> the model of the CRF with the features of interest
* BiLSTM_CRF-> modeling using conditional chain of probabilities and analyzing the weight contributon of the Neural Network
* Transition_exploration-> to analyze transition probability and frequency of appration 
* time_an -> some utils to analyze time in the dataset, the class related to the logistic is in this file
* Pre_process-> some pre_process and utils are in this folder
* Markov_chain -> some codes to estimate te transition matrix





## Different link of interest for CRF

* https://en.wikipedia.org/wiki/Conditional_random_field#Examples
* https://en.wikipedia.org/wiki/Structured_support_vector_machine
* https://towardsdatascience.com/conditional-random-field-tutorial-in-pytorch-ca0d04499463
* https://www.kaggle.com/code/shoumikgoswami/ner-using-random-forest-and-crf/notebook
* https://towardsdatascience.com/conditional-random-fields-explained-e5b8256da776
* https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf


## Models used for Experiments

* Bidirectional Long Short Term Memory-Conditional Random Fields (BiLSTM-CRF)
* Explainable One-class classification with the transition profile
* Logistic Regression on the three first step of the Process
* Event clustering with Levenschstein distance.

## Citations
```
@inproceedings{Panchendrarajan2018BidirectionalLF,
  title={Bidirectional LSTM-CRF for Named Entity Recognition},
  author={Rrubaa Panchendrarajan and Aravindh Amaresan},
  booktitle={Pacific Asia Conference on Language, Information and Computation},
  year={2018}
}
```