# Named Entity Recognition in Turkish Using Deep Learning Models and Joint Learning

This page describes the work done during the Ms.Thesis by Arda Akdemir. The main aim of this page is to provide a framework for reproducing the work done in our Thesis conveniently. 

We have developed a BiLSTM CRF based joint learning tool for Dependency Parsing and Named Entity Recognition using separate datasets.


The source codes can be found under the src folder.

### Reproducing the work

Dependencies:

```
python 2.7
dyNet
```

To train your own models you can use the program with the following code:

```
python main.py --predout [output file] --outdir [output directory] --params [parameter file] --model [model file] --train [training file for dependency parsing] --dev [deveplopment file for dependency parsing] --trainner [training file for named entity recognition] --devner [development file for named entity recognition] 
```

In order to use the system in the prediction mode you can use the following code:

```
python main.py --predict --model [model file to be inputted] --params [parameter file to be inputted] --predout [output file] --test [test file for dependency parsing] --testner [test file for named entity recognition]
```

Below you can find the results we have obtained for the joint learning system. 

|a|a|a|a|a|a|a|a|a|a|
|Type|Precision|Recall|F1|Precision|Recall|F1|Precision|Recall|F1|
|------|------|-------|-------|------|-------|-------|------|-------|-------|
|PER|89.74| 89.89|89.81 | 92.43|77.82|84.50 | 86.29|86.66 |86.48|
|LOC|  89.95 | 90.04 | 89.99 | 77.07|87.53|81.97| 86.84|85.89|86.36|
|ORG| 87.56 | 86.65 | 87.10 | 81.21|75.67|78.34| 80.97|76.41|78.63|
|Overall|  89.28 | 89.15 | **89.21** | 83.78|80.51|82.11| 85.23|83.91|**84.56**|



