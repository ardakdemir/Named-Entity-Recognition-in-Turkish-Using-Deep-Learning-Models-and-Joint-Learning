# Named Entity Recognition in Turkish Using Deep Learning Models and Joint Learning

This page describes the work done in the Ms.Thesis by Arda Akdemir. The main aim of this page is to provide a framework for reproducing the work done in our Thesis conveniently.

Contributions planned:

* Dependency parser output as feature to ML model
  - Output of pretrained model and train on our own training set with [jPTDP](https://github.com/datquocnguyen/jPTDP) but need labels.
  * Joint learning of dependency parsing together with ML
  - We need a gold labeled data for dependency parser
* Combining various ML methods with Adaboost or some other boosting algorithm
* Extensive analysis of all frequently used ML methods
* Analysis of the problems and limitations related to the available dataset

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

