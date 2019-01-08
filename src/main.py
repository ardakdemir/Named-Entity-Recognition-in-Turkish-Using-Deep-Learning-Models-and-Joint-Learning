# coding=utf-8
from optparse import OptionParser
import pickle, utils, learner, os, os.path, time


def convert(inputfile,outfile):
    out = open(outfile, "w")
    tags = ["BPER", "IPER", "BLOC", "ILOC", "BORG", "IORG"]
    for line in open(inputfile, "r"):
        w = line.replace("ROOT-POS", "O")
        for tag in tags:
            newtag = tag[0] + "-" + tag[1:]
            w = w.replace(tag, newtag)
        out.write(w)
    out.close()

def merge_counters(ctr1,ctr2):
    for word in ctr2:
        if word not in ctr1.keys():
            ctr1[word]=ctr2[word]
        else:
            ctr1[word]+=ctr2[word]
    return ctr1

def merge_c2i_dicts(c2i1,c2i2):
    for char in c2i2.keys():
        if char not in c2i1:
            c2i1[char] = len(c2i1)
    return c2i1

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Path to annotated CONLL train file", metavar="FILE", default="N/A")
    parser.add_option("--dev", dest="conll_dev", help="Path to annotated CONLL dev file", metavar="FILE", default="N/A")
    parser.add_option("--trainner", dest="conll_trainner", help="Path to annotated CONLL train ner file", metavar="FILE",
                      default="N/A")
    parser.add_option("--devner", dest="conll_devner", help="Path to annotated CONLL dev ner file", metavar="FILE", default="N/A")
    parser.add_option("--test", dest="conll_test", help="Path to CONLL test file", metavar="FILE", default="N/A")
    parser.add_option("--output", dest="conll_test_output", help="File name for predicted output", metavar="FILE", default="N/A")
    parser.add_option("--predout", dest="predout", help="File name for predicted appended output", metavar="FILE", default="predoutfile")
    parser.add_option("--prevectors", dest="external_embedding", help="Pre-trained vector embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="model.params")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=50)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=100)
    parser.add_option("--relembedding", type="int", dest="relembedding_dims", default=100)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    #parser.add_option("--lr", type="float", dest="learning_rate", default=None)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=128)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--disablelabels", action="store_false", dest="labelsFlag", default=True)
    parser.add_option("--disabledep", action="store_false", dest="depFlag", default=True)
    parser.add_option("--secondner", action="store_false", dest="sNerFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--bibi-lstm", action="store_false", dest="bibiFlag", default=True)
    parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)

    (options, args) = parser.parse_args()

    #print 'Using external embedding:', options.external_embedding

    if options.predictFlag:
        with open(options.params, 'r') as paramsfp:
            words, w2i, c2i, pos, rels, caps, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = None
        print 'Loading pre-trained model'
        parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, caps,stored_opt)
        parser.Load(options.model)
        
        testoutpath = os.path.join(options.output, options.conll_test_output)
        print 'Predicting POS tags and parsing dependencies'
        #ts = time.time()
        #test_pred = list(parser.Predict(options.conll_test))
        #te = time.time()
        #print 'Finished in', te-ts, 'seconds.'
        #utils.write_conll(testoutpath, test_pred)

        with open(testoutpath, 'w') as fh:
            for sentence in parser.Predict(options.conll_test):
                print sentence
                for entry in sentence[1:]:
                    fh.write(str(entry) + '\n')
                fh.write('\n')

    else:
        print("Training file: " + options.conll_train)
        if options.conll_dev != "N/A":
            print("Development file: " + options.conll_dev)

        highestScore = 0.0
        eId = 0
        flag1=1
        if os.path.isfile(os.path.join(options.output, options.params)) and \
                os.path.isfile(os.path.join(options.output, os.path.basename(options.model))) and flag1==0 :

            print 'Found a previous saved model => Loading this model'
            with open(os.path.join(options.output, options.params), 'r') as paramsfp:
                words, w2i, c2i, pos, rels, stored_opt = pickle.load(paramsfp)
            stored_opt.external_embedding = None
            parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, stored_opt)
            parser.Load(os.path.join(options.output, os.path.basename(options.model)))
            parser.trainer.restart()
            if options.conll_dev != "N/A":
                devPredSents = parser.Predict(options.conll_dev)
                count = 0
                lasCount = 0
                uasCount = 0
                posCount = 0
                tp=1
                fp=0
                fn=0
                eh=0
                poslasCount = 0
                for idSent, devSent in enumerate(devPredSents):
                    conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]

                    for entry in conll_devSent:
                        if entry.id <= 0:
                            continue
                        if entry.pos == entry.pred_pos and entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            poslasCount += 1
                        if entry.pos == entry.pred_pos:
                            posCount += 1
                            if entry.pos!="O":
                                tp+=1
                        if entry.pos != entry.pred_pos:
                            if entry.pos == "O":
                                fp+=1
                            else:
                                if entry.pred_pos!="O":
                                    eh+1
                                fn+=1
                        if entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            lasCount += 1
                        if entry.parent_id == entry.pred_parent_id:
                            uasCount += 1
                        count += 1

                print "---\nLAS accuracy:\t%.2f" % (float(lasCount) * 100 / count)
                print "UAS accuracy:\t%.2f" % (float(uasCount) * 100 / count)
                print "NER accuracy:\t%.2f" % (float(posCount) * 100 / count)
                print "NER precision:\t%.2f" % (float(tp) * 100 / (tp+fp))
                print "NER recall:\t%.2f" % (float(tp) * 100 / (tp+fn))
                print "NER eh accuracy:\t%.2f" % (float(tp+eh) * 100 / (tp+fn+eh))
                print "NER&LAS accuracy:\t%.2f" % (float(poslasCount) * 100 / count)

                score = float(poslasCount) * 100 / count
                if score >= highestScore:
                    parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                    highestScore = score

                print "POS&LAS of the previous saved model: %.2f" % (highestScore)

        else:
            ner_epoch = 1
            dep_epoch = 1
            print 'Extracting vocabulary'
            wordsdep, w2idep, c2idep, posdep, relsdep, capsdep = utils.vocab(options.conll_train)
            wordsner, w2iner, c2iner, posner, relsner, capsner = utils.vocab_ner(options.conll_trainner)
            words, c2i, pos, rels, caps = merge_counters(wordsdep,wordsner), merge_c2i_dicts(c2idep,c2iner), posner, relsdep, capsdep
            w2i = {w: i for i, w in enumerate(words.keys())}
            with open(os.path.join(options.output,  options.params), 'w') as paramsfp:
                pickle.dump((words, w2i, c2i, pos, rels, options), paramsfp)

            #print 'Initializing joint model'
            parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, caps,options)
        

        for epoch in xrange(options.epochs):
            print '\n-----------------\nStarting epoch', epoch + 1

            if epoch % 10 == 0:
                if epoch == 0:
                    parser.trainer.restart(learning_rate=0.001) 
                elif epoch == 10:
                    parser.trainer.restart(learning_rate=0.0005)
                else:
                    parser.trainer.restart(learning_rate=0.00025)
            parser.Train(options.conll_train,dep_epoch=1)
            parser.Train(options.conll_trainner,ner_epoch=1)
            
            if options.conll_dev == "N/A":  
                parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                
            else: 
                devPredSents = parser.Predict(options.conll_dev,ner_epoch=0)
                devNerPredSents = parser.Predict(options.conll_devner)
                out = open(os.path.join(options.output, options.predout), "w")
                count = 0
                lasCount = 0
                uasCount = 0
                posCount = 0
                tp=1.0
                fp=0.0
                fn=0.0
                eh=0

                poslasCount = 0
                for idSent, devSent in enumerate(devPredSents):
                    conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]
                    for entry in conll_devSent:
                        if entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            lasCount += 1
                        if entry.parent_id == entry.pred_parent_id:
                            uasCount += 1
                        count += 1
                for idSent, devnerSent in enumerate(devNerPredSents):
                    conll_devnerSent = [entry for entry in devnerSent if isinstance(entry, utils.ConllEntry)]
                    for entry in conll_devnerSent:
                        if entry.norm == "*root*":
                            out.write("\n")
                        out.write(entry.norm + '\t' + entry.pos + "\t" + entry.pred_pos + "\n")
                        if entry.id <= 0:
                            continue
                        if entry.pos == entry.pred_pos:
                            posCount += 1
                            if entry.pos != "O" and entry.pos != "ROOT-POS":
                                tp += 1
                        if entry.pos != entry.pred_pos:
                            if entry.pos == "O":
                                fp += 1
                            else:
                                if entry.pred_pos != "O":
                                    eh + 1
                                fn += 1
                out.close()
                ner_precision = tp / (tp+fp)
                ner_recall = tp / (tp + fn)
                print "---\nLAS accuracy:\t%.2f" % (float(lasCount) * 100 / count)
                print "UAS accuracy:\t%.2f" % (float(uasCount) * 100 / count)
                #print "NER accuracy:\t%.2f" % (float(posCount) * 100 / count)
                print "NER precision:\t%.2f" % (float(tp) * 100 / (tp + fp))
                print "NER recall:\t%.2f" % (float(tp) * 100 / (tp + fn))
                #print "NER eh accuracy:\t%.2f" % (float(tp + eh) * 100 / (tp + fn + eh))
                #print "NER&LAS accuracy:\t%.2f" % (float(poslasCount) * 100 / count)
                outscore=open("conllouts/out{}_score".format(epoch),"w")
                outscore.write("---\nLAS accuracy:\t%.2f" % (float(lasCount) * 100 / count))
                outscore.write("UAS accuracy:\t%.2f" % (float(uasCount) * 100 / count))
                #outscore.write("NER accuracy:\t%.2f" % (float(posCount) * 100 / count))
                outscore.write("NER precision:\t%.2f" % (float(tp) * 100 / (tp + fp)))
                outscore.write("NER recall:\t%.2f" % (float(tp) * 100 / (tp + fn)))
                #outscore.write("NER eh accuracy:\t%.2f" % (float(tp + eh) * 100 / (tp + fn + eh)))
                #outscore.write("NER&LAS accuracy:\t%.2f" % (float(poslasCount) * 100 / count))
                outscore.close()
                convert(os.path.join(options.output, options.predout),"conllouts/predout{}".format(epoch))
                dep_score = float(lasCount) * 100 / count
                ner_score = 2 * ner_precision * ner_recall / (ner_precision + ner_recall)
                score = (dep_score + ner_score) / 2
                print ("Combined Score : %.2f" %score)
                if score >= highestScore:
                    parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                    highestScore = score
                    eId = epoch + 1
                
                print "Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId)

