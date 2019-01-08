# coding=utf-8
from dynet import *
import dynet
from utils import read_conll,read_conll_ner, read_conll_predict,read_conll_predict_ner, write_conll, load_embeddings_file
from operator import itemgetter
import utils, time, random, decoder
import numpy as np
from mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor

class jPosDepLearner:
    def __init__(self, vocab, pos, rels, w2i, c2i, caps,options):
        self.model = ParameterCollection()
        random.seed(1)
        self.trainer = AdamTrainer(self.model)
        #if options.learning_rate is not None:
        #    self.trainer = AdamTrainer(self.model, alpha=options.learning_rate)
        #    print("Adam initial learning rate:", options.learning_rate)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify,
                            'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag
        self.depFlag = options.depFlag
        self.sNerFlag=options.sNerFlag
        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.cdims = options.cembedding_dims
        self.reldims=options.relembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.id2pos = {ind: word for ind, word in enumerate(pos)}
        self.c2i = c2i
        self.caps= {word: ind for ind, word in enumerate(caps)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels
        self.pdims = options.pembedding_dims
        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2
        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims))
        self.plookup = self.model.add_lookup_parameters((len(pos), self.pdims))
        self.caps_lookup=self.model.add_lookup_parameters((len(caps),self.cdims))
        transition_array=np.random.rand(len(pos)+2,len(pos)+2)
        #cap_array=np.random.rand(len(caps),len(pos))
        def normalizeprobs(arr):
            return np.array([np.divide(arr1,sum(arr1)) for arr1 in arr])
        self.nertrans_lookup=self.model.add_lookup_parameters((len(pos)+2,len(pos)+2))
        #self.caplookup = self.model.lookup_parameters_from_numpy(normalizeprobs(cap_array))
        if options.external_embedding is not None:
            ext_embeddings, ext_emb_dim = load_embeddings_file(options.external_embedding, lower=True)
            assert (ext_emb_dim == self.wdims)
            print("Initializing word embeddings by pre-trained vectors")
            count = 0
            for word in self.vocab:
                _word = unicode(word, "utf-8")
                if _word in ext_embeddings:
                    count += 1
                    self.wlookup.init_row(self.vocab[word], ext_embeddings[_word])
            print("Vocab size: %d; #words having pretrained vectors: %d" % (len(self.vocab), count))

        self.ffSeqPredictor = FFSequencePredictor(Layer(self.model, self.ldims * 2, len(self.pos), "idt"))

        self.hidden_units = options.hidden_units

        if not self.depFlag:

            self.pos_builders = [VanillaLSTMBuilder(1, self.wdims + self.cdims * 3, self.ldims, self.model),
                                 VanillaLSTMBuilder(1, self.wdims + self.cdims * 3, self.ldims, self.model)]
            self.pos_bbuilders = [VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                                  VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]

            self.ffSeqPredictor = FFSequencePredictor(Layer(self.model, self.ldims * 2, len(self.pos),softmax))

            self.hidden_units = options.hidden_units

        if self.depFlag:

            if self.bibiFlag:
                self.builders = [
                    VanillaLSTMBuilder(1, self.wdims + self.cdims * 3 , self.ldims, self.model),
                    VanillaLSTMBuilder(1, self.wdims + self.cdims * 3 , self.ldims, self.model)]
                self.bbuilders = [VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                                  VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
            elif self.layers > 0:
                self.builders = [
                    VanillaLSTMBuilder(self.layers, self.wdims + self.cdims * 3 , self.ldims, self.model),
                    VanillaLSTMBuilder(self.layers, self.wdims + self.cdims * 3 , self.ldims, self.model)]
            else:
                self.builders = [SimpleRNNBuilder(1, self.wdims + self.cdims * 3, self.ldims, self.model),
                                 SimpleRNNBuilder(1, self.wdims + self.cdims * 3, self.ldims, self.model)]

            self.hidBias = self.model.add_parameters((self.ldims * 8))
            self.hidLayer = self.model.add_parameters((self.hidden_units, self.ldims * 8))
            self.hid2Bias = self.model.add_parameters((self.hidden_units))

            self.outLayer = self.model.add_parameters((1, self.hidden_units if self.hidden_units > 0 else self.ldims * 8))

            if self.labelsFlag:
                self.rhidBias = self.model.add_parameters((self.ldims * 8))
                self.rhidLayer = self.model.add_parameters((self.hidden_units, self.ldims * 8))
                self.rhid2Bias = self.model.add_parameters( (self.hidden_units))
                self.routLayer = self.model.add_parameters(
                    (len(self.irels), self.hidden_units if self.hidden_units > 0 else self.ldims * 8))
                self.routBias = self.model.add_parameters((len(self.irels)))
                self.ffRelPredictor = FFSequencePredictor(
                    Layer(self.model, self.hidden_units if self.hidden_units > 0 else self.ldims * 8, len(self.irels),
                          softmax))

            if self.sNerFlag:
                self.sner_builders=[VanillaLSTMBuilder(1, self.wdims + self.cdims * 3 + self.reldims, self.ldims, self.model),
                                    VanillaLSTMBuilder(1, self.wdims + self.cdims * 3 + self.reldims, self.ldims, self.model)]

                self.sner_bbuilders=[VanillaLSTMBuilder(1, self.ldims*2, self.ldims, self.model),
                                     VanillaLSTMBuilder(1, self.ldims*2, self.ldims, self.model)]
                ##relation embeddings
                self.rellookup=self.model.add_lookup_parameters((len(self.rels),self.reldims))



        self.char_rnn = RNNSequencePredictor(LSTMBuilder(1, self.cdims, self.cdims, self.model))

    def __getExpr(self, sentence, i, j):

        if sentence[i].headfov is None:
            sentence[i].headfov = concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].modfov is None:
            sentence[j].modfov = concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        _inputVector = concatenate(
            [sentence[i].headfov, sentence[j].modfov, dynet.abs(sentence[i].headfov - sentence[j].modfov),
             dynet.cmult(sentence[i].headfov, sentence[j].modfov)])

        if self.hidden_units > 0:
            output = self.outLayer.expr() * self.activation(
                self.hid2Bias.expr() + self.hidLayer.expr() * self.activation(
                    _inputVector + self.hidBias.expr()))
        else:
            output = self.outLayer.expr() * self.activation(_inputVector + self.hidBias.expr())

        return output

    def __evaluate(self, sentence):
        exprs = [[self.__getExpr(sentence, i, j) for j in xrange(len(sentence))] for i in xrange(len(sentence))]
        scores = np.array([[output.scalar_value() for output in exprsRow] for exprsRow in exprs])

        return scores, exprs

    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))
    def pick_neg_log_2(self, pred_param, gold):
        gold_arr=inputVector([1 if gold==i else 0 for i in range(len(self.pos)+2)])
        x=scalarInput(1)
        pred_arr=softmax(pred_param*x)
        return -dynet.log(transpose(pred_arr)*gold_arr)
    def pick_gold_score(self,preds,golds):
        score=0
        prev_tag=len(self.pos)
        for pred , gold in zip(preds,golds):
            score+=dynet.pick(pred,gold)+dynet.pick(self.nertrans_lookup[gold],prev_tag)
            prev_tag=gold
        score+=dynet.pick(self.nertrans_lookup[len(self.pos)+1],prev_tag)
        return score

    def pick_crf_score(self,preds,golds):
        return dynet.exp(self.pick_gold_tag_score(preds,golds)+self.pick_gold_trans_score(golds))

    def forward_score(self,preds):
        def log_sum_exp(tag_score_arr):
            argmax=np.argmax(tag_score_arr.value())
            max_score=tag_score_arr[argmax]
            score=max_score
            max_arr = dynet.concatenate([max_score for i  in range(len(self.pos)+2)]  )
            score+=dynet.log(dynet.sum_dim(dynet.exp(tag_score_arr-max_arr),[0]))
            return score
        score=0
        len1=len(self.pos)+2
        for_score=[-1e10 for i in range(len1)]
        for_score[-2]=0
        #print(len(preds))
        for i , pred in enumerate(preds):
            tag_scores=[dynet.scalarInput(-1e10) for j in range(len1)]
            for i,score in enumerate(pred):
                tag_score=dynet.concatenate([score+dynet.pick(self.nertrans_lookup[i],prev_tag)+for_score[prev_tag]
                                              for prev_tag in range(len1)])
                log_1=log_sum_exp(tag_score)
                tag_scores[i]=log_1
                #print("tag score: %f"%log_1.value())
            for_score=tag_scores
            #print(dynet.concatenate(for_score).value())
        term_exp=dynet.concatenate([score+tr for score,tr in zip(for_score , self.nertrans_lookup[len(self.pos)+1])])
        term_score=log_sum_exp(term_exp)
        #print("score : %f"%term_score.value())
        return term_score

    def nextPerm(self, perm1, taglen):
        a = []
        for ind, x in enumerate(reversed(perm1)):
            if x < taglen - 1:
                for i in range(len(perm1) - ind - 1):
                    a.append(perm1[i])
                a.append(x + 1)
                for i in range(ind):
                    a.append(0)
                return a
        return -1
    ## takes toooo long
    def forward_score2(self,taglen,senlen,preds):
        score=0
        perm1=[0 for i in range(senlen)]
        while perm1!=-1:
            score+=self.pick_crf_score(preds,perm1)
            perm1=self.nextPerm(perm1,taglen)
        return score


    def __getRelVector(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov = concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])
        _outputVector = concatenate(
            [sentence[i].rheadfov, sentence[j].rmodfov, abs(sentence[i].rheadfov - sentence[j].rmodfov),
             cmult(sentence[i].rheadfov, sentence[j].rmodfov)])

        if self.hidden_units > 0:
            return self.rhid2Bias.expr() + self.rhidLayer.expr() * self.activation(
                _outputVector + self.rhidBias.expr())
        else:
            return _outputVector

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def Predict(self, conll_path,dep_epoch=1 , ner_epoch=1):
        with open(conll_path, 'r') as conllFP:
            if ner_epoch == 0 :
                read_conll_nerdep = read_conll_predict(conllFP, self.c2i, self.wordsCount)
            else:
                read_conll_nerdep = read_conll_predict_ner(conllFP, self.c2i, self.wordsCount)
            for iSentence, sentence in enumerate(read_conll_nerdep):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    capvec  = self.caps_lookup[entry.capInfo]
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None

                    last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in entry.idChars])[-1]
                    rev_last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in reversed(entry.idChars)])[
                        -1]

                    entry.vec = concatenate(filter(None, [wordvec, last_state, rev_last_state,capvec]))
                    entry.vec2 = concatenate(filter(None, [wordvec, last_state, rev_last_state, capvec]))

                    entry.pos_lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                if not self.depFlag:

                    #Predicted pos tags
                    lstm_forward = self.pos_builders[0].initial_state()
                    lstm_backward = self.pos_builders[1].initial_state()
                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.pos_lstms[1] = lstm_forward.output()
                        rentry.pos_lstms[0] = lstm_backward.output()

                    for entry in conll_sentence:
                        entry.pos_vec = concatenate(entry.pos_lstms)

                    blstm_forward = self.pos_bbuilders[0].initial_state()
                    blstm_backward = self.pos_bbuilders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        blstm_forward = blstm_forward.add_input(entry.pos_vec)
                        blstm_backward = blstm_backward.add_input(rentry.pos_vec)
                        entry.pos_lstms[1] = blstm_forward.output()
                        rentry.pos_lstms[0] = blstm_backward.output()

                    concat_layer = [concatenate(entry.pos_lstms) for entry in conll_sentence]
                    #cap_info_sentence=[self.caplookup[entry.capInfo] for entry in conll_sentence]
                    outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                    best_parentids, bestscores = self.ffSeqPredictor.viterbi_sequence(outputFFlayer,
                                                                                      self.nertrans_lookup)
                    predicted_pos_indices = [np.argmax(o.value()) for o in outputFFlayer]
                    root_predicted_postags = ["O"]
                    predicted_postags = [self.id2pos[idx] for idx in best_parentids]
                    for pos in predicted_postags:
                        root_predicted_postags.append(pos)
                    if iSentence < 5:
                        for word, tag in zip(conll_sentence, root_predicted_postags):
                            print("word : {}  gold : {} pred : {}".format(word.form, word.pos, tag))
                    for entry, pos in zip(conll_sentence, root_predicted_postags):
                        entry.pred_pos = pos
                    dump = False

                if self.depFlag:

                    # Add predicted pos tags for parsing prediction
                    #for entry, posid in zip(conll_sentence, viterbi_pred_tagids):
                    #    entry.vec = concatenate([entry.vec, self.plookup[posid]])
                    #    entry.lstms = [entry.vec, entry.vec]
                    for entry in conll_sentence:

                        entry.lstms=[entry.vec,entry.vec]

                    if self.blstmFlag:
                        lstm_forward = self.builders[0].initial_state()
                        lstm_backward = self.builders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            lstm_forward = lstm_forward.add_input(entry.vec)
                            lstm_backward = lstm_backward.add_input(rentry.vec)
                            entry.lstms[1] = lstm_forward.output()
                            rentry.lstms[0] = lstm_backward.output()

                        if self.bibiFlag:
                            for entry in conll_sentence:
                                entry.vec = concatenate(entry.lstms)

                            blstm_forward = self.bbuilders[0].initial_state()
                            blstm_backward = self.bbuilders[1].initial_state()

                            for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                                blstm_forward = blstm_forward.add_input(entry.vec)
                                blstm_backward = blstm_backward.add_input(rentry.vec)

                                entry.lstms[1] = blstm_forward.output()
                                rentry.lstms[0] = blstm_backward.output()

                    scores, exprs = self.__evaluate(conll_sentence)
                    heads = decoder.parse_proj(scores)

                    # Multiple roots: heading to the previous "rooted" one
                    rootCount = 0
                    rootWid = -1
                    for index, head in enumerate(heads):
                        if head == 0:
                            rootCount += 1
                            if rootCount == 1:
                                rootWid = index
                            if rootCount > 1:
                                heads[index] = rootWid
                                rootWid = index

                    for entry, head in zip(conll_sentence, heads):
                        entry.pred_parent_id = head
                        entry.pred_relation = '_'
                        #entry.pred_pos = pos


                    if self.labelsFlag:
                        concat_layer = [self.__getRelVector(conll_sentence, head, modifier + 1) for modifier, head in
                                        enumerate(heads[1:])]
                        outputFFlayer = self.ffRelPredictor.predict_sequence(concat_layer)
                        predicted_rel_indices = [np.argmax(o.value()) for o in outputFFlayer]
                        predicted_rels = [self.irels[idx] for idx in predicted_rel_indices]
                        for modifier, head in enumerate(heads[1:]):
                            conll_sentence[modifier + 1].pred_relation = predicted_rels[modifier]

                    if self.sNerFlag and ner_epoch==1:

                        conll_sentence[0].vec=concatenate([conll_sentence[0].vec2,self.rellookup[self.rels["rroot"]]])
                        for entry , pred in zip(conll_sentence[1:],predicted_rel_indices):
                            relvec=self.rellookup[pred]
                            # for entry, posid in zip(conll_sentence, viterbi_pred_tagids):
                            entry.vec = concatenate([entry.vec2, relvec])
                        for entry in conll_sentence:
                            entry.ner2_lstms = [entry.vec, entry.vec]

                        slstm_forward = self.sner_builders[0].initial_state()
                        slstm_backward = self.sner_builders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            lstm_forward = slstm_forward.add_input(entry.vec)
                            lstm_backward = slstm_backward.add_input(rentry.vec)

                            entry.ner2_lstms[1] = lstm_forward.output()
                            rentry.ner2_lstms[0] = lstm_backward.output()

                        for entry in conll_sentence:
                            entry.ner2_vec = concatenate(entry.ner2_lstms)

                        sblstm_forward = self.sner_bbuilders[0].initial_state()
                        sblstm_backward = self.sner_bbuilders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            blstm_forward = sblstm_forward.add_input(entry.ner2_vec)
                            blstm_backward = sblstm_backward.add_input(rentry.ner2_vec)

                            entry.ner2_lstms[1] = blstm_forward.output()
                            rentry.ner2_lstms[0] = blstm_backward.output()

                        concat_layer = [dynet.dropout(concatenate(entry.ner2_lstms), 0.33) for entry in conll_sentence]
                        outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                        best_parentids, bestscores = self.ffSeqPredictor.viterbi_sequence(outputFFlayer, self.nertrans_lookup)
                        predicted_pos_indices = [np.argmax(o.value()) for o in outputFFlayer]
                        root_predicted_postags=["O"]
                        predicted_postags = [self.id2pos[idx] for idx in best_parentids]
                        for pos in predicted_postags:
                            root_predicted_postags.append(pos)
                        if iSentence < 1:
                            for word, tag in zip(conll_sentence,root_predicted_postags):
                                print("word : {}  gold : {} pred : {}".format(word.form,word.pos,tag))
                        for entry, pos in zip(conll_sentence,root_predicted_postags):
                            entry.pred_pos = pos

                    dump = False

                renew_cg()
                if not dump:
                    yield sentence

    def Train(self, conll_path,dep_epoch=0,ner_epoch=0):
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()
        dep_epoch = dep_epoch
        ner_epoch = ner_epoch
        with open(conll_path, 'r') as conllFP:
            if ner_epoch == 0 :
                read_conll_nerdep = read_conll(conllFP, self.c2i)
            else:
                read_conll_nerdep = read_conll_ner(conllFP, self.c2i)
            shuffledData = list(read_conll_nerdep)
            random.shuffle(shuffledData)

            errs = []
            lerrs = []
            posErrs = 0
            postrErrs=[]
            nertr2Errs = []
            ner2Errs = dynet.inputVector([0])
            startind = 0
            e = 0
            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 500 == 0 and iSentence != 0:
                    print "Processing sentence number: %d" % iSentence, ", Loss: %.4f" % (
                                eloss / etotal), ", Time: %.2f" % (time.time() - start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    c = float(self.wordsCount.get(entry.norm, 0))
                    dropFlag = (random.random() < (c / (0.25 + c)))
                    capvec  = self.caps_lookup[entry.capInfo]
                    wordvec = self.wlookup[
                        int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None

                    last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in entry.idChars])[-1]
                    rev_last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in reversed(entry.idChars)])[
                        -1]

                    entry.vec = dynet.dropout(concatenate(filter(None, [wordvec, last_state, rev_last_state,capvec])), 0.33)
                    entry.vec2=entry.vec
                    entry.pos_lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                if not self.depFlag:

                    #NER tagging loss
                    lstm_forward = self.pos_builders[0].initial_state()
                    lstm_backward = self.pos_builders[1].initial_state()
                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.pos_lstms[1] = lstm_forward.output()
                        rentry.pos_lstms[0] = lstm_backward.output()

                    for entry in conll_sentence:
                        entry.pos_vec = concatenate(entry.pos_lstms)

                    blstm_forward = self.pos_bbuilders[0].initial_state()
                    blstm_backward = self.pos_bbuilders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        blstm_forward = blstm_forward.add_input(entry.pos_vec)
                        blstm_backward = blstm_backward.add_input(rentry.pos_vec)

                        entry.pos_lstms[1] = blstm_forward.output()
                        rentry.pos_lstms[0] = blstm_backward.output()

                    concat_layer = [dynet.dropout(concatenate(entry.pos_lstms), 0.33) for entry in conll_sentence]
                    cap_info_sentence=[self.caps_lookup[entry.capInfo] for entry in conll_sentence]
                    outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                    posIDs = [self.pos.get(entry.pos) for entry in conll_sentence]
                    posErrs=(self.forward_score(outputFFlayer) - self.pick_gold_score(outputFFlayer,posIDs))

                ##dependency Flag
                if self.depFlag:
                    # Add predicted ner tags
                    #for entry, poses in zip(conll_sentence, outputFFlayer):
                    #    entry.vec = concatenate([entry.vec, dynet.dropout(self.plookup[np.argmax(poses.value())], 0.33)])
                    for entry in conll_sentence:
                        entry.lstms = [entry.vec, entry.vec]

                    #Parsing losses
                    if self.blstmFlag:
                        lstm_forward = self.builders[0].initial_state()
                        lstm_backward = self.builders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            lstm_forward = lstm_forward.add_input(entry.vec)
                            lstm_backward = lstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = lstm_forward.output()
                            rentry.lstms[0] = lstm_backward.output()

                        if self.bibiFlag:
                            for entry in conll_sentence:
                                entry.vec = concatenate(entry.lstms)

                            blstm_forward = self.bbuilders[0].initial_state()
                            blstm_backward = self.bbuilders[1].initial_state()

                            for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                                blstm_forward = blstm_forward.add_input(entry.vec)
                                blstm_backward = blstm_backward.add_input(rentry.vec)

                                entry.lstms[1] = blstm_forward.output()
                                rentry.lstms[0] = blstm_backward.output()

                    scores, exprs = self.__evaluate(conll_sentence)
                    gold = [entry.parent_id for entry in conll_sentence]
                    heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

                    if self.labelsFlag:

                        concat_layer = [dynet.dropout(self.__getRelVector(conll_sentence, head, modifier + 1), 0.33) for
                                        modifier, head in enumerate(gold[1:])]
                        outputFFlayer = self.ffRelPredictor.predict_sequence(concat_layer)
                        if dep_epoch==1:
                            relIDs = [self.rels[conll_sentence[modifier + 1].relation] for modifier, _ in
                                      enumerate(gold[1:])]
                            for pred, goldid in zip(outputFFlayer, relIDs):
                                lerrs.append(self.pick_neg_log(pred, goldid))
                    if dep_epoch==1:
                        e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])

                    if self.sNerFlag and ner_epoch==1:

                        conll_sentence[0].vec=concatenate([conll_sentence[0].vec2,self.rellookup[self.rels["rroot"]]])
                        for entry , pred in zip(conll_sentence[1:],outputFFlayer):
                            relvec=self.rellookup[np.argmax(pred.value())]
                            entry.vec = concatenate([entry.vec2, dynet.dropout(relvec, 0.33)])

                        for entry in conll_sentence:
                            entry.ner2_lstms = [entry.vec, entry.vec]

                        slstm_forward = self.sner_builders[0].initial_state()
                        slstm_backward = self.sner_builders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            lstm_forward = slstm_forward.add_input(entry.vec)
                            lstm_backward = slstm_backward.add_input(rentry.vec)

                            entry.ner2_lstms[1] = lstm_forward.output()
                            rentry.ner2_lstms[0] = lstm_backward.output()

                        for entry in conll_sentence:
                            entry.ner2_vec = concatenate(entry.ner2_lstms)

                        sblstm_forward = self.sner_bbuilders[0].initial_state()
                        sblstm_backward = self.sner_bbuilders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            blstm_forward = sblstm_forward.add_input(entry.ner2_vec)
                            blstm_backward = sblstm_backward.add_input(rentry.ner2_vec)

                            entry.ner2_lstms[1] = blstm_forward.output()
                            rentry.ner2_lstms[0] = blstm_backward.output()

                        concat_layer = [dynet.dropout(concatenate(entry.ner2_lstms), 0.33) for entry in conll_sentence]
                        outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                        posIDs = [self.pos.get(entry.pos) for entry in conll_sentence]
                        gold_score=self.pick_gold_score(outputFFlayer,posIDs)
                        ner2Errs=(self.forward_score(outputFFlayer)-gold_score)

                    if iSentence<5:
                        print("ner and dep loss")
                        if ner2Errs!=0:
                            print(ner2Errs.value())
                        else:
                            print(0)
                        if dep_epoch!=0:
                            print(esum(lerrs).value())
                        else:
                            print(0)

                eerrors += e
                if e > 0:
                    loss = [(exprs[h][i] - exprs[g][i]) for i, (h, g) in enumerate(zip(heads, gold)) if h != g]  # * (1.0/float(e))
                    eloss += (e)
                    mloss += (e)
                    errs.extend(loss)

                etotal += len(conll_sentence)

                if iSentence % 1 == 0:
                    if len(errs) > 0 or len(lerrs) > 0 or posErrs > 0 or len(postrErrs) > 0 or ner2Errs > 0 or len(nertr2Errs)>0:
                        eerrs=0
                        if len(errs + lerrs+ postrErrs  + nertr2Errs)>0:
                            eerrs = esum(errs + lerrs+postrErrs  + nertr2Errs)
                        eerrs += (posErrs + ner2Errs)
                        #print(eerrs.value())
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        e=0
                        lerrs = []
                        posErrs = []
                        postrErrs= []
                        ner2Errs = []
                        nertr2Errs  = []
                        posErrs=0
                        ner2Errs= 0

                    renew_cg()

        print "Loss: %.4f" % (mloss / iSentence)

