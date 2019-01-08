"""
https://raw.githubusercontent.com/bplank/bilstm-aux/master/src/lib/mnnl.py
"""
import dynet
import numpy as np

import sys

## NN classes
class SequencePredictor:
    def __init__(self):
        pass
    
    def predict_sequence(self, inputs):
        raise NotImplementedError("SequencePredictor predict_sequence: Not Implemented")

class FFSequencePredictor(SequencePredictor):
    def __init__(self, network_builder):
        self.network_builder = network_builder
        
    def predict_sequence(self, inputs):
        return [self.network_builder(x) for x in inputs]


    ##this returns scores taking into account the transition probs
    ## receives the input scores which is the output of predict_sequence
    ## not finalized yet
    ## not working properly
    def predict_transition_sequence(self,scores,transitions):
        tagnum=transitions.shape()[0]
        bestparents=[[-1 for t in range(tagnum)]]
        bestscores=[dynet.cmult(transitions[tagnum-2][:-2],scores[0])]
        for i, wscore in enumerate(scores[1:]):
            parent_values=[o.value() for o in bestscores[i]]
            best_parent_id=np.argmax(parent_values)
            last_scores=[t*s for t, s in zip(transitions[best_parent_id],wscore)]
            last2=dynet.softmax(dynet.cmult(transitions[best_parent_id][:-2],wscore))
            bestscores.append(last2)
        return bestscores

    ##find the best ner tag sequence by taking into account tag transition probabilities
    ## now does not work for too long sentences
    def viterbi_sequence(self,tagprobs,transitionprobs):
        all_parentids=[]
        bestscores=[(tagp+trprob).value() for tagp , trprob in zip(tagprobs[1],transitionprobs[len(transitionprobs)-1])]
        for tagprob in tagprobs[2:]:
            bestparents=[]
            best_t=[]
            for i, tagp in enumerate(tagprob):
                tagscore=[(tagp.value() +trprob.value() +bestscore)for trprob,bestscore in zip (transitionprobs[i],bestscores)]
                argmax=np.argmax(tagscore)
                best_t.append(tagscore[argmax])
                bestparents.append(argmax)
            bestscores=best_t
            all_parentids.append(bestparents)
        bestscores=[score+trprob.value() for score , trprob in zip(bestscores,transitionprobs[len(transitionprobs)-1])]
        final_best_id=np.argmax(bestscores)
        parent_path=[final_best_id]
        best_tag_id=final_best_id
        for parents in reversed(all_parentids):
            parent_path.append(parents[best_tag_id])
            best_tag_id=parents[best_tag_id]
        return reversed(parent_path),bestscores
    ##given parentids and bestscores for each word
    ##returns the best tag sequence for ner
    def decode_viterbi_seq(self,parentids,bestscores):
        tagids=[]
        lasttag=np.argmax(bestscores[-1])
        for layer in reversed(parentids):
            tagids.append(lasttag)
            lasttag=layer[lasttag]
        return [tagid for tagid in reversed(tagids)]
class RNNSequencePredictor(SequencePredictor):
    def __init__(self, rnn_builder):
        """
        rnn_builder: a LSTMBuilder/SimpleRNNBuilder or GRU builder object
        """
        self.builder = rnn_builder
        
    def predict_sequence(self, inputs):
        s_init = self.builder.initial_state()
        return s_init.transduce(inputs)

class BiRNNSequencePredictor(SequencePredictor):
    """ a bidirectional RNN (LSTM/GRU) """
    def __init__(self, f_builder, b_builder):
        self.f_builder = f_builder
        self.b_builder = b_builder

    def predict_sequence(self, f_inputs, b_inputs):
        f_init = self.f_builder.initial_state()
        b_init = self.b_builder.initial_state()
        forward_sequence = f_init.transduce(f_inputs)
        backward_sequence = b_init.transduce(reversed(b_inputs))
        return forward_sequence, backward_sequence 
        

class Layer:
    """ Class for affine layer transformation or two-layer MLP """
    def __init__(self, model, in_dim, output_dim, activation=dynet.tanh, mlp=0, mlp_activation=dynet.rectify):
        # if mlp > 0, add a hidden layer of that dimension
        self.act = activation
        self.mlp = mlp
        if mlp:
            print('>>> use mlp with dim {} ({})<<<'.format(mlp, mlp_activation))
            mlp_dim = mlp
            self.mlp_activation = mlp_activation
            self.W_mlp = model.add_parameters((mlp_dim, in_dim))
            self.b_mlp = model.add_parameters((mlp_dim))
        else:
            mlp_dim = in_dim
        self.W = model.add_parameters((output_dim, mlp_dim))
        self.b = model.add_parameters((output_dim))
        
    def __call__(self, x):
        if self.mlp:
            W_mlp = dynet.parameter(self.W_mlp)
            b_mlp = dynet.parameter(self.b_mlp)
            act = self.mlp_activation
            x_in = act(W_mlp * x + b_mlp)
        else:
            x_in = x
        # from params to expressions
        W = dynet.parameter(self.W)
        b = dynet.parameter(self.b)
        if self.act!="idt":
            return self.act(W*x_in + b)
        else:
            return W*x_in+b
