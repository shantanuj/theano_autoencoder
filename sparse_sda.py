
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pylab
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from sparse_dae import dA
     
"""Stacked denoising autoencoder class"""
class Stacked_denoising_autoencoder(object):
    '''This class does the following:
    1) It takes as its input:
     a) The number of hidden neurons in each layer as a list. 
     b) The corruption levels to corrupt input of each autoencoder
     c) The final layer/output dimensions/number of classes
     c) Optional: Whether weights are to be tied or not
     d) For training, it takes as input the entire training set, batch size, and learning rate; However, this can be called as a function and
     /is not required during object initialization
     e) Additionally, to mantain reproducibility of results, it inputs the numpy random state and 
    
    2) Running of unsupervised code:
    Given a training dataset with no labels, it performs unsupervised learning by finding the best weights and biases to map the input to a lower level representation
    Since, the input to the second autoencoder is the feature learned by the previous autoencoder, the encoding weights and biases of the first encoder are preserved
    
    The structure of autoencoders is mantained by simply storing each autoencoder object in a list
    An autoencoder object primarily purpose is to get the best weights and biases to encode the input. 'best' means most useful/insightful representation of input --> extracting most number of unique features in given dimensions 
    
    However, to pass the input from previous autoencoder, the activation/hidden representation produced by the encoder is required as input
    This activation is equivalent to a sigmoid layer activation. Hence, a list with all sigmoid activations is mantained with weights and biases tied to the autoencoder.
    
    This reduces to the clear functioning:
    1) Store a list of autoencoder objects and a list of corresponding (weight tied) sigmoid activations.
    2) PRETRAINING: For each autoencoder, train its weights and biases with input as either--> if first autencoder, then training set; else the sigmoid activation/hidden rep from previous sigmoid layer
    3) CLASSIFCATION TRAINING: Once all levels of autencoders are trained, use the learned weights and biases to intialize a supervised network 
    4) Train this supervised network normally
    To produce 
    
    
    To preserve memory, a generator function is used to yield batches 
    to be trained in an unsupervised manner
    2) Running of stacked
    
    
    '''
    
    def __init__(self, numpy_rng, theano_rng = None, hidden_layer_dims = [500]*3, n_ins = 784, n_outs = 10, corruption_levels = [0.1]*3, activation_func = T.nnet.sigmoid):
        self.activation_layers = []  #this stores the layers(set to sigmoid) objects to produce activations from each autencoder trained weights
        self.dA_layers = [] #this stores the autoencoder objects
        self.params = [] #stores the learned weights from autencoder/tied with activation layer
        self.n_layers = len(hidden_layer_dims)
        self.x = T.matrix('x') #store input dataset provided during pretraining
        self.y = T.ivector('y') #store labels during classification training
        if(not theano_rng): #make theano random number generator to ensure consistency of results/starting weights
            theano_rng  = RandomStreams(numpy_rng.randint(2**30))
        
        #Fill up the activation layers and autencoder layers
        for i in range(self.n_layers):
            if(i==0): #first hidden layer
                layer_input_dims = n_ins 
                layer_input = self.x 
            else:
                layer_input_dims = hidden_layer_dims[i-1]
                layer_input = self.activation_layers[i-1].output
            
            #instantiate a hidden layer which computes the hidden representation from parameters tied with autoencoder
            activation_layer = HiddenLayer(rng= numpy_rng, input = layer_input, n_in = layer_input_dims, n_out = hidden_layer_dims[i], activation = T.nnet.sigmoid)
            self.activation_layers.append(activation_layer)
            self.params.extend(activation_layer.params) #we do not need the autencoder output biases and weights (if not constrained)  
            #instantiate the denoising autencoder which contains all relevant parameters, etc
            d_autoencoder = dA(numpy_rng = numpy_rng, theano_rng = theano_rng, input = layer_input, n_visible= layer_input_dims, n_hidden = hidden_layer_dims[i], W = activation_layer.W, bhid = activation_layer.b)
            '''Experiment with unrestricted weights'''
            #d_autoencoder = dA(untied_weights = True, numpy_rng = numpy_rng, theano_rng = theano_rng, input = layer_input, n_visible= layer_input_dims, n_hidden = hidden_layer_dims[i], W = activation_layer.W, bhid = activation_layer.b)
            self.dA_layers.append(d_autoencoder)
        
        #the classification layer with softmax output
        self.classificationLayer = LogisticRegression(input = self.activation_layers[-1].output, n_in = hidden_layer_dims[-1], n_out = n_outs)
        
        #update parameters with last layer parameters
        self.params.extend(self.classificationLayer.params)
#############################################################################################
        '''To edit for second part'''
        #compute classification cost
        self.classification_cost = self.classificationLayer.negative_log_likelihood(self.y)
#############################################################################################
        #compute classification error
        self.errors = self.classificationLayer.errors(self.y)
        
    def pretraining_da_fns(self, train_set_x, batch_size):
        """Returns list of pretraining theano functions which can be called later for pretraining
        Performs pretraining layer by layer for each autoencoder"""
        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')
        sparsity_parameter = T.scalar('sparsity')
        penalty_parameter = T.scalar('penalty')
        momentum_parameter = T.scalar('momentum')
        batch_begin = index*batch_size
        batch_end = batch_begin + batch_size
        pretrain_fns = []
        for dn_autoencoder in self.dA_layers:
            cost, updates = dn_autoencoder.get_cost_updates(corruption_level,
                                                learning_rate, sparsity_parameter, penalty_parameter, momentum_parameter)
            
            ######################################
            #Line 172 of main needs to be edited
            #See if line 119 of this one makes a difference
            #####################################
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(corruption_level, value=0.1),
                    theano.In(learning_rate, value=0.1),
                    theano.In(sparsity_parameter, value =0.05),
                    theano.In(penalty_parameter, value= 0.5),
                    theano.In(momentum_parameter, value= 0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
        return pretrain_fns
    
    def classification_training(self, train_tuple, test_tuple, batch_size, learning_rate):
        
        index = T.lscalar('index')
        grad_params = T.grad(self.classification_cost, self.params) #computes gradient of parameters (weights and biases) with respect to classification cost
#############################################################################################
        '''Update of parameters here to change for part 3'''
##############################################################################################
        updates = [(param, param - grad_params*learning_rate) for param, grad_params in zip(self.params, grad_params)]
        
        (train_set_x, train_set_y) = train_tuple
        (test_set_x, test_set_y) = test_tuple
        
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size
        
        #Make functions for training and testing
        train_fn = theano.function(inputs=[index], outputs=self.classification_cost, updates=updates, givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )
        test_fn = theano.function(inputs = [index], outputs= self.errors, givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )
        
        def test_score():
            return [test_fn(i) for i in range(n_test_batches)]
        
        
        return train_fn, test_score
