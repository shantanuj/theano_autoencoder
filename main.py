'''Dependencies'''
import os
import sys
import timeit
import matplotlib
matplotlib.use("Agg") #for server running
import time
import numpy as np
from load import mnist
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pylab
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA
from Stacked_denoising_autoencoder import Stacked_denoising_autoencoder

'''Hyperparameters'''

pretraining_batch_size = 128
num_pretraining_epochs = 25
pretraining_lr = 0.1
corruption_levels = [0.1,0.1,0.1]
hidden_layer_dims = [900,625,400]
primary_submission= False
#####################################################
pretraining_mom = 0.1
penalty_parameter = 0.5
sparsity_constraint = 0.05
with_momentum = False
################################################
classification_training_epochs = 25 
classification_lr = 0.1
classification_act = "sigmoid"

auto_enc_hyperparameters= [pretraining_lr, pretraining_batch_size, num_pretraining_epochs, corruption_levels, hidden_layer_dims]
classification_hyperparameters = [classification_training_epochs, classification_lr]

hyperparameter_string=  "Alpha:{}; Batch_size:{}; Num_epochs:{}; Corruption:{} ; Hidden_layers:{}".format(str(auto_enc_hyperparameters[0]), str(auto_enc_hyperparameters[1]), str(auto_enc_hyperparameters[2]), str(auto_enc_hyperparameters[3][0]), str(auto_enc_hyperparameters[4]))
if(primary_submission):
    hyperparameter_string = "Assignment_params"+hyperparameter_string

print("Hyperparameters:")
print(hyperparameter_string)
os_dir = "./plots_part2/q1/hyperparameters:{}".format(hyperparameter_string)
if(os.path.exists(os_dir)):
    os_dir = "./plots_part2/q1/2ndRUNhyperparameters:{}".format(hyperparameter_string)
    os.makedirs(os_dir)
else:
    os.makedirs(os_dir)
plots_dir = os_dir



"""Theano dataset loading """
def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    
    
    
"""Loading dataset"""

print("Loading dataset")
trX, teX, trY, teY = mnist(onehot=False)
#trX, trY = trX[:12000], trY[:12000]
#teX, teY = teX[:2000], teY[:2000]

trX_tensor, trY_tensor = shared_dataset((trX,trY))
teX_tensor, teY_tensor = shared_dataset((teX, teY))
train_tuple = (trX_tensor, trY_tensor)
test_tuple = (teX_tensor, teY_tensor)
n_train_batches = trX_tensor.get_value(borrow=True).shape[0]
batch_size = pretraining_batch_size
n_train_batches //= pretraining_batch_size



        
        
   
"""Building stacked autoencoder model"""
numpy_rng = np.random.RandomState(123)
print("Building model")
sda = Stacked_denoising_autoencoder(numpy_rng =numpy_rng, n_ins = 28*28, hidden_layer_dims=hidden_layer_dims, n_outs= 10, corruption_levels = corruption_levels, activation_func= T.nnet.sigmoid)


'''Plot functions'''

def plot_autoencoder_costs(epoch_cost, layer_no):
    autoencoder_layer_str = "dae_layer{}".format(str(layer_no))
    label = "DAE layer {} training costs".format(layer_no)
    title = "DAE layer {}; Min cost:{}".format(layer_no, str(np.min(epoch_cost)))
    #pylab.figure(figsize=(15,8))
    pylab.figure()
    pylab.plot(range(len(epoch_cost)), epoch_cost, label = label)
    pylab.xlabel('Epochs')
    pylab.ylabel('Cross entropy cost')
    pylab.title(title)
    pylab.legend()
    
    pylab.savefig(os.path.join(plots_dir,autoencoder_layer_str+"Costs.png"), dpi = 300)
    pylab.show()

    
def plot_autoencoder_weights(weights, layer_no):
    autoencoder_layer_str = "dae_layer{}".format(str(layer_no))
    title = "Weights for DAE layer {}".format(layer_no)
    #pylab.figure(figsize=(15,8))
    pylab.figure()
    pylab.gray()
    shape_of_figure = int(np.sqrt(weights.shape[0]))
    for i in range(100):
        pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(weights[:,i].reshape(shape_of_figure,shape_of_figure))
    pylab.savefig(os.path.join(plots_dir, autoencoder_layer_str+"Weights.png"), dpi = 300)
    pylab.show()
    
def plot_matrix_as_image(matrix, output_name):
    pylab.figure()
    pylab.gray()
    shape_of_figure = int(np.sqrt(matrix.shape[1])) #so if 10*784 is passed
    for i in range(100):
        pylab.subplot(10,10,i+1); pylab.axis('off'); pylab.imshow(matrix[i,:].reshape(shape_of_figure, shape_of_figure))
    pylab.savefig(os.path.join(plots_dir, output_name+'.png'), dpi = 300)
    pylab.show()
    


def plot_classification_results(epoch_cost,epoch_acc):
    label = "Classification training cost"
    title = "Classification training cost"
    pylab.figure()
    pylab.plot(range(len(epoch_cost)), epoch_cost, label = label)
    pylab.xlabel('Epochs')
    pylab.ylabel('Cross entropy cost')
    pylab.title(title+"; Min cost: {}".format(np.min(epoch_cost)))
    pylab.legend()
    pylab.savefig(os.path.join(plots_dir,title+".png"), dpi = 300)
    pylab.show()
    
    label = "Classification testing accuracy"
    title = "Classification testing accuracy"
    pylab.figure()
    pylab.plot(range(len(epoch_acc)), epoch_acc, label = label)
    pylab.xlabel('Epochs')
    pylab.ylabel('Classification accuracy')
    pylab.title(title+"; Max accuracy: {}".format(np.max(epoch_acc)))
    pylab.legend()
    pylab.savefig(os.path.join(plots_dir,title+".png"), dpi = 300)
    pylab.show()



    
"""Training autoencoder"""

pretraining_fns = sda.pretraining_da_fns(train_set_x=trX_tensor, batch_size=batch_size)


print('Training stacked denoising autoencoder layer by layer (unsupervised)')
start_time = time.time()
    ## Pre-train layer-wise

for i in range(sda.n_layers):
        # go through pretraining epochs
    epoch_cost = []
    for epoch in range(num_pretraining_epochs):
        # go through the training set
        c = []
        for batch_index in range(n_train_batches):
            c.append(pretraining_fns[i](index=batch_index, corruption=corruption_levels[i],lr=pretraining_lr))
        print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c, dtype='float64')))
        epoch_cost.append(np.mean(c, dtype='float64'))
    plot_autoencoder_costs(epoch_cost, i)
    plot_autoencoder_weights(sda.dA_layers[i].W.get_value(),i)
    #plot_autoencoder_weights()

end_time = time.time()

print("Unsupervised training took {} seconds for {} epochs".format(end_time-start_time, num_pretraining_epochs))


print("Producing plots for autoencoder with testing set")
"""Testing for autoencoder"""

def sigmoid(z):
    return 1/(1+np.exp(-z))

def get_test_autoencoder_representations(test_set_100, num_layers, activation = sigmoid):
    
    layer_params = []
    for i in range(num_layers): #go through each autoencoder
        layer = sda.activation_layers[i]
        if(i==0):
            layer_params.append((layer.W.get_value(), layer.b.get_value(), sda.dA_layers[i].b_prime.get_value()))
        else:
            layer_params.append((layer.W.get_value(), layer.b.get_value()))
    plot_matrix_as_image(test_set_100,"inputs")
    activation = test_set_100
    for i in range(num_layers):
        #first activation
        activation = sigmoid(np.dot(activation,layer_params[i][0])+layer_params[i][1])
        if(i==0):
            reconstructed_image = sigmoid(np.dot(activation,layer_params[i][0].transpose())+layer_params[i][-1])

        plot_matrix_as_image(activation, "layer{}_activation".format(i))
    
    plot_matrix_as_image(reconstructed_image, "reconstructed_images")
get_test_autoencoder_representations(teX[:100,:], len(sda.activation_layers))
       



print('Training classification layer')
train_fn, test_model = sda.classification_training(train_tuple=train_tuple, test_tuple=test_tuple, batch_size=batch_size, learning_rate=classification_lr)

training_loss_list = []
test_accuracy_list = []
epoch=0
while(epoch<classification_training_epochs):
    epoch=epoch+1
    minibatch_costs = []
    for minibatch_index in range(n_train_batches):
        minibatch_cost = train_fn(minibatch_index)
        minibatch_costs.append(minibatch_cost)
    
    training_loss_list.append(np.mean(minibatch_costs))
    test_accuracy_list.append(1-np.mean(test_model()))
    print("Epoch {} avg training loss: {} avg accuracy: {}".format(epoch, training_loss_list[-1], test_accuracy_list[-1]))
plot_classification_results(training_loss_list,test_accuracy_list)   
    
print("Finished training classification layer")
print("Output directory: {}".format(plots_dir))
