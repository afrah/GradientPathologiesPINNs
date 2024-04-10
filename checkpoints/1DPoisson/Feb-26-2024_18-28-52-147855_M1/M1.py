

import tensorflow as tf
import numpy as np
import timeit
from scipy.interpolate import griddata
import seaborn as sns
import matplotlib.pyplot as plt
import os

import sys
import logging

import os.path
import shutil
from datetime import datetime

import pickle
class Possion_M1():
    def __init__(self, layers, X_u, Y_u, X_r, Y_r ,mode , PATH ,  sess):


        self.mode = mode

        self.dirname, logpath = self.make_output_dir(PATH)
        self.logger = self.get_logger(logpath)     

        self.mu_X, self.sigma_X = X_r.mean(0), X_r.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]

        # Normalize
        self.X_u = (X_u - self.mu_X) / self.sigma_X
        self.Y_u = Y_u
        self.X_r = (X_r - self.mu_X) / self.sigma_X
        self.Y_r = Y_r

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
            
        # Define the size of the Kernel
        self.kernel_size = X_u.shape[0]
        # Define Tensorflow session
        self.sess = sess# tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.lam_bc =  np.array(1.0)
        self.lam_res =  np.array(1.0)
        self.lam_res_tf = tf.placeholder(tf.float32, shape=self.lam_res.shape)
        self.lam_bc_tf = tf.placeholder(tf.float32, shape=self.lam_bc.shape)

        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        
        self.x_u_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))
        self.x_r_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))

        self.lambda1 = tf.constant(1.0 , dtype=tf.float32)
        self.lambda2 = tf.constant(500.0 , dtype=tf.float32 )

        # Evaluate predictions
        self.u_bc_pred = self.net_u(self.x_bc_tf)

        self.u_pred = self.net_u(self.x_u_tf)
        self.r_pred = self.net_r(self.x_r_tf)
        
        self.u_ntk_pred = self.net_u(self.x_u_ntk_tf)
        self.r_ntk_pred = self.net_r(self.x_r_ntk_tf)
     
        # Boundary loss
        self.loss_bcs = tf.reduce_mean(tf.square(self.u_bc_pred - self.u_bc_tf))

        # Residual loss        
        self.loss_res =  tf.reduce_mean(tf.square(self.r_tf - self.r_pred))
        
        # Total loss
        self.loss = self.lam_res_tf * self.loss_res + self.lam_bc_tf *  self.loss_bcs

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-5
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        # To compute NTK, it is better to use SGD optimizer
        # since the corresponding gradient flow is not exactly same.
        # self.train_op = tf.train.GradientDescentOptimizer(starter_learning_rate).minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)


        
        # Logger
        # Loss logger
        self.loss_bcs_log = []
        self.loss_res_log = []

        # NTK logger 
        self.K_uu_log = []
        self.K_rr_log = []
        self.K_ur_log = []
        
        # Weights logger 
        self.weights_log = []
        self.biases_log = []
       # Gradients Storage



        # Generate dicts for gradients storage
        self.dict_gradients_res_layers = self.generate_grad_dict()
        self.dict_gradients_bc_layers = self.generate_grad_dict()

        self.grad_res = []
        self.grad_bc = []
        self.grad_res_list = []
        self.grad_bc_list = []

        for i in range(len(self.layers)-1):
            self.grad_res.append(tf.gradients(self.loss_res, self.weights[i])[0])
            self.grad_bc.append(tf.gradients(self.loss_bcs, self.weights[i])[0])


        self.adaptive_constant_bcs_log = []

        self.mean_grad_res_list = []
        self.mean_grad_bcs_list = []
    
        self.mean_grad_res_list_log = []
        self.mean_grad_bcs_list_log = []

        for i in range( len(self.layers) -1):
            self.mean_grad_res_list.append(tf.math.reduce_mean(tf.abs(self.grad_res[i]))) 
            self.mean_grad_bcs_list.append(tf.math.reduce_mean(tf.abs(self.grad_bc[i])))
        
        self.mean_grad_res = tf.math.reduce_mean(tf.stack(self.mean_grad_res_list))
        self.mean_grad_bcs = tf.math.reduce_mean(tf.stack(self.mean_grad_bcs_list))
    

        # for i in range(1 , len(self.layers) - 2):
        #     self.grad_res_list.append(tf.reduce_mean(tf.abs(self.grad_bc[i])))
        #     self.grad_bc_list.append(tf.reduce_mean(tf.abs(self.grad_res[i])))

        self.loss_tensor_list = [self.loss ,  self.loss_res,  self.loss_bcs] 
        self.loss_list = ["total loss" , "loss_res" , "loss_bcs"] 

        self.epoch_loss = dict.fromkeys(self.loss_list, 0)
        self.loss_history = dict((loss, []) for loss in self.loss_list)
        

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()

        self.sess.run(init)
        

###############################################################################################################

    def generate_grad_dict(self):
        num = len(self.layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict

    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)
    
    # NTK initialization
    def NTK_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        std = 1. / np.sqrt(in_dim)
        return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * std,
                           dtype=tf.float32)

     # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.NTK_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.random.normal([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Evaluates the PDE solution
    def net_u(self, x):
        u = self.forward_pass(x)
        return u

    # Forward pass for the residual
    def net_r(self, x):
        u = self.net_u(x)

        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x

        res_u = u_xx
        return res_u
    
    # Trains the model by minimizing the MSE loss
    def trainmb(self, nIter=10000, batch_size=128, log_NTK=True, log_weights=True):


        itValues = [1,100,1000,39999]
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_bc_tf: self.X_u, self.u_bc_tf: self.Y_u,
                       self.x_u_tf: self.X_u, self.x_r_tf: self.X_r,
                       self.r_tf: self.Y_r,
                       self.lam_bc_tf : self.lam_bc,
                       self.lam_res_tf : self.lam_res
                       }
        
            # Run the Tensorflow session to minimize the loss

            # print(self.lam_bc_tf.shape)
            _, batch_losses = self.sess.run([self.train_op, self.loss_tensor_list] ,tf_dict)
            self.assign_batch_losses(batch_losses)
            for key in self.loss_history:
                self.loss_history[key].append(self.epoch_loss[key])
            
            # self.print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                [loss ,  loss_res,  loss_bcs]  = batch_losses


                self.print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_res: %.3e ,Time: %.2f' %  (it, loss, loss_bcs, loss_res, elapsed))
                
            # provide x, x' for NTK
            if it % 100 == 0:
                mean_grad_bcs , mean_grad_res = self.sess.run([self.mean_grad_bcs , self.mean_grad_res],  tf_dict)

                self.lam_bc = 1.0 # #mean_grad_res / (mean_grad_bcs  +1e-8 )
                self.lam_res =  1.0 #mean_grad_res

                self.print("mean_grad_bcs: " ,  mean_grad_bcs)    
                self.print("mean_grad_res: " , mean_grad_res)    
                self.mean_grad_bcs_list_log.append(mean_grad_bcs)
                self.mean_grad_res_list_log.append(mean_grad_res)

                self.adaptive_constant_bcs_log.append(self.lam_bc)

                start_time = timeit.default_timer()

            if it in itValues:
                    self.plot_layerLoss(tf_dict , it)
                    self.print("Gradients information stored ...")

            sys.stdout.flush()
  
    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_r_tf: X_star}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star
 ############################################################

    def assign_batch_losses(self, batch_losses):
        for loss_values, key in zip(batch_losses, self.epoch_loss):
            self.epoch_loss[key] = loss_values

  ############################################################
###################################################################################################################

    def plot_layerLoss(self , tf_dict , epoch):
        ## Gradients #
        num_layers = len(self.layers)
        for i in range(num_layers - 1):
            grad_res, grad_bc  = self.sess.run([ self.grad_res[i],self.grad_bc[i]], feed_dict=tf_dict)

            # save gradients of loss_r and loss_u
            self.dict_gradients_res_layers['layer_' + str(i + 1)].append(grad_res.flatten())
            self.dict_gradients_bc_layers['layer_' + str(i + 1)].append(grad_bc.flatten())

        num_hidden_layers = num_layers -1
        cnt = 1
        fig = plt.figure(4, figsize=(13, 4))
        for j in range(num_hidden_layers):
            ax = plt.subplot(1, num_hidden_layers, cnt)
            ax.set_title('Layer {}'.format(j + 1))
            ax.set_yscale('symlog')
            gradients_res = self.dict_gradients_res_layers['layer_' + str(j + 1)][-1]
            gradients_bc = self.dict_gradients_bc_layers['layer_' + str(j + 1)][-1]

            sns.distplot(gradients_res, hist=False,kde_kws={"shade": False},norm_hist=True,  label=r'$\nabla_\theta \mathcal{L}_r$')

            sns.distplot(gradients_bc, hist=False,kde_kws={"shade": False},norm_hist=True,   label=r'$\nabla_\theta \mathcal{L}_{u_{bc}}$')

            #ax.get_legend().remove()
            ax.set_xlim([-1.0, 1.0])
            #ax.set_ylim([0, 150])
            cnt += 1
        handles, labels = ax.get_legend_handles_labels()

        fig.legend(handles, labels, loc="center",  bbox_to_anchor=(0.5, -0.03),borderaxespad=0,bbox_transform=fig.transFigure, ncol=2)
        text = 'layerLoss_epoch' + str(epoch) +'.png'
        plt.savefig(os.path.join(self.dirname,text) , bbox_inches='tight')
        plt.close("all")


    def get_logger(self, logpath):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)        
        sh.setFormatter(logging.Formatter('%(message)s'))
        fh = logging.FileHandler(logpath)
        logger.addHandler(sh)
        logger.addHandler(fh)
        return logger
    
    def print(self, *args):
        for word in args:
            if len(args) == 1:
                self.logger.info(word)
            elif word != args[-1]:
                for handler in self.logger.handlers:
                    handler.terminator = ""
                if type(word) == float or type(word) == np.float64 or type(word) == np.float32: 
                    self.logger.info("%.4e" % (word))
                else:
                    self.logger.info(word)
            else:
                for handler in self.logger.handlers:
                    handler.terminator = "\n"
                if type(word) == float or type(word) == np.float64 or type(word) == np.float32:
                    self.logger.info("%.4e" % (word))
                else:
                    self.logger.info(word)


    def plot_loss_history(self , path):

        fig, ax = plt.subplots()
        fig.set_size_inches([15,8])
        for key in self.loss_history:
            self.print("Final loss %s: %e" % (key, self.loss_history[key][-1]))
            ax.semilogy(self.loss_history[key], label=key)
        ax.set_xlabel("epochs", fontsize=15)
        ax.set_ylabel("loss", fontsize=15)
        ax.tick_params(labelsize=15)
        ax.legend()
        plt.savefig(path)
            #plt.show()
       #######################
    def save_NN(self):

        uv_weights = self.sess.run(self.weights)
        uv_biases = self.sess.run(self.biases)

        with open(os.path.join(self.dirname,'model.pickle'), 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            self.print("Save uv NN parameters successfully in %s ..." , self.dirname)

        # with open(os.path.join(self.dirname,'loss_history_BFS.pickle'), 'wb') as f:
        #     pickle.dump(self.loss_rec, f)
        with open(os.path.join(self.dirname,'loss_history_BFS.png'), 'wb') as f:
            self.plot_loss_history(f)

        return self.dirname
    
    def plt_prediction(self , X_star , u_star , u_pred):
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        plt.plot(X_star, u_star, label='Exact')
        plt.plot(X_star, u_pred, '--', label='Predicted')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend(loc='upper right')

        plt.subplot(1,2,2)
        plt.plot(X_star, np.abs(u_star - u_pred), label='Error')
        plt.yscale('log')
        plt.xlabel('$x$')
        plt.ylabel('Point-wise error')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirname,"prediction.png"))
        plt.close("all")


    def plot_grad(self):

        fig, ax = plt.subplots()
        fig.set_size_inches([15,8])
        ax.semilogy(self.mean_grad_res_list_log, label=r'$\bar{\nabla_\theta \mathcal{L}_{u_{phy}}}$')
        ax.semilogy(self.mean_grad_bcs_list_log, label=r'$\bar{\nabla_\theta \mathcal{L}_{u_{bc}}}$')

        ax.set_xlabel("epochs", fontsize=15)
        ax.set_ylabel("loss", fontsize=15)
        ax.tick_params(labelsize=15)
        ax.legend()
        path = os.path.join(self.dirname,'grad_history.png')
        plt.savefig(path)
        plt.close("all" , )

    def plot_lambda(self ):

        fontsize = 17
        fig, ax = plt.subplots()
        fig.set_size_inches([16,8])
        ax.semilogy(self.mean_grad_bcs_list_log, label=r'$\bar{\nabla_\theta {u_{bc}}}$' , color = 'tab:green')
        ax.semilogy(self.mean_grad_res_list_log, label=r'$Max{\nabla_\theta {u_{phy}}}$' , color = 'tab:red')
        ax.set_xlabel("epochs", fontsize=fontsize)
        ax.set_ylabel(r'$\bar{\nabla_\theta {u}}$', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.legend(loc='center left', bbox_to_anchor=(-0.25, 0.5))

        ax2 = ax.twinx() 

        # fig, ax = plt.subplots()
        # fig.set_size_inches([15,8])
    
        ax2.semilogy(self.adaptive_constant_bcs_log, label=r'$\bar{\lambda_{bc}}$'  ,  linestyle='dashed' , color = 'tab:green') 
        ax2.set_xlabel("epochs", fontsize=fontsize)
        ax2.set_ylabel(r'$\bar{\lambda}$', fontsize=fontsize)
        ax2.tick_params(labelsize=fontsize)
        ax2.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))

        plt.tight_layout()

        path = os.path.join(self.dirname,'grad_history.png')
        plt.savefig(path)
        plt.close("all" , )


    def make_output_dir(self , PATH):
        import shutil
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        dirname = os.path.join(PATH, datetime.now().strftime("%b-%d-%Y_%H-%M-%S-%f_") + self.mode)
        os.mkdir(dirname)
        text = 'output.log'
        logpath = os.path.join(dirname, text)
        shutil.copyfile('/home/vlq26735/code/PhD/GradientPathologiesPINNs/src/cases/Poisson/M1.py', os.path.join(dirname, 'M1.py'))
        return dirname, logpath
    
