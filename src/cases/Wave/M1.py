import tensorflow as tf
# from Compute_Jacobian import jacobian # Please download 'Compute_Jacobian.py' in the repository 
import numpy as np
import timeit
from scipy.interpolate import griddata
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os



os.environ["KMP_WARNINGS"] = "FALSE" 

import timeit

import sys

import scipy.io


import logging

import os.path
from datetime import datetime
import pickle



def operator(u, t, x, c, sigma_t=1.0, sigma_x=1.0):
    u_t = tf.gradients(u, t)[0] / sigma_t
    u_x = tf.gradients(u, x)[0] / sigma_x
    u_tt = tf.gradients(u_t, t)[0] / sigma_t
    u_xx = tf.gradients(u_x, x)[0] / sigma_x
    residual = u_tt - c**2 * u_xx
    return residual

class PINN:
    # Initialize the class
    def __init__(self, layers, ics_sampler, bcs_sampler, res_sampler, c , mode ,  starter_learning_rate , PATH , sess ):
        # Normalization 




        self.mode = mode

        self.dirname, logpath = self.make_output_dir(PATH)
        self.logger = self.get_logger(logpath)     

        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        self.sess = sess
        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # weights
        self.adaptive_constant_bcs_val = np.array(1.0)
        self.adaptive_constant_ics_val = np.array(1.0)
        self.adaptive_constant_res_val = np.array(1.0)
        self.rate = 0.9

        # Wave constant
        self.c = tf.constant(c, dtype=tf.float32)
        
        # self.kernel_size = kernel_size # Size of the NTK matrix

        # Define Tensorflow session
        self.sess = sess

        # Define placeholders and computational graph
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        
        self.adaptive_constant_bcs_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_bcs_val.shape)
        self.adaptive_constant_ics_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_ics_val.shape)
        self.adaptive_constant_res_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_res_val.shape)
        

        # Evaluate predictions
        self.u_ics_pred = self.net_u(self.t_ics_tf, self.x_ics_tf)
        self.u_t_ics_pred = self.net_u_t(self.t_ics_tf, self.x_ics_tf)
        self.u_bc1_pred = self.net_u(self.t_bc1_tf, self.x_bc1_tf)
        self.u_bc2_pred = self.net_u(self.t_bc2_tf, self.x_bc2_tf)

        self.u_pred = self.net_u(self.t_u_tf, self.x_u_tf)
        self.r_pred = self.net_r(self.t_r_tf, self.x_r_tf)
        

        # Boundary loss and Initial loss
        self.loss_ics_u = tf.reduce_mean(tf.square(self.u_ics_tf - self.u_ics_pred))
        self.loss_ics_u_t = tf.reduce_mean(tf.square(self.u_t_ics_pred))
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_pred))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_pred))

        self.loss_bcs = self.loss_ics_u + self.loss_bc1 + self.loss_bc2

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_pred))

        # Total loss
        self.loss =  self.adaptive_constant_res_tf * self.loss_res + \
            self.adaptive_constant_bcs_tf * self.loss_bcs + \
        self.adaptive_constant_ics_tf * self.loss_ics_u_t 

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate =starter_learning_rate
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.loss_tensor_list = [self.loss ,  self.loss_res,  self.loss_bcs,  self.loss_bc1 , self.loss_bc2 , self.loss_ics_u , self.loss_ics_u_t] 
        self.loss_list = ["total loss" , "loss_res" , "loss_bcs" , "loss_bc1", "loss_bc2", "loss_ics_u", "loss_ics_u_t"] 

        self.epoch_loss = dict.fromkeys(self.loss_list, 0)
        self.loss_history = dict((loss, []) for loss in self.loss_list)
        # Logger
        self.loss_u_log = []
        self.loss_r_log = []

        # self.saver = tf.train.Saver()

         # # Generate dicts for gradients storage
        self.dict_gradients_res_layers = self.generate_grad_dict()
        self.dict_gradients_bcs_layers = self.generate_grad_dict()
        self.dict_gradients_ics_layers = self.generate_grad_dict()
        
        # Gradients Storage
        self.grad_res = []
        self.grad_ics = []
        self.grad_bcs = []

        for i in range(  len( self.layers) - 1):
            self.grad_res.append(tf.gradients(self.loss_res, self.weights[i])[0])
            self.grad_bcs.append(tf.gradients(self.loss_bcs, self.weights[i])[0])
            self.grad_ics.append(tf.gradients(self.loss_ics_u_t, self.weights[i])[0])
          
        self.mean_grad_res_list = []
        self.mean_grad_bcs_list = []
        self.mean_grad_ics_list = []

        self.adaptive_constant_bcs_log = []
        self.adaptive_constant_ics_log = []
        self.adaptive_constant_res_log = []

        self.mean_grad_res_log = []
        self.mean_grad_bcs_log = []
        self.mean_grad_ics_log = []

        for i in range( len( self.layers) - 1):
            self.mean_grad_res_list.append(tf.reduce_mean(tf.abs(self.grad_res[i]))) 
            self.mean_grad_bcs_list.append(tf.reduce_mean(tf.abs(self.grad_bcs[i])))
            self.mean_grad_ics_list.append(tf.reduce_mean(tf.abs(self.grad_ics[i])))
        
        self.mean_grad_res = tf.reduce_mean(tf.stack(self.mean_grad_res_list))
        self.mean_grad_bcs = tf.reduce_mean(tf.stack(self.mean_grad_bcs_list))
        self.mean_grad_ics = tf.reduce_mean(tf.stack(self.mean_grad_ics_list))
        
        
         # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 2.0 / np.sqrt((in_dim +  out_dim) )
            # xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)

        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Forward pass for u
    def net_u(self, t, x):
        u = self.forward_pass(tf.concat([t, x], 1),
                              self.layers,
                              self.weights,
                              self.biases)
        return u

    # Forward pass for du/dt
    def net_u_t(self, t, x):
        u_t = tf.gradients(self.net_u(t, x), t)[0] / self.sigma_t
        return u_t

    # Forward pass for the residual
    def net_r(self, t, x):
        u = self.net_u(t, x)
        residual = self.operator(u, t, x,
                                 self.c,
                                 self.sigma_t,
                                 self.sigma_x)
        return residual
    
    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

        # Trains the model by minimizing the MSE loss

    
    def trainmb(self, nIter=10000, batch_size=128, log_NTK=False, update_lam=False):
        itValues = [1,100,1000,39999]

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches , 
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size // 3)
            X_bc1_batch, _ = self.fetch_minibatch(self.bcs_sampler[0], batch_size // 3)
            X_bc2_batch, _ = self.fetch_minibatch(self.bcs_sampler[1], batch_size // 3)
            
            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.t_ics_tf: X_ics_batch[:, 0:1],
                       self.x_ics_tf: X_ics_batch[:, 1:2],
                       self.u_ics_tf: u_ics_batch,
                       self.t_bc1_tf: X_bc1_batch[:, 0:1],
                        self.x_bc1_tf: X_bc1_batch[:, 1:2],
                       self.t_bc2_tf: X_bc2_batch[:, 0:1], 
                       self.x_bc2_tf: X_bc2_batch[:, 1:2],
                       self.t_r_tf: X_res_batch[:, 0:1], 
                       self.x_r_tf: X_res_batch[:, 1:2],
                       self.adaptive_constant_bcs_tf: self.adaptive_constant_bcs_val,
                       self.adaptive_constant_ics_tf: self.adaptive_constant_ics_val,
                       self.adaptive_constant_res_tf: self.adaptive_constant_res_val
                       }#self.lam_r_val}

            # Run the Tensorflow session to minimize the loss
            _ , batch_losses = self.sess.run( [  self.train_op , self.loss_tensor_list ] ,tf_dict)
            self.assign_batch_losses(batch_losses)
            for key in self.loss_history:
                self.loss_history[key].append(self.epoch_loss[key])

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time

                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs = self.sess.run(self.loss_bcs, tf_dict)
                loss_ics_u_t = self.sess.run(self.loss_ics_u_t, tf_dict)
                loss_res_value = self.sess.run(self.loss_res, tf_dict)

                self.print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e, Loss_ut_ics: %.3e,, Time: %.2f' %(it, loss_value, loss_res_value, loss_bcs, loss_ics_u_t, elapsed))
                
                # Compute and Print adaptive weights during training
                    # Compute the adaptive constant

                
            if it % 1000 == 0:

                mean_grad_res, mean_grad_bcs, mean_grad_ics = self.sess.run( [self.mean_grad_res,  self.mean_grad_bcs, self.mean_grad_ics  ], tf_dict)

                # # Print adaptive weights during training
                # self.adaptive_constant_res_val = mean_grad_res * ( 1.0 - self.rate) + self.rate * self.adaptive_constant_res_val
                # self.adaptive_constant_ics_val =  mean_grad_ics * ( 1.0 - self.rate) + self.rate * self.adaptive_constant_ics_val
                # self.adaptive_constant_bcs_val =  mean_grad_bcs * ( 1.0 - self.rate) + self.rate * self.adaptive_constant_bcs_val


                self.mean_grad_res_log.append( mean_grad_res)
                self.mean_grad_bcs_log.append( mean_grad_bcs)
                self.mean_grad_ics_log.append( mean_grad_ics)

                self.print('mean_grad_res: {:.3e}'.format( mean_grad_res))
                self.print('mean_grad_ics: {:.3e}'.format( mean_grad_ics))
                self.print('mean_grad_bcs: {:.3e}'.format( mean_grad_bcs))

                start_time = timeit.default_timer()
            if it in itValues:
                    self. save_gradients(tf_dict)
                    self.plot_layerLoss(it)
                    self.print("Gradients information stored ...")

            sys.stdout.flush()
 
        
    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 0:1], self.x_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

        # Evaluates predictions at test points

    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_r_tf: X_star[:, 0:1], self.x_r_tf: X_star[:, 1:2]}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star

             
    def save_gradients(self, tf_dict):
        ## Gradients #
        num_layers = len(self.layers)
        for i in range(num_layers - 1):
            grad_res, grad_bc1  , grad_ics  = self.sess.run([ self.grad_res[i],self.grad_bcs[i],self.grad_ics[i]], feed_dict=tf_dict)

            # save gradients of loss_r and loss_u
            self.dict_gradients_res_layers['layer_' + str(i + 1)].append(grad_res.flatten())
            self.dict_gradients_bcs_layers['layer_' + str(i + 1)].append(grad_bc1.flatten())
            self.dict_gradients_ics_layers['layer_' + str(i + 1)].append(grad_ics.flatten())

    def plot_layerLoss(self  , epoch):

        num_layers = len(self.layers)
        num_hidden_layers = num_layers -1
        cnt = 1
        fig = plt.figure(4, figsize=(13, 4))
        for j in range(num_hidden_layers):
            ax = plt.subplot(1, num_hidden_layers, cnt)
            ax.set_title('Layer {}'.format(j + 1))
            ax.set_yscale('symlog')
            gradients_res = self.dict_gradients_res_layers['layer_' + str(j + 1)][-1]
            gradients_bc1 = self.dict_gradients_bcs_layers['layer_' + str(j + 1)][-1]
            gradients_ics = self.dict_gradients_ics_layers['layer_' + str(j + 1)][-1]

            sns.distplot(gradients_res, hist=False,kde_kws={"shade": False},norm_hist=True,  label=r'$\nabla_\theta \mathcal{L}_r$')

            sns.distplot(gradients_bc1, hist=False,kde_kws={"shade": False},norm_hist=True,   label=r'$\nabla_\theta \mathcal{L}_{u_{bc1}}$')
            sns.distplot(gradients_ics, hist=False,kde_kws={"shade": False},norm_hist=True,   label=r'$\nabla_\theta \mathcal{L}_{u_{ics}}$')

            #ax.get_legend().remove()
            ax.set_xlim([-1.0, 1.0])
            #ax.set_ylim([0, 150])
            cnt += 1
        handles, labels = ax.get_legend_handles_labels()

        fig.legend(handles, labels, loc="center",  bbox_to_anchor=(0.5, -0.03),borderaxespad=0,bbox_transform=fig.transFigure, ncol=3)
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
        plt.close("all" , )
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


    def assign_batch_losses(self, batch_losses):
        for loss_values, key in zip(batch_losses, self.epoch_loss):
            self.epoch_loss[key] = loss_values


    def generate_grad_dict(self):
        num = len(self.layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict
  
            
    def plt_prediction(self , t , x , X_star , u_star , u_pred , r_star , r_pred):
        
        U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')
        r_star = griddata(X_star, r_star.flatten(), (t, x), method='cubic')
        U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
        R_pred = griddata(X_star, r_pred.flatten(), (t, x), method='cubic')


        plt.figure(figsize=(18, 9))
        plt.subplot(2, 3, 1)
        plt.pcolor(t, x, U_star, cmap='jet')
        plt.colorbar()
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Exact u(t, x)')
        plt.tight_layout()

        plt.subplot(2, 3, 2)
        plt.pcolor(t, x, U_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('Predicted u(t, x)')
        plt.tight_layout()

        plt.subplot(2, 3, 3)
        plt.pcolor(t, x, np.abs(U_star - U_pred), cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('Absolute error')
        plt.tight_layout()

        plt.subplot(2, 3, 4)
        plt.pcolor(t, x, r_star, cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('Exact r(t, x)')
        plt.tight_layout()

        plt.subplot(2, 3, 5)
        plt.pcolor(t, x, R_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('Predicted r(t, x)')
        plt.tight_layout()

        plt.subplot(2, 3, 6)
        plt.pcolor(t, x, np.abs(r_star - R_pred), cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('Absolute error')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirname,"prediction.png"))
        plt.close("all")

        

    def plot_grad(self ):

        fontsize = 17
        fig, ax = plt.subplots()
        fig.set_size_inches([16,8])
        ax.semilogy(self.mean_grad_bcs_log, label=r'$\bar{\nabla_\theta {u_{bc}}}$' , color = 'tab:green')
        ax.semilogy(self.mean_grad_ics_log, label=r'$\bar{\nabla_\theta {u_{ics}}}$' , color = 'tab:blue')
        ax.semilogy(self.mean_grad_res_log, label=r'$Max{\nabla_\theta {u_{phy}}}$' , color = 'tab:red')
        ax.set_xlabel("epochs", fontsize=fontsize)
        ax.set_ylabel(r'$\bar{\nabla_\theta {u}}$', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.legend(loc='center left', bbox_to_anchor=(-0.25, 0.5))

        plt.tight_layout()

        path = os.path.join(self.dirname,'grad_history.png')
        plt.savefig(path)
        plt.close("all" , )

    def plot_lambda(self ):

        fontsize = 17
        fig, ax = plt.subplots()
        fig.set_size_inches([16,8])
        ax.semilogy(self.mean_grad_bcs_log, label=r'$\bar{\nabla_\theta {u_{bc}}}$' , color = 'tab:green')
        ax.semilogy(self.mean_grad_ics_log, label=r'$\bar{\nabla_\theta {u_{ics}}}$' , color = 'tab:blue')
        ax.semilogy(self.mean_grad_res_log, label=r'$Max{\nabla_\theta {u_{phy}}}$' , color = 'tab:red')
        ax.set_xlabel("epochs", fontsize=fontsize)
        ax.set_ylabel(r'$\bar{\nabla_\theta {u}}$', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.legend(loc='center left', bbox_to_anchor=(-0.25, 0.5))

        ax2 = ax.twinx() 

        # fig, ax = plt.subplots()
        # fig.set_size_inches([15,8])
    
        ax2.semilogy(self.adaptive_constant_bcs_log, label=r'$\bar{\lambda_{bc}}$'  ,  linestyle='dashed' , color = 'tab:green') 
        ax2.semilogy(self.adaptive_constant_ics_log, label=r'$\bar{\lambda_{ics}}$' , linestyle='dashed'  , color = 'tab:blue')
        ax.semilogy(self.adaptive_constant_res_log, label=r'$Max{\lambda_{phy}}$' ,  linestyle='dashed' , color = 'tab:red')
        ax2.set_xlabel("epochs", fontsize=fontsize)
        ax2.set_ylabel(r'$\bar{\lambda}$', fontsize=fontsize)
        ax2.tick_params(labelsize=fontsize)
        ax2.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))

        plt.tight_layout()

        path = os.path.join(self.dirname,'lambda_history.png')
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
        shutil.copyfile('/home/vlq26735/code/PhD/GradientPathologiesPINNs/src/cases/Wave/M1.py', os.path.join(dirname, 'M1.py'))
        return dirname, logpath
    
