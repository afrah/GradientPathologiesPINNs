import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
# from Helmholtz2D_model_tf import Sampler, Helmholtz2D
import timeit
import os
os.environ["KMP_WARNINGS"] = "FALSE" 

import scipy
import scipy.io
import time
import logging
import sys

class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y

class Helmholtz2D:
    def __init__(self, layers, operator, ics_sampler, bcs_sampler, res_sampler, lam, mode ,  starter_learning_rate , PATH , sess ):

        
        self.mode = mode

        self.dirname, logpath = self.make_output_dir(PATH)
        self.logger = self.get_logger(logpath)     

        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x1, self.sigma_x1 = self.mu_X[0], self.sigma_X[0]
        self.mu_x2, self.sigma_x2 = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # Helmoholtz constant
        self.lam = tf.constant(lam, dtype=tf.float32)

        # Mode

        # Record stiff ratio

        # Adaptive constant
        self.beta = 0.9
        self.adaptive_constant_val = np.array(1.0)

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Define Tensorflow session
        self.sess = sess

        # Define placeholders and computational graph
        self.x1_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_bc4_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_bc4_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc4_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Define placeholder for adaptive constant
        self.adaptive_constant_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_val.shape)

        # Evaluate predictions
        self.u_bc1_pred = self.net_u(self.x1_bc1_tf, self.x2_bc1_tf)
        self.u_bc2_pred = self.net_u(self.x1_bc2_tf, self.x2_bc2_tf)
        self.u_bc3_pred = self.net_u(self.x1_bc3_tf, self.x2_bc3_tf)
        self.u_bc4_pred = self.net_u(self.x1_bc4_tf, self.x2_bc4_tf)

        self.u_pred = self.net_u(self.x1_u_tf, self.x2_u_tf)
        self.r_pred = self.net_r(self.x1_r_tf, self.x2_r_tf)

        # Boundary loss
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_tf - self.u_bc1_pred))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_tf - self.u_bc2_pred))
        self.loss_bc3 = tf.reduce_mean(tf.square(self.u_bc3_tf - self.u_bc3_pred))
        self.loss_bc4 = tf.reduce_mean(tf.square(self.u_bc4_tf - self.u_bc4_pred))
        self.loss_bcs =  (self.loss_bc1 + self.loss_bc2 + self.loss_bc3 + self.loss_bc4)

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_tf - self.r_pred))

        # Total loss
        self.loss = self.loss_res + self.loss_bcs

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = starter_learning_rate
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.loss_tensor_list = [self.loss ,  self.loss_res,  self.loss_bc1 ,  self.loss_bc2 ,  self.loss_bc3  ,  self.loss_bc4] 
        self.loss_list = ["total loss" , "loss_res" , "loss_bcs1" , "loss_bcs2", "loss_bcs3" ,  "loss_bcs4"] 

        self.epoch_loss = dict.fromkeys(self.loss_list, 0)
        self.loss_history = dict((loss, []) for loss in self.loss_list)
        

         # # Generate dicts for gradients storage
        self.dict_gradients_res_layers = self.generate_grad_dict()
        self.dict_gradients_bcs_layers = self.generate_grad_dict()
        
        # Gradients Storage
        self.grad_res = []
        self.grad_ics = []
        self.grad_bcs = []

        for i in range(  len( self.layers) - 1):
            self.grad_res.append(tf.gradients(self.loss_res, self.weights[i])[0])
            self.grad_bcs.append(tf.gradients(self.loss_bcs, self.weights[i])[0])
          
        self.mean_grad_res_list = []
        self.mean_grad_bcs_list = []


        self.mean_grad_res_log = []
        self.mean_grad_bcs_log = []

        for i in range( len( self.layers) - 1):
            self.mean_grad_res_list.append(tf.reduce_mean(tf.abs(self.grad_res[i]))) 
            self.mean_grad_bcs_list.append(tf.reduce_mean(tf.abs(self.grad_bcs[i])))
        
        self.mean_grad_res = tf.reduce_mean(tf.stack(self.mean_grad_res_list))
        self.mean_grad_bcs = tf.reduce_mean(tf.stack(self.mean_grad_bcs_list))
        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

 ############################################################

    def assign_batch_losses(self, batch_losses):
        for loss_values, key in zip(batch_losses, self.epoch_loss):
            self.epoch_loss[key] = loss_values
 ############################################################

     # Create dictionary to store gradients
    def generate_grad_dict(self):
        num = len(self.layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict

    # Save gradients
    def save_gradients(self, tf_dict):
        num_layers = len(self.layers)
        for i in range(num_layers - 1):
            grad_res_value, grad_bcs_value = self.sess.run([self.grad_res[i], self.grad_bcs[i]], feed_dict=tf_dict)

            # save gradients of loss_res and loss_bcs
            self.dict_gradients_res_layers['layer_' + str(i + 1)].append(grad_res_value.flatten())
            self.dict_gradients_bcs_layers['layer_' + str(i + 1)].append(grad_bcs_value.flatten())
        return None

    # Compute the Hessian
    def flatten(self, vectors):
        return tf.concat([tf.reshape(v, [-1]) for v in vectors], axis=0)

    def get_Hv(self, v):
        loss_gradients = self.flatten(tf.gradients(self.loss, self.weights))
        vprod = tf.math.multiply(loss_gradients,tf.stop_gradient(v))
        Hv_op = self.flatten(tf.gradients(vprod, self.weights))
        return Hv_op

    def get_Hv_res(self, v):
        loss_gradients = self.flatten(tf.gradients(self.loss_res, self.weights))
        vprod = tf.math.multiply(loss_gradients, tf.stop_gradient(v))
        Hv_op = self.flatten(tf.gradients(vprod, self.weights))
        return Hv_op

    def get_Hv_bcs(self, v):
        loss_gradients = self.flatten(tf.gradients(self.loss_bcs, self.weights))
        vprod = tf.math.multiply(loss_gradients,
                                 tf.stop_gradient(v))
        Hv_op = self.flatten(tf.gradients(vprod, self.weights))
        return Hv_op

    def get_H_op(self):
        self.P = self.flatten(self.weights).get_shape().as_list()[0]
        H = tf.map_fn(self.get_Hv, tf.eye(self.P, self.P),dtype='float32')
        H_bcs = tf.map_fn(self.get_Hv_bcs, tf.eye(self.P, self.P), dtype='float32')
        H_res = tf.map_fn(self.get_Hv_res, tf.eye(self.P, self.P), dtype='float32')

        return H, H_bcs, H_res

    # Xavier initialization
    def xavier_init(self,size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 2.0 / np.sqrt((in_dim + out_dim))
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H


    # Forward pass for u
    def net_u(self, x1, x2):
        u = self.forward_pass(tf.concat([x1, x2], 1))
        return u

    # Forward pass for residual
    def net_r(self, x1, x2):
        u = self.net_u(x1, x2)
        residual = self.operator(u, x1, x2,
                                 self.lam,
                                 self.sigma_x1,
                                 self.sigma_x2)
        return residual

    # Feed minibatch
    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=128):
        itValues = [1,100,1000,39999]

        start_time = timeit.default_timer()
        # Fetch boundary mini-batches
        for it in range(nIter):
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size)
            X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size)
            X_bc3_batch, u_bc3_batch = self.fetch_minibatch(self.bcs_sampler[2], batch_size)
            X_bc4_batch, u_bc4_batch = self.fetch_minibatch(self.bcs_sampler[3], batch_size)

            # Fetch residual mini-batch
            X_res_batch, f_res_batch = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x1_bc1_tf: X_bc1_batch[:, 0:1], self.x2_bc1_tf: X_bc1_batch[:, 1:2],
                        self.u_bc1_tf: u_bc1_batch,
                        self.x1_bc2_tf: X_bc2_batch[:, 0:1], self.x2_bc2_tf: X_bc2_batch[:, 1:2],
                        self.u_bc2_tf: u_bc2_batch,
                        self.x1_bc3_tf: X_bc3_batch[:, 0:1], self.x2_bc3_tf: X_bc3_batch[:, 1:2],
                        self.u_bc3_tf: u_bc3_batch,
                        self.x1_bc4_tf: X_bc4_batch[:, 0:1], self.x2_bc4_tf: X_bc4_batch[:, 1:2],
                        self.u_bc4_tf: u_bc4_batch,
                        self.x1_r_tf: X_res_batch[:, 0:1], self.x2_r_tf: X_res_batch[:, 1:2], self.r_tf: f_res_batch,
                        }

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)
            _, batch_losses = self.sess.run([self.train_op, self.loss_tensor_list] ,tf_dict)
            self.assign_batch_losses(batch_losses)
            for key in self.loss_history:
                self.loss_history[key].append(self.epoch_loss[key])

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss , loss_bcs , loss_res = self.sess.run([self.loss , self.loss_bcs, self.loss_res], tf_dict)

                print('It: %d| Loss: %.3e| Loss_bcs: %.3e| Loss_res: %.3e|Time: %.2f' % (it, loss, loss_bcs, loss_res,  elapsed))
                start_time = timeit.default_timer()

                
            if it % 1000 == 0:

                mean_grad_res, mean_grad_bcs = self.sess.run( [self.mean_grad_res,  self.mean_grad_bcs  ], tf_dict)
                self.mean_grad_res_log.append( mean_grad_res)
                self.mean_grad_bcs_log.append( mean_grad_bcs)

                self.print('mean_grad_res: {:.3e}'.format( mean_grad_res))
                self.print('mean_grad_bcs: {:.3e}'.format( mean_grad_bcs))

                start_time = timeit.default_timer()
            if it in itValues:
                    self. save_gradients(tf_dict)
                    self.plot_layerLoss(it)
                    self.print("Gradients information stored ...")

            sys.stdout.flush()
 
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
            sns.distplot(gradients_res, hist=False,kde_kws={"shade": False},norm_hist=True,  label=r'$\nabla_\theta \mathcal{L}_r$')
            sns.distplot(gradients_bc1, hist=False,kde_kws={"shade": False},norm_hist=True,   label=r'$\nabla_\theta \mathcal{L}_{u_{bc1}}$')

            #ax.get_legend().remove()
            ax.set_xlim([-1.0, 1.0])
            #ax.set_ylim([0, 150])
            cnt += 1
        handles, labels = ax.get_legend_handles_labels()

        fig.legend(handles, labels, loc="center",  bbox_to_anchor=(0.5, -0.03),borderaxespad=0,bbox_transform=fig.transFigure, ncol=3)
        text = 'layerLoss_epoch' + str(epoch) +'.png'
        plt.savefig(os.path.join(self.dirname,text) , bbox_inches='tight')
        plt.close("all")
   
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x1_u_tf: X_star[:, 0:1], self.x2_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x1_r_tf: X_star[:, 0:1], self.x2_r_tf: X_star[:, 1:2]}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star



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

    def make_output_dir(self , PATH):
        import shutil
        from datetime import datetime
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        dirname = os.path.join(PATH, datetime.now().strftime("%b-%d-%Y_%H-%M-%S-%f_") + self.mode)
        os.mkdir(dirname)
        text = 'output.log'
        logpath = os.path.join(dirname, text)
        shutil.copyfile('/home/vlq26735/code/PhD/GradientPathologiesPINNs/src/cases/Helmholtz/M1.py', os.path.join(dirname, 'M1.py'))
        return dirname, logpath
    
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

    def save_NN(self):
        import pickle
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
    
    def plt_prediction( self , x1 ,x2 , X_star , u_star , u_pred , f_star , f_pred):

        ### Plot ###

        # Exact solution & Predicted solution
        # Exact soluton
        U_star = griddata(X_star, u_star.flatten(), (x1, x2), method='cubic')
        F_star = griddata(X_star, f_star.flatten(), (x1, x2), method='cubic')

        # Predicted solution
        U_pred = griddata(X_star, u_pred.flatten(), (x1, x2), method='cubic')
        F_pred = griddata(X_star, f_pred.flatten(), (x1, x2), method='cubic')

        fig_1 = plt.figure(1, figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.pcolor(x1, x2, U_star, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Exact $u(x)$')

        plt.subplot(1, 3, 2)
        plt.pcolor(x1, x2, U_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Predicted $u(x)$')

        plt.subplot(1, 3, 3)
        plt.pcolor(x1, x2, np.abs(U_star - U_pred), cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Absolute error')
        plt.tight_layout()
        plt.show()

        fig_1 = plt.figure(1, figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.pcolor(x1, x2, F_star, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Exact $u(x)$')

        plt.subplot(1, 3, 2)
        plt.pcolor(x1, x2, F_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Predicted $u(x)$')

        plt.subplot(1, 3, 3)
        plt.pcolor(x1, x2, np.abs(F_star - F_pred), cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Absolute error')
        plt.show()
        plt.tight_layout()
        path = os.path.join(self.dirname,'prediction.png')
        plt.savefig(path)
        plt.close("all" , )

    def plot_grad(self):

        fig, ax = plt.subplots()
        fig.set_size_inches([15,8])
        ax.semilogy(self.mean_grad_res_log, label=r'$\bar{\nabla_\theta \mathcal{L}_{u_{phy}}}$')
        ax.semilogy(self.mean_grad_bcs_log, label=r'$\bar{\nabla_\theta \mathcal{L}_{u_{bc}}}$')

        ax.set_xlabel("epochs", fontsize=15)
        ax.set_ylabel(r'$\bar{\nabla_\theta \mathcal{L}}$', fontsize=15)
        ax.tick_params(labelsize=15)
        ax.legend()
        path = os.path.join(self.dirname,'grad_history.png')
        plt.savefig(path)
        plt.close("all" , )
