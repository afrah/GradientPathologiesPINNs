
import os
import sys
import scipy
import scipy.io
import time

import os.path
import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_WARNINGS"] = "FALSE"


import tensorflow as tf
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

CHECKPOINT_PATH = "/home/vlq26735/code/PhD/GradientPathologiesPINNs/checkpoints/Klein_Gordon/final"
MODE = "M1_f1"

src_path = os.path.abspath(os.path.join('../../'))
print(f"src_path= {src_path}")

if src_path not in sys.path:
    sys.path.insert(0, src_path)


from src.cases.Klein_Gordon import M1_f1

def u(x):
    """
    :param x: x = (t, x)
    """
    return x[:, 1:2] * np.cos(5 * np.pi * x[:, 0:1]) + (x[:, 0:1] * x[:, 1:2])**3

def u_tt(x):
    return - 25 * np.pi**2 * x[:, 1:2] * np.cos(5 * np.pi * x[:, 0:1]) + 6 * x[:,0:1] * x[:,1:2]**3

def u_xx(x):
    return np.zeros((x.shape[0], 1)) +  6 * x[:,1:2] * x[:,0:1]**3

def f(x, alpha, beta, gamma, k):
    return u_tt(x) + alpha * u_xx(x) + beta * u(x) + gamma * u(x)**k

def operator(u, t, x, alpha, beta, gamma, k,  sigma_t=1.0, sigma_x=1.0):
    u_t = tf.gradients(u, t)[0] / sigma_t
    u_x = tf.gradients(u, x)[0] / sigma_x
    u_tt = tf.gradients(u_t, t)[0] / sigma_t
    u_xx = tf.gradients(u_x, x)[0] / sigma_x
    residual = u_tt + alpha * u_xx + beta * u + gamma * u**k
    return residual


class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name = None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name
    def sample(self, N):
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y
    
# Parameters of equations
alpha = -1.0
beta = 0.0
gamma = 1.0
k = 3

# Domain boundaries
ics_coords = np.array([[0.0, 0.0], [0.0, 1.0]])
bc1_coords = np.array([[0.0, 0.0], [1.0, 0.0]])
bc2_coords = np.array([[0.0, 1.0], [1.0, 1.0]])
dom_coords = np.array([[0.0, 0.0], [1.0, 1.0]])

# Create initial conditions samplers
ics_sampler = Sampler(2, ics_coords, lambda x: u(x), name='Initial Condition 1')

# Create boundary conditions samplers
bc1 = Sampler(2, bc1_coords, lambda x: u(x), name='Dirichlet BC1')
bc2 = Sampler(2, bc2_coords, lambda x: u(x), name='Dirichlet BC2')
bcs_sampler = [bc1, bc2]

# Create residual sampler
res_sampler = Sampler(2, dom_coords, lambda x: f(x, alpha, beta, gamma, k), name='Forcing')

# Define model
layers =[2, 50, 50, 50, 50, 50, 1]


nIter =40001
batch_size = 128


method =   "mini_batch"
activFun = "tanh"
starter_learning_rate = 1.0e-3

tf.reset_default_graph()
gpu_options = tf.GPUOptions(visible_device_list="0")
sess =  tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=False, log_device_placement=False))
# sess.run(init)
model = M1_f1.Klein_Gordon(layers, operator, ics_sampler, bcs_sampler, res_sampler, alpha, beta, gamma, k, MODE,starter_learning_rate, CHECKPOINT_PATH , sess)

model.print("Using mode: " , model.mode)
model.print("neural network: " , model.layers )
model.print("Batch size : " ,batch_size)

model.print("Activation function: " , activFun)
model.print("number of iterations: " , nIter)
model.print("starter_learning_rate: " , starter_learning_rate)

model.print("Method desciption : gradual learing rate , " ,  model.mode , ", with  " , method ," batch. ")

model.print("File directory: " , model.dirname)
sys.stdout.flush()

# Train model
model.train(nIter=nIter, batch_size=batch_size)


# Test data
nn = 100
t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
t, x = np.meshgrid(t, x)
X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

# Exact solution
u_star = u(X_star)
f_star = f(X_star, alpha, beta, gamma, k)

# Predictions
u_pred = model.predict_u(X_star)
r_pred = model.predict_r(X_star)

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
error_r = np.linalg.norm(f_star - r_pred, 2) / np.linalg.norm(f_star, 2)

model.print('Relative L2 error_u: {:.2e}'.format(error_u))

model.print('Relative L2 error_f: {:.2e}'.format(error_r))


model.print("file directory:" , model.dirname)
model.save_NN()
model.plot_grad()
model.plt_prediction( t ,x , X_star , u_star , u_pred , f_star , r_pred)
sys.stdout.flush()
