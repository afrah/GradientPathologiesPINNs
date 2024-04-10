
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

CHECKPOINT_PATH = "/home/vlq26735/code/PhD/GradientPathologiesPINNs/checkpoints/Helmholtz/final"
MODE = "M1"

src_path = os.path.abspath(os.path.join('../../'))
print(f"src_path= {src_path}")

if src_path not in sys.path:
    sys.path.insert(0, src_path)


from src.cases.Helmholtz import M1



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


a_1 = 1
a_2 = 4

def u(x, a_1, a_2):
    return np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

def u_xx(x, a_1, a_2):
    return - (a_1 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

def u_yy(x, a_1, a_2):
    return - (a_2 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

# Forcing
def f(x, a_1, a_2, lam):
    return u_xx(x, a_1, a_2) + u_yy(x, a_1, a_2) + lam * u(x, a_1, a_2)

def operator(u, x1, x2, lam, sigma_x1=1.0, sigma_x2=1.0):
    u_x1 = tf.gradients(u, x1)[0] / sigma_x1
    u_x2 = tf.gradients(u, x2)[0] / sigma_x2
    u_xx1 = tf.gradients(u_x1, x1)[0] / sigma_x1
    u_xx2 = tf.gradients(u_x2, x2)[0] / sigma_x2
    residual = u_xx1 + u_xx2 + lam * u
    return residual

# Parameter
lam = 1.0



kernel_size = 300

bc1_coords = np.array([[-1.0, -1.0], [1.0, -1.0]])
bc2_coords = np.array([[1.0, -1.0],[1.0, 1.0]])
bc3_coords = np.array([[1.0, 1.0], [-1.0, 1.0]])
bc4_coords = np.array([[-1.0, 1.0], [-1.0, -1.0]])

dom_coords = np.array([[-1.0, -1.0], [1.0, 1.0]])


# Create boundary conditions samplers
bc1 = Sampler(2, bc1_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC1')
bc2 = Sampler(2, bc2_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC2')
bc3 = Sampler(2, bc3_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC3')
bc4 = Sampler(2, bc4_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC4')
bcs_sampler = [bc1, bc2, bc3, bc4]
ics_sampler = None
res_sampler = Sampler(2, dom_coords, lambda x: f(x, a_1, a_2, lam), name='Forcing')

# Test data  
nn = 100
x1 = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
x2 = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
x1, x2 = np.meshgrid(x1, x2)
X_star = np.hstack((x1.flatten()[:, None], x2.flatten()[:, None]))

# Exact solution
u_star = u(X_star, a_1, a_2)
f_star = f(X_star, a_1, a_2, lam)

nIter =40001
batch_size = 128

# Define model
layers = [2, 50, 50, 50, 1]

method =   "mini_batch"
activFun = "tanh"
starter_learning_rate = 1.0e-3

# [elapsed, error_u , model] = test_method(mtd , layers,  ics_sampler, bcs_sampler, res_sampler, c ,kernel_size , X_star , u_star , r_star , nIter ,mbbatch_size , bcbatch_size , ubatch_size )
tf.reset_default_graph()
gpu_options = tf.GPUOptions(visible_device_list="0")
sess =  tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=False, log_device_placement=False))
# sess.run(init)

model = M1.Helmholtz2D(layers, operator, ics_sampler, bcs_sampler, res_sampler, lam, MODE,starter_learning_rate , CHECKPOINT_PATH , sess)

# Train model

model.print("Using mode: " , model.mode)
model.print("neural network: " , model.layers )
model.print("Batch size : " ,kernel_size)

model.print("Activation function: " , activFun)
model.print("number of iterations: " , nIter)
model.print("starter_learning_rate: " , starter_learning_rate)

model.print("Method desciption : gradual learing rate , " ,  model.mode , ", with  " , method ," batch. ")

model.print("File directory: " , model.dirname)
sys.stdout.flush()
# Train model
start_time = time.time()

model.train(nIter, batch_size)
elapsed = time.time() - start_time

# Predictions
u_pred = model.predict_u(X_star)
f_pred = model.predict_r(X_star)
# Predictions

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
error_f = np.linalg.norm(f_star - f_pred, 2) / np.linalg.norm(f_star, 2)

model.print('elapsed: {:.2e}'.format(elapsed))

model.print('Relative L2 error_u: {:.2e}'.format(error_u))
model.print('Relative L2 error_f: {:.2e}'.format(error_f))


model.print("file directory:" , model.dirname)
model.save_NN()
model.plot_grad()
model.plt_prediction( x1 ,x2 , X_star , u_star , u_pred , f_star , f_pred)
sys.stdout.flush()
