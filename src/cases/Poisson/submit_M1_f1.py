
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

CHECKPOINT_PATH = "/home/vlq26735/code/PhD/GradientPathologiesPINNs/checkpoints/1DPoisson/final"
MODE = "M1_f1"

src_path = os.path.abspath(os.path.join('../../'))
print(f"src_path= {src_path}")

if src_path not in sys.path:
    sys.path.insert(0, src_path)


from src.cases.Poisson import M1_f1

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
    
    # Define solution and its Laplace
a = 4

def u(x, a):
  return np.sin(np.pi * a * x)

def u_xx(x, a):
  return -(np.pi * a)**2 * np.sin(np.pi * a * x)

# Define computional domain
bc1_coords = np.array([[0.0], [0.0]])
bc2_coords = np.array([[1.0], [1.0]])
dom_coords = np.array([[0.0], [1.0]])

# Training data on u(x) -- Dirichlet boundary conditions

nn  = 100

X_bc1 = dom_coords[0, 0] * np.ones((nn // 2, 1))
X_bc2 = dom_coords[1, 0] * np.ones((nn // 2, 1))
X_u = np.vstack([X_bc1, X_bc2])
Y_u = u(X_u, a)

X_r = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
Y_r = u_xx(X_r, a)

nn = 1000
X_star = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
u_star = u(X_star, a)
r_star = u_xx(X_star, a)

nIter =40001
bcbatch_size = 500
ubatch_size = 5000
mbbatch_size = 128



# Define model
layers = [1, 500 , 500 , 1]
activFun = "tanh"
starter_learning_rate = 1.0e-5

method =  "mini_batch"


# Create residual sampler
gpu_options = tf.GPUOptions(visible_device_list="0")
tf.reset_default_graph()
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=False, log_device_placement=False))
# with  as sess:

model = M1_f1.Possion_M1_f1(layers, X_u, Y_u, X_r, Y_r , MODE ,starter_learning_rate , CHECKPOINT_PATH , sess)    


model.print("Using mode: " , model.mode)
model.print("neural network: " , model.layers )
model.print("Bc1 Training data size : " ,X_bc1.shape[0])
model.print("Bc2 Training data size : " ,X_bc2.shape[0])
model.print("interior Training data size : " ,X_r.shape[0])

model.print("Activation function: " , activFun)
model.print("number of iterations: " , nIter)
model.print("starter_learning_rate: " , starter_learning_rate)

model.print("Method desciption : gradual learing rate , " ,  model.mode , ", with  " , method ," batch. ")

model.print("File directory: " , model.dirname)
sys.stdout.flush()
# Train model
start_time = time.time()

model.trainmb(nIter, mbbatch_size)
elapsed = time.time() - start_time

# Predictions
u_pred = model.predict_u(X_star)
r_pred = model.predict_r(X_star)
# Predictions

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
error_r = np.linalg.norm(r_star - r_pred, 2) / np.linalg.norm(r_star, 2)

model.print('elapsed: {:.2e}'.format(elapsed))

model.print('Relative L2 error_u: {:.2e}'.format(error_u))
model.print('Relative L2 error_r: {:.2e}'.format(error_r))


model.print('mean value of lambda_bc: {:.2e}'.format(np.average(model.adaptive_constant_bcs_log)))
model.print('first value of lambda_bc: {:.2e}'.format(model.adaptive_constant_bcs_log[0]))
model.print('Relative L2 error_u: {:.2e}'.format(error_u))
model.print('Relative L2 error_v: {:.2e}'.format(error_r))
model.print("file directory:" , model.dirname)
model.save_NN()
model.plot_grad()
model.plot_lambda()
model.plt_prediction( X_star , u_star , u_pred)
sys.stdout.flush()
