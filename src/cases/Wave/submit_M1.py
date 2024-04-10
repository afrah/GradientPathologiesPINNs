
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

CHECKPOINT_PATH = "/home/vlq26735/code/PhD/GradientPathologiesPINNs/checkpoints/1DWave/final"
MODE = "M1"

src_path = os.path.abspath(os.path.join('../../'))
print(f"src_path= {src_path}")

if src_path not in sys.path:
    sys.path.insert(0, src_path)


from src.cases.Wave import M1

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



# Define the exact solution and its derivatives
def u(x, a, c):
    """
    :param x: x = (t, x)
    """
    t = x[:,0:1]
    x = x[:,1:2]
    return np.sin(np.pi * x) * np.cos(c * np.pi * t) + a * np.sin(2 * c * np.pi* x) * np.cos(4 * c  * np.pi * t)

def u_t(x,a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_t = -  c * np.pi * np.sin(np.pi * x) * np.sin(c * np.pi * t) -  a * 4 * c * np.pi * np.sin(2 * c * np.pi* x) * np.sin(4 * c * np.pi * t)
    return u_t

def u_tt(x, a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_tt = -(c * np.pi)**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - a * (4 * c * np.pi)**2 *  np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return u_tt

def u_xx(x, a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_xx = - np.pi**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) -  a * (2 * c * np.pi)** 2 * np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return  u_xx


def r(x, a, c):
    return u_tt(x, a, c) - c**2 * u_xx(x, a, c)
# Define PINN model
a = 0.5
c = 2

kernel_size = 300

# Domain boundaries
ics_coords = np.array([[0.0, 0.0],  [0.0, 1.0]])
bc1_coords = np.array([[0.0, 0.0],  [1.0, 0.0]])
bc2_coords = np.array([[0.0, 1.0],  [1.0, 1.0]])
dom_coords = np.array([[0.0, 0.0],  [1.0, 1.0]])

# Create initial conditions samplers
ics_sampler = Sampler(2, ics_coords, lambda x: u(x, a, c), name='Initial Condition 1')

# Create boundary conditions samplers
bc1 = Sampler(2, bc1_coords, lambda x: u(x, a, c), name='Dirichlet BC1')
bc2 = Sampler(2, bc2_coords, lambda x: u(x, a, c), name='Dirichlet BC2')
bcs_sampler = [bc1, bc2]

# Create residual sampler
res_sampler = Sampler(2, dom_coords, lambda x: r(x, a, c), name='Forcing')
coll_sampler = Sampler(2, dom_coords, lambda x: u(x, a, c), name='coll')


nIter =40001
mbbatch_size = 300




# Define model
layers = [2, 256, 256,256, 1]


nn = 200
t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
t, x = np.meshgrid(t, x)
X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

u_star = u(X_star, a,c)
r_star = r(X_star, a, c)

method =   "mini_batch"
activFun = "tanh"
starter_learning_rate = 1.0e-3

# [elapsed, error_u , model] = test_method(mtd , layers,  ics_sampler, bcs_sampler, res_sampler, c ,kernel_size , X_star , u_star , r_star , nIter ,mbbatch_size , bcbatch_size , ubatch_size )
tf.reset_default_graph()
gpu_options = tf.GPUOptions(visible_device_list="0")
sess =  tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=False, log_device_placement=False))
# sess.run(init)

model = M1.PINN(layers ,  ics_sampler, bcs_sampler, res_sampler, c , MODE , starter_learning_rate , CHECKPOINT_PATH , sess)
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


model.print("file directory:" , model.dirname)
model.save_NN()
model.plot_grad()
model.plot_lambda()
model.plt_prediction( t , x , X_star , u_star , u_pred , r_star , r_pred)
sys.stdout.flush()
