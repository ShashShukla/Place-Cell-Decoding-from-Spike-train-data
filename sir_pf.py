'''
Place cell decoding using Iterated Extended Kalman Filter

Author: Shashwat Shukla
Date: 23rd November 2018
'''
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

# Seed the random number generator with the current time
np.random.seed(np.int(time.time()))

# Simulate random walk
T = 1000  # number of time-steps
path = np.random.randn(T, 2) * 0.03
path = np.cumsum(path, axis=0)

# Create place cells
N = 25  # Number of place cells
param = np.zeros((N, 5))  # Parameter table for the place cells
param[:, 0:2] = np.random.randn(N, 2)  # Centres of the receptive fields
param[:, 2:4] = np.abs(np.random.randn(N, 2))  # Size of the receptive fields
param[:, 4] = np.random.randn(N) * 0.5  # Offsets of the receptive fields

# Compute responses of place cells along the traversed path
R = np.zeros((T, N))  # Spiking responses of the place cells
spikes = [[] for i in range(N)]  # Raster plot
for i in range(N):
    r = np.exp(- 0.5 * ((path[:, 0] - param[i, 0]) / param[i, 2]) ** 2)
    r = r * np.exp(- 0.5 * ((path[:, 1] - param[i, 1]) / param[i, 3]) ** 2)
    r = r * np.exp(param[i, 4])
    r = np.random.poisson(r)
    R[:, i] = r
    spikes[i] = np.where(r == 1)[0]

# Display raster plot
if(len(spikes[0]) == 0):  # Handle bug in eventplot
    spikes[0] = np.array([1])
plt.eventplot(spikes)
plt.title('Raster plot', fontweight='bold')
plt.ylabel('Neuron index', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Function to compute the log-likelihood for each of the particles
def computeLogLikelihoods(xtildes, y, param):
    lls = np.zeros(nparticles)
    for i in range(nparticles):
        xdelta = ((xtildes[i, 0] - param[:, 0]) / param[:, 2]) ** 2
        ydelta = ((xtildes[i, 1] - param[:, 1]) / param[:, 3]) ** 2
        loglambdas = -0.5 * (xdelta + ydelta) + param[:, 4]
        lambdas = np.exp(loglambdas)
        # Compute log-posterior
        lls[i] = np.sum(loglambdas * y - lambdas)
    return lls


# Decoding stage
nparticles = 1000  # Number of particles
# Draw from the initial distribution
xtildes = np.random.randn(nparticles, 2)
xs = np.zeros((T, 2))
Ws = np.zeros((T, 2, 2))
particles = np.zeros((T, nparticles, 2))
W = np.identity(2) * 0.03 ** 2
Wsqrt = np.sqrt(W)

for i in range(T):
    # Compute weights according to w = p(y|x_t)
    y = R[i]
    ws = computeLogLikelihoods(xtildes, y, param)
    # Normalize weights
    ws = np.exp(ws - np.max(ws))
    ws = np.divide(ws, np.sum(ws))
    # Importance sampling with multinomial resampling
    ns = np.random.multinomial(nparticles, ws)
    c = 0
    for j in range(nparticles):
        for k in range(ns[j]):
            xtildes[c, :] = xtildes[j, :]
            c += 1
    # Store estimates from the ith timestep
    particles[i, :, :] = xtildes
    xs[i, :] = np.mean(xtildes, axis=0)
    Ws[i, :, :] = np.cov(xtildes, rowvar=False)
    # Propagate each particle through the state equation
    xtildes = xtildes + np.matmul(Wsqrt, np.random.randn(2, nparticles)).T


# Display recorded and decoded path of rat
plt.plot(path[:, 0], path[:, 1], label='Actual path')
plt.plot(xs[:, 0], xs[:, 1], label='Decoded path')
plt.legend()
plt.title('Path of the rat', fontweight='bold')
plt.show()

# Display recorded and decoded x-coordinate
plt.plot(path[:, 0], label='True')
plt.plot(xs[:, 0], label='Decoded')
plt.legend()
plt.title('x-coordinate', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Display recorded and decoded y-coordinate
plt.plot(path[:, 0], label='True')
plt.plot(xs[:, 0], label='Decoded')
plt.legend()
plt.title('y-coordinate', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()
