import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import ast

# Functions:

def Force(i, j, n):
    r1 = rMat[n][i]
    r2 = rMat[n][j]
    m1 = mvec[i]
    m2 = mvec[j]

    F = G * m1 * m2 /(np.linalg.norm(r2 - r1) ** 3)
    return F * (r2 - r1)

def NetForce(i, n):
    netF = np.array([0.0, 0.0, 0.0])
    for j in range(N):
        if i!=j:
            netF = netF + Force(i,j,n)
    return netF

# Constants, Timing and step size setup, and Initial Conditions

N = 5

G = 1
minCrashdist = 10**(-3)

totalTime = 100
n = 100000
dt = totalTime / n

mvec = np.zeros(N)
rMat = np.zeros((n, N, 3))
vMat = np.zeros((n, N, 3))
aMat = np.zeros((n, N, 3))

with open('initialCond.csv', newline='') as f:
    reader = csv.reader(f)
    for i in range(N):
        row = next(reader)
        mval = float(row[0])
        mvec[i] = mval
        rinitial = ast.literal_eval(row[1])
        rMat[0][i] = rinitial
        vinitial = ast.literal_eval(row[2])
        vMat[0][i] = vinitial
    for i in range(N):
        aMat[0][i] = NetForce(i, 0)/mvec[i]

# print("Initial Conds:")
# for i in range(N):
#     print("Object ",i)
#     print(mvec[i])
#     print(rMat[0][i])
#     print(vMat[0][i])
#     print(aMat[0][i])

for i in range(n-1):
    for j  in range(N):
        rMat[i+1][j] = rMat[i][j] + vMat[i][j] * dt + 0.5 * aMat[i][j] * dt**2
    for j  in range(N):
        aNew = NetForce(j, i+1)/mvec[j]
        vMat[i+1][j] = (vMat[i][j] + 0.5 * (aMat[i][j] + aNew) * dt)
        aMat[i+1][j] = aNew

##### Animation Code #####
sqFrameSize = 4

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis limits (adjust as needed)
ax.set_xlim([-sqFrameSize, sqFrameSize])
ax.set_ylim([-sqFrameSize, sqFrameSize])
ax.set_zlim([-sqFrameSize, sqFrameSize])

# Scatter plot for each body
points = [ax.plot([], [], [], 'o')[0] for _ in range(N)]

# Optional: trails (show path history)
trails = [ax.plot([], [], [], lw=0.5, alpha=0.5)[0] for _ in range(N)]

def init():
    for p, t in zip(points, trails):
        p.set_data([], [])
        p.set_3d_properties([])
        t.set_data([], [])
        t.set_3d_properties([])
    return points + trails

def update(frame):
    for j in range(N):
        x, y, z = rMat[frame, j]
        points[j].set_data([x], [y])
        points[j].set_3d_properties([z])
        
        # Trail up to current frame
        trails[j].set_data(rMat[:frame+1, j, 0], rMat[:frame+1, j, 1])
        trails[j].set_3d_properties(rMat[:frame+1, j, 2])
    return points + trails

ani = FuncAnimation(fig, update, frames=n, init_func=init, blit=False, interval=0.5)
## ani.save("nbody_sim.mp4", fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()