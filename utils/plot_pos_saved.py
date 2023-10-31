import matplotlib.pyplot as plt
import numpy as np

file_name = "./saved_pos"

with open(file_name, 'r') as f:
	data = f.read()

vals = data.split('\n')[:-1]

#vals = vals[:264]
#vals = vals[971:1220]

poss = [[float(j) for j in i.split('|')[0].split(',')] for i in vals]
rots = [[float(j) for j in i.split('|')[1].split(',')] for i in vals]
vels = [[float(j) for j in i.split('|')[2].split(',')] for i in vals]


fig, ax = plt.subplots()

X = np.flip(np.array(poss)[:,1])
Y = np.arange(len(X))

#print(len(X))
#print(X[-1])

angles = np.flip(np.array(rots)[:,0] + np.pi/2)

#print(len(angles))
#print(angles[-2:])

X_dir = 1 * np.cos(angles)
Y_dir = 1 * np.sin(angles)

ax.quiver(X, Y, X_dir, Y_dir, width=0.002, headwidth=5, headlength=6)

#ax.axis([0, 6, -1, 10])
ax.set_xlim(-5, 5)


plt.show()
