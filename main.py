import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wgd 
from scipy.integrate import odeint
import os


print('''
      N-body simulation
      -----------------
      
Simulations available to run:
      ''')
      
cwd = os.getcwd() # get current directory
files = [x for x in os.listdir(cwd) if (x.endswith(".txt") and x != "README.txt")] # find all txt files
index = 1
for file in files:
    print(str(index)+".",file[:-4])
    index+=1
    
selection = input("Type number: ")

filename = files[int(selection)-1] 
with open(filename,mode="r") as f: # open selected file and get relevant data
    lines = f.readlines()
    n = int(lines[0]) # number of bodies
    T = float(lines[2]) # time period to show when stop is clicked
    limit = float(lines[1]) # x,y,z limits
    masses = [float(i) for i in lines[3:n+3]]
    state0 = np.array([float(i) for i in lines[n+3:]])
    
    
def totalEnergy(y):
    '''
    Parameters
    ----------
    y : state vector at a time, t

    Returns
    -------
    E : total energy of system, E = K + U 
    '''
    E = 0
    for i in range(n):
        E += 1/2 * masses[i] * sum(y[3*n+3*i:3*n+3*i+2]**2) # kinetic energy of body i 
        for j in range(n):
            if j > i:
                r = np.sqrt((y[3*i]-y[3*j])**2 +    # radial distance between bodies i and j
                            (y[3*i+1]-y[3*j+1])**2 +
                            (y[3*i+2]-y[3*j+2])**2)
                E -= G*masses[i]*masses[j]/r # potential energy between bodies i and j
    return E


def motion(y,t):
    '''
    Parameters
    ----------
    y : state vector at a time, t, ordered:
    [x_1,y_1,z_1,x_2,y_2,z_2...,vx_1,vy_1...]
    t : time vector
    
    Returns
    -------
    subsequent v and vdot at each time in t, for each body ordered:
    [vx_1,vy_1,vz_1,vx_2,vy_2...,vdotx_1,vdoty_1,vdotz_1...]

    '''
    r_mags = np.zeros([n,n]) # define matrix of radial distances
    for r in range(n):
        for c in range(n):
            r_mags[r,c] = np.sqrt((y[3*r]-y[3*c])**2 + 
                                  (y[3*r+1]-y[3*c+1])**2 +
                                  (y[3*r+2]-y[3*c+2])**2)
    v = y[n*3:n*6]
    vdot = np.zeros((n,3)) # compute acceleration due to gravity for each pair of bodies
    for i in range(n):
        for j in range(n):
            if i!=j:
                vdot[i] = vdot[i] + G*(masses[j]*(y[3*j:3*j+3]-y[3*i:3*i+3])/r_mags[i,j]**3)
    vdot = np.concatenate(vdot)
    return np.concatenate((v,vdot))


def resetAxes(axes):
    '''
    Parameters
    ----------
    axes : The axes to reset
            
    '''
    axes.axes.set_xlim3d(-limit,limit)
    axes.axes.set_ylim3d(-limit,limit)
    axes.axes.set_zlim3d(-limit,limit)
    
    axes.set_xlabel("x / AU")
    axes.set_ylabel("y / AU")
    axes.set_zlabel("z / AU")


def closeCallback(event):
    global running
    running = False
    plt.close('all')
    
def stopCallback(event):
    global running
    running = False
    time = np.arange(0,T,T/100000)
    state = odeint(motion, state0, time).T

    for i in range(n):
        ax.plot(state[3*i],state[3*i+1],state[3*i+2])
    
     
    
G = 4*np.pi**2 # natural units   
    
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')

bax1 = plt.axes([0.85, 0.16, 0.1, 0.1]) 
buttonHandle1 = wgd.Button(bax1, 'Stop, show \nfull paths')
buttonHandle1.on_clicked(stopCallback)

bax2 = plt.axes([0.85, 0.05, 0.1, 0.1]) 
buttonHandle2 = wgd.Button(bax2, 'Close')
buttonHandle2.on_clicked(closeCallback)

total_t = 0
time = np.arange(0,10,0.01)

E0 = totalEnergy(state0)

state = odeint(motion, state0, time).T

running = True

while running:
    time = np.arange(total_t,total_t+0.2,0.005)
    state = odeint(motion, state[:,1], time).T
    
    ax.cla() 
    resetAxes(ax)
    
    for i in range(n):
        ax.plot(state[3*i],state[3*i+1],state[3*i+2])
    
    plt.show()
    plt.pause(0.01)
    total_t += 0.005