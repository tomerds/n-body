import sys
from Nbody import Force
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time

# Run file for parallel openmp scripts

# To run enter:
# python run_Nbody.py <number of particles> <number of threads> <filenumber>

#sys.argv[1] = n
#sys.argv[2] = num
#sys.argv[3] = File number

n =  int(sys.argv[1]) #number of particles


##############################################################################

V_range = 100 #Range of velocities particles can be initialised with
pos_range = 6*(10**8) #Range of positions which will be randomly initialised
m_lower = 10**23 #Lower mass particles can be initialised with (similar to the moon)
m_upper = 10**24 #Upper mass particles can be initialised with (similar to Earth)

dt = 100 #Timestep (seconds)
e = 100 #This is the softening length introduced to control forces at small distances, e>0

##############################################################################

#These initialise each positon and velocity between a range defined above
#Random initialisations used for time testing while varying number of particles

x = np.random.uniform(low = -pos_range, high = pos_range, size = n) #x: x1 x2 ... xn
y = np.random.uniform(low = -pos_range, high = pos_range, size = n) #y: y1 y2 ... yn
z = np.random.uniform(low = -pos_range, high = pos_range, size = n) #z: z1 z2 ... zn


vx = np.random.uniform(low = -V_range , high = V_range, size = n)
vy = np.random.uniform(low = -V_range , high = V_range, size = n)
vz = np.random.uniform(low = -V_range , high = V_range, size = n)
m = np.random.uniform(low = m_lower , high = m_upper, size = n)


##############################################################################

#opening a file to write positions

#Local (Change to relevant directory):
#file = open("/Users/...<directory>.../positions_%s.txt" % sys.argv[3],"w")

#BC:
file = open("positions_%s.txt" % sys.argv[3],"w")


#FUNCTIONS:

#OpenMP - set threads
Force(n,x,y,z,vx,vy,vz,dt,e,m,file,int(sys.argv[2]))

#Cython Serial - no threads
#Force(n,x,y,z,vx,vy,vz,dt,e,m,file)
