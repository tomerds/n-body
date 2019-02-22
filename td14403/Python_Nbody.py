import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time

#########################################################################

#SUBMIT

#GOOD PYTHON SCRIPT!
#This file is just the python script used before translation to cython
#This file writes everything to a text file and can be used for comparison of speeds to cython scripts

#########################################################################


n = 20 #number of particles

#These can be played with:
V_range = 100
pos_range = 6*(10**8)
m_lower = 10**23 #similar to the moon
m_upper = 10**24 #similar to Earth

#Time step:
dt = 100 #seconds

e = 100 #This is the softening length introduced to control forces at small distances, e>0

#These initialise each positon and velocity between a range defined above

x = np.random.uniform(low = -pos_range, high = pos_range, size = n) #x: x1 x2 ... xn
y = np.random.uniform(low = -pos_range, high = pos_range, size = n) #y: y1 y2 ... yn
z = np.random.uniform(low = -pos_range, high = pos_range, size = n) #z: z1 z2 ... zn

vx = np.random.uniform(low = -V_range , high = V_range, size = n)
vy = np.random.uniform(low = -V_range , high = V_range, size = n)
vz = np.random.uniform(low = -V_range , high = V_range, size = n)
m = np.random.uniform(low = m_lower , high = m_upper, size = n)

#opening a file to write positions

####################Insert relevant path:###########################
file = open("/Users/...<DIRECTORY>.../positions.txt","w")


def Force(n,x,y,z,vx,vy,vz,dt,e):
    #Array for force in each coordinate built up with zeros
    #neccessary as code updates values later and can't append arrays
    Fx = np.zeros([n,n])
    Fy = np.zeros([n,n])
    Fz = np.zeros([n,n])

    G = 6.67*10**(-11)
    step=0
    plotstep=0
    PsMax=100 #Increase to increase number of data points
    sMax = 10**3 #Decrease to increase resolution of data points

    #Total time Timer
    initial = time.time()

    while plotstep<PsMax:
        step = 0 #initialises the inner loop
        while step<sMax:
            for i in range(n):

                for j in range(i,n):
                    #adding (i,n) here only defines values in top right triangle
                    #The bottom half are the same values but negative

                    if i!=j:

                        #Unit vector distance
                        rUx = x[i]-x[j]
                        rUy = y[i]-y[j]
                        rUz = z[i]-z[j]

                        #Distance
                        rD = rUx**2+rUy**2+rUz**2


                        Fx[i,j] = (-G*(m[i]*m[j])/((rD+e**2)*math.sqrt(rD+e**2)))*rUx
                        Fy[i,j] = (-G*(m[i]*m[j])/((rD+e**2)*math.sqrt(rD+e**2)))*rUy
                        Fz[i,j] = (-G*(m[i]*m[j])/((rD+e**2)*math.sqrt(rD+e**2)))*rUz

                        #adds bottom triangle half of matrix
                        #halves the number of caluclations needed
                        Fx[j,i] = -Fx[i,j]
                        Fy[j,i] = -Fy[i,j]
                        Fz[j,i] = -Fz[i,j]

                    if i == j:
                        Fx[i,j] = 0
                        Fy[i,j] = 0
                        Fz[i,j] = 0


            #Finding total force on each particle:

            Ftx = np.sum(Fx, axis = 1) #Gives total force on each particle as an array Fx: Fx1 Fx2 ... Fxn
            Fty = np.sum(Fy, axis = 1)
            Ftz = np.sum(Fz, axis = 1)

            #Finding new positions and velocities and changing the initial arrays for each coordinate

            for i in range(n):
                x[i]+=vx[i]*dt+0.5*(Ftx[i]/m[i])*(dt**2)
                y[i]+=vy[i]*dt+0.5*(Fty[i]/m[i])*(dt**2)
                z[i]+=vz[i]*dt+0.5*(Ftz[i]/m[i])*(dt**2)

                vx[i]+=(Ftx[i]/m[i])*dt
                vy[i]+=(Fty[i]/m[i])*dt
                vz[i]+=(Ftz[i]/m[i])*dt

            step+=1

        file.write(str(x) + "," + str(y) + "," + str(z)+ "\n")

        plotstep+=1

    final = time.time()

    print("Elapsed time: {:8.6f} s".format(final-initial))




Force(n,x,y,z,vx,vy,vz,dt,e)
