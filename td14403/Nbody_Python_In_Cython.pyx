
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time

#########################################################################

#SUBMIT

#Python only script in Cython
#Non-cythonsied and non-parallel

#This file is just the python script put into .pyx

#always use --- CC=gcc-5 python setup.py build_ext -fi --- to build file

#########################################################################



def Force(n,x,y,z,vx,vy,vz,dt,e,m,file):
    #Array for force in each coordinate built up with zeros
    #neccessary as code updates values later and can't append arrays
    Fx = np.zeros([n,n])
    Fy = np.zeros([n,n])
    Fz = np.zeros([n,n])

    G = 6.67*10**(-11) #This doesnt need to be -11
    step=0
    plotstep=0
    PsMax=100
    sMax = 10**3

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

          

            #We are now going to find the total of forces on each particle

            Ftx = np.sum(Fx, axis = 1) #Gives total force on each particle as an array Fx: Fx1 Fx2 ... Fxn
            Fty = np.sum(Fy, axis = 1)
            Ftz = np.sum(Fz, axis = 1)

            #finding new positions and velocities and changing the initial arrays for each coordinate

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
