import numpy as np
cimport numpy as np
import time
from libc.math cimport sqrt


#########################################################################

#SUBMIT

#Cythonised non-parallel

#Speed up of factor 72X

#This file has cythonised variables
#Big improvements came from cdefining the numpy arrays
#Big improvement from using C sqrt
#Is it possible to cdef the force function and put inside def function to loop?

#always use --- CC=gcc-5 python setup.py build_ext -fi --- to build file


#########################################################################



def Force(int n,
          np.ndarray[np.float64_t, ndim=1] x,
          np.ndarray[np.float64_t, ndim=1] y,
          np.ndarray[np.float64_t, ndim=1] z,
          np.ndarray[np.float64_t, ndim=1] vx,
          np.ndarray[np.float64_t, ndim=1] vy,
          np.ndarray[np.float64_t, ndim=1] vz,
          double dt,
          double e,
          np.ndarray[np.float64_t, ndim=1] m,
          file):

    #Array for force in each coordinate built up with zeros
    #neccessary as code updates values later and can't append arrays
    cdef np.ndarray[np.float64_t, ndim=2] Fx = np.zeros([n,n],dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Fy = np.zeros([n,n],dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Fz = np.zeros([n,n],dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] Ftx = np.zeros(n)
    cdef np.ndarray[np.float64_t, ndim=1] Fty = np.zeros(n)
    cdef np.ndarray[np.float64_t, ndim=1] Ftz = np.zeros(n)


    cdef:
      double G = 6.67E-11
      int step=0, plotstep=0, PsMax=100, sMax = 1000, i, j
      double rUx, rUy, rUz, totalx, totaly, totalz


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


                        ###################################
                        #finding force from each particle interaction i.e. 1 on 2, 1 on 3 etc.
                        #Gives a matrix of the form:

                        #     P1   P2   P3
                        #   ----------------
                        # P1|  0    3N  12N
                        # P2| -3N   0    7N
                        # P3| -12N -7N   0

                        #C library sqrt has been used here which gives a speed up

                        ###################################

                        Fx[i,j] = (-G*(m[i]*m[j])/((rD+e**2)*sqrt(rD+e**2)))*rUx
                        Fy[i,j] = (-G*(m[i]*m[j])/((rD+e**2)*sqrt(rD+e**2)))*rUy
                        Fz[i,j] = (-G*(m[i]*m[j])/((rD+e**2)*sqrt(rD+e**2)))*rUz


                        #adds bottom triangle half of matrix
                        #halves the number of caluclations needed
                        Fx[j,i] = -Fx[i,j]
                        Fy[j,i] = -Fy[i,j]
                        Fz[j,i] = -Fz[i,j]


                    if i == j:
                        Fx[i,j] = 0
                        Fy[i,j] = 0
                        Fz[i,j] = 0

            ###################################
            #finding total force on each particle
            #Massive speed up here as opposed to using numpy.sum!
            ###################################

            for i in range(n):
                      totalx=0
                      totaly=0
                      totalz=0
                      for j in range(n):
                          totalx+=Fx[i,j]
                          totaly+=Fy[i,j]
                          totalz+=Fz[i,j]
                      Ftx[i] = totalx
                      Fty[i] = totaly
                      Ftz[i] = totalz


            ###################################
            #finding new positions and velocities and changing the initial arrays for each coordinate
            ###################################

            for i in range(n):
                x[i]+=vx[i]*dt+0.5*(Ftx[i]/m[i])*(dt**2)
                y[i]+=vy[i]*dt+0.5*(Fty[i]/m[i])*(dt**2)
                z[i]+=vz[i]*dt+0.5*(Ftz[i]/m[i])*(dt**2)

                vx[i]+=(Ftx[i]/m[i])*dt
                vy[i]+=(Fty[i]/m[i])*dt
                vz[i]+=(Ftz[i]/m[i])*dt

            step+=1


        for i in range(n):
          file.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i])+ " ")
        file.write("\n")

        plotstep+=1

    final = time.time()

    print("Elapsed time: {:8.6f} s".format(final-initial))
