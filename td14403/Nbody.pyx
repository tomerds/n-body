import numpy as np
cimport numpy as np
import time
from libc.math cimport sqrt
from cython.parallel cimport prange
cimport openmp
cimport cython


#########################################################################

#SUBMIT

#Parallel Version optimised for BLUECRYSTAL

#Number of threads set in command line
#Number of particles set in command line
#File name set in command line

#Too explore improvements in efficiency on laptop keep PsMax small 1-3

#always use --- CC=gcc-5 python setup.py build_ext -fi --- to build file on laptop

# BLUECRYSTAL: python setup.py build_ext -fi


#########################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
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
          file,
          int num):

    #Array for force in each coordinate built up with zeros
    #neccessary as code updates values later and can't append arrays

    cdef np.ndarray[np.float64_t, ndim=2] Fx = np.zeros([n,n],dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Fy = np.zeros([n,n],dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Fz = np.zeros([n,n],dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] Ftx = np.zeros(n)
    cdef np.ndarray[np.float64_t, ndim=1] Fty = np.zeros(n)
    cdef np.ndarray[np.float64_t, ndim=1] Ftz = np.zeros(n)


    #Positions are printed every PlotStep up to PsMax
    #Increase PsMax to increase number of data points in text File
    #Recomendation: if PsMax ~ 100 keep number of particles (n) ~ 100 to keep run time down
    #PsMax = 1 used for time tests

    #sMax defines number of timesteps calculated before datapoints are printed
    #Varying sMax will change the resolution of the data
    #1000 chosen as optimum

    cdef:
      double G = 6.67E-11
      int step=0, plotstep=0, PsMax=1, sMax = 1000, i, j
      double rUx, rUy, rUz, totalx, totaly, totalz, rD, sum = 0.0, av


    #Timer for total time
    initial = openmp.omp_get_wtime()

    while plotstep<PsMax:
        step = 0 #initialises the inner loop
        while step<sMax:

            #nxn force array timer
            i1 = openmp.omp_get_wtime()

            ###################################

            #LOOP 1
            #Total force on each particle matrix

            ###################################

            for i in prange(n,nogil=True, schedule = 'static', num_threads=num):

                for j in range(i,n):

                    #adding (i,n) here only defines values in top right triangle
                    #The bottom half are the same values but negative

                    if i!=j:

                        #Unit vector distance
                        rUx = x[i]-x[j]
                        rUy = y[i]-y[j]
                        rUz = z[i]-z[j]

                        #Distance
                        rD = rUx*rUx+rUy*rUy+rUz*rUz


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

                        #Adding softening factor (e) here can improve the interactions at small distances
                        #Chose e as 100 by analysing particle

                        Fx[i,j] = (-G*(m[i]*m[j])/((rD+e*e)*sqrt(rD+e*e)))*rUx
                        Fy[i,j] = (-G*(m[i]*m[j])/((rD+e*e)*sqrt(rD+e*e)))*rUy
                        Fz[i,j] = (-G*(m[i]*m[j])/((rD+e*e)*sqrt(rD+e*e)))*rUz


                        #adds bottom triangle half of matrix
                        #halves the number of caluclations needed

                        Fx[j,i] = -Fx[i,j]
                        Fy[j,i] = -Fy[i,j]
                        Fz[j,i] = -Fz[i,j]


                    if i == j:
                        Fx[i,j] = 0
                        Fy[i,j] = 0
                        Fz[i,j] = 0

            i2 = openmp.omp_get_wtime()
            #print("Elapsed time 1: {:8.6f} s".format(i2-i1))
            sum += i2-i1


            ###################################

            #LOOP 2
            #finding total force on each particle

            ###################################

            #de-comment to check times
            #i3 = openmp.omp_get_wtime()

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

            #i4 = openmp.omp_get_wtime()
            #print("Elapsed time 2: {:8.6f} s".format(i4-i3))

            ###################################

            #LOOP 3
            #finding new positions and velocities and changing the initial arrays for each coordinate

            ###################################

            #i5 = openmp.omp_get_wtime()

            for i in range(n):
                x[i]+=vx[i]*dt+0.5*(Ftx[i]/m[i])*(dt**2)
                y[i]+=vy[i]*dt+0.5*(Fty[i]/m[i])*(dt**2)
                z[i]+=vz[i]*dt+0.5*(Ftz[i]/m[i])*(dt**2)

                vx[i]+=(Ftx[i]/m[i])*dt
                vy[i]+=(Fty[i]/m[i])*dt
                vz[i]+=(Ftz[i]/m[i])*dt

            #i6 = openmp.omp_get_wtime()
            #print("Elapsed time 3: {:8.6f} s".format(i6-i5))

            step+=1

        #This has been changed to print a better formatted text file but may increase total time.
        #Previous script printed all x, y, & z values together and did not use a for loop
        #This was a neccessary addition for plotter file to check physicality

        for i in range(n):
          file.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i])+ " ")
        file.write("\n")

        plotstep+=1


    final = openmp.omp_get_wtime()

    av = sum/(PsMax*sMax) #Calculating average time to build nxn array over the number of steps

    totaltime = final-initial

    #print(av)
    #print("Elapsed time: {:8.6f} s".format(final-initial))

    file.write("Total elapsed time:" + str(totaltime) + "\n" + "Average nxn array time:" + str(av))
