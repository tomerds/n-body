from mpi4py import MPI
import numpy as np
import math
import random
import time


#########################################################################

#SUBMIT

#mpi4py script

#Significantly slower than OpenMP

#########################################################################


n = 3 #Vary number of particles

#These can be played with:
V_range = 100
pos_range = 6*(10**8)
m_lower = 10**23 #similar to the moon
m_upper = 10**24 #similar to Earth

#Timestep:
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

#Local:
#file = open("/Users/...<PATH>.../positions.txt","w")

#BC:
file = open("positions.txt","w")


def Force(n,x,y,z,vx,vy,vz,dt,e):

    #Array for force in each coordinate built up with zeros
    #neccessary as code updates values later and can't append arrays
    Fx = np.zeros([n,n])
    Fy = np.zeros([n,n])
    Fz = np.zeros([n,n])

    Ftx = np.zeros(n)
    Fty = np.zeros(n)
    Ftz = np.zeros(n)

    MASTER = 0
    FROM_MASTER = 1
    FROM_WORKER = 2


    comm = MPI.COMM_WORLD
    taskid = comm.Get_rank()
    numtasks = comm.Get_size()
    numworkers = numtasks-1

    G = 6.67*10**(-11)
    step=0
    plotstep=0
    PsMax=10 #Number of data points
    sMax = 10**3 #Decrease to increase resolution of data points

    averow = n//numworkers
    extra = n%numworkers

    initial = time.time()

    while plotstep<PsMax:
        step = 0 #initialises the inner loop
        while step<sMax:

            #*************************** MASTER **************************
            if taskid == MASTER:


                if numworkers<2:
                    #De-Comment for local:
                    #print("You must enter >2 processors\n")
                    comm.Abort()

                offset = 0

                for i in range(1,numworkers+1):
                    rows = averow
                    if i <= extra:
                        rows+=1

                    #Sending data to workers

                    comm.send(offset, dest=i, tag=FROM_MASTER)
                    comm.send(rows, dest=i, tag=FROM_MASTER)
                    comm.Send(x, dest=i, tag=FROM_MASTER)
                    comm.Send(y, dest=i, tag=FROM_MASTER)
                    comm.Send(z, dest=i, tag=FROM_MASTER)

                    offset += rows

                    #Waiting for results from workers

                #Initialising total force array inbetween each step
                for i in range(n):
                    Ftx[i]=0
                    Fty[i]=0
                    Ftz[i]=0

                #Looping through receiving from different workers
                #Set up such that first worker will finish first, second finshes second, etc.
                #Master will begin updating total force array based on first block of data received
                #Once total force for first block is calculated Master loops to receive from second worker
                #And so on till array is fully calculated...
                for xi in range(1,numworkers+1):

                    offset = comm.recv(source=xi, tag=FROM_WORKER)
                    rows = comm.recv(source=xi, tag=FROM_WORKER)
                    comm.Recv(Fx, source=xi, tag=FROM_WORKER)
                    comm.Recv(Fy, source=xi, tag=FROM_WORKER)
                    comm.Recv(Fz, source=xi, tag=FROM_WORKER)

                    for i in range(n):

                        for j in range(n):

                            #updating total force with first part of array from j=0 to j=n/num_threads
                            #
                            # _
                            #| |_
                            #|   |_
                            #|_____|
                            #
                            #^Sums these parts of the total force array^

                            if i>=offset and i<(offset+rows) and j>=0 and j<(offset+rows):
                                Ftx[i]+=Fx[i,j]
                                Fty[i]+=Fy[i,j]
                                Ftz[i]+=Fz[i,j]
                                #print("Ftx=",Ftx)



                            #Fills the remaining parts of the array once they have arrived
                            # _____
                            #| |xxx|
                            #|   |x|
                            #|_____|


                            if i>=0 and i<offset and j>= offset and j<(offset+rows):
                                Ftx[i]+=Fx[i,j]
                                Fty[i]+=Fy[i,j]
                                Ftz[i]+=Fz[i,j]

                            else:
                                continue

                    #Updates positions and velocities
                    for i in range(n):
                        x[i]+=vx[i]*dt+0.5*(Ftx[i]/m[i])*(dt**2)
                        y[i]+=vy[i]*dt+0.5*(Fty[i]/m[i])*(dt**2)
                        z[i]+=vz[i]*dt+0.5*(Ftz[i]/m[i])*(dt**2)

                        vx[i]+=(Ftx[i]/m[i])*dt
                        vy[i]+=(Fty[i]/m[i])*dt
                        vz[i]+=(Ftz[i]/m[i])*dt

                if step == (sMax-1):
                    file.write(str(x) + "," + str(y) + "," + str(z)+ "\n")


            #*************************** WORKER **************************

            elif taskid != MASTER:

                offset = comm.recv(source=MASTER, tag=FROM_MASTER)
                rows = comm.recv(source=MASTER, tag=FROM_MASTER)
                comm.Recv(x, source=MASTER, tag=FROM_MASTER)
                comm.Recv(y, source=MASTER, tag=FROM_MASTER)
                comm.Recv(z, source=MASTER, tag=FROM_MASTER)

                #Builds bottom left half of nxn force array
                #This ensures first worker finishes first as Master is dependent on this

                for i in range(offset,(offset+rows)):
                    for j in range(0,i+1):
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

                            #adds top triangle half of matrix
                            #halves the number of caluclations needed
                            Fx[j,i] = -Fx[i,j]
                            Fy[j,i] = -Fy[i,j]
                            Fz[j,i] = -Fz[i,j]

                        if i == j:
                            Fx[i,j] = 0
                            Fy[i,j] = 0
                            Fz[i,j] = 0

                comm.send(offset, dest=MASTER, tag=FROM_WORKER)
                comm.send(rows, dest=MASTER, tag=FROM_WORKER)
                comm.Send(Fx, dest=MASTER, tag=FROM_WORKER)
                comm.Send(Fy, dest=MASTER, tag=FROM_WORKER)
                comm.Send(Fz, dest=MASTER, tag=FROM_WORKER)


            step+=1

        plotstep+=1

    final = time.time()
    totaltime = final-initial

    #WARNING: This sometimes overwrites first line of text file so disregard first data points
    file.write("Total elapsed time: " + str(totaltime))

    #print("Elapsed time: {:8.6f} s".format(final-initial))

Force(n,x,y,z,vx,vy,vz,dt,e)
