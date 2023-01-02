import pickle 
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math as m
import os 
import copy



np.random.seed(234)

#Function to initialize the velocities

def InitialVelocities(Temp):
    
    v=np.zeros([N,3])
    
    
    for i in range(0,N,1):
        
        v[i,0]=np.random.choice([1,-1])*np.random.normal(np.sqrt(Temp),1)
        v[i,1]=np.random.choice([1,-1])*np.random.normal(np.sqrt(Temp),1)
        v[i,2]=np.random.choice([1,-1])*np.random.normal(np.sqrt(Temp),1)
        HistIni.append(v[i,0])
    
    return v
    
#Function to initialize the positions

def lc_lattice(M, a):
    pos = []
    for nx in range(M):
        for ny in range(M):
            for nz in range(M):
                pos.append([nx,  ny, nz])
    return np.array(pos)*a

#Function to establish a thermal bath

def therm_Andersen(vel, nu, sigma):
    
    for n in range(N):

        if np.random.ranf() < nu:
 
            vel[n] = np.random.normal(0,sigma,3)

    return vel

#Periodic boundary conditions

def pbc(vec, L):
    vec = vec - np.rint(vec / L) * L
    return np.array(vec)
            
#Function to modelize the interactions of particles under a Lennard-Jones potential

def find_force_LJ0(r, L):
    
    
    N=len(r)
    F = np.zeros((N,3))
    pot = 0.0
    cutoff2=cutoff*cutoff
    
    for i in range(N):
        
        for j in range(i+1, N):
            
            rij = pbc(r[i]-r[j], L)
            d2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
            d4=d2*d2; d6=d4*d2; d8=d6*d2; d12=d6*d6; d14=d8*d6

            if d2 < cutoff2:
             
                F[i] = F[i] + (48 / d14 - 24 /d8)*rij
                F[j] = F[j] - (48 / d14 - 24 /d8)*rij
                
                pot = pot + 4*( 1/ d12 - 1 /d6)- 4*( 1/cutoff**12 - 1/cutoff**6)
   
    return np.array(F), pot

#Euler algorithm

def time_step_Euler(r, vel, F, L):
    
    F, pot = find_force_LJ0(r, L)
    r = r + vel * dt[l] + 0.5* F * dt[l]*dt[l]
    r = pbc(r, L)   
                           
    vel += F * dt[l]

    kin = 0.5 * np.sum(vel**2)
    
    return r, vel, F, pot, kin

#Velocity Verlet algorithm

def time_step_vVerlet(r, vel, F, L):
    
    r = r + vel * dt[l] + 0.5* F * dt[l]*dt[l]
    r=pbc(r,L)
    vel += F* 0.5 * dt[l]
    F, pot = find_force_LJ0(r, L)
    vel += F* 0.5 * dt[l]
    kin=0.5*np.sum(vel**2)

    return r, vel, F, pot, kin


#------------------------------------------------------------------------------

#Parameters

M=5
N=M**3
rho=0.7
L=(N/rho)**(1/3)
a=L/M
Temp=100
nu=0.1
sigma = np.sqrt(Temp)
dt=[0.001, 0.0001]
cutoff=L/2
Ntimesteps=1000
HistIni=[]

for l in range(0,len(dt),1):  #Loop for the timesteps
    
    np.random.seed(234)
    vel = InitialVelocities(Temp)
    vel = InitialVelocities(Temp)
    r = lc_lattice(M, a)
    F, pot = find_force_LJ0(r, L)
    E_t_list=[]
    Po_list=[]
    temps=0
    Temps=[]
    HistVx=[]
    
    for t in range(Ntimesteps): #Loop for the steps
        
        temps=temps+dt[l]
        
        Temps.append(temps)
        
        r, vel, F, pot, kin = time_step_vVerlet(r, vel, F, L)
    
        E_t=pot+kin
   
        E_t_list.append(E_t)
        
        
        P=np.abs(np.sum(vel))
        
        
        Po_list.append(P)
        
            

    Tinst=2/(3*N)*kin
    
    #Plots of the total energy and total momentum
    
    plt.plot(Temps, E_t_list, color='red')
    plt.ylim(18230,18350)
    plt.ylabel('Total Energy (E_t)')
    plt.xlabel("t")
    plt.show()
    
    plt.plot(Temps, Po_list, color='blue')
    plt.ylim(80,120)
    plt.ylabel('Total Momentum (P)')
    plt.xlabel("t")
    plt.show()
    
     
    #Histograms for the initial and final distribution of velocities


    for i in range(0,len(vel),1):
    
        HistVx.append(vel[i,0])


    plt.hist(HistIni, bins=50, color='red', density=True)
    plt.xlabel('Velocity (V)')
    plt.ylabel('Inital dist. of frequencies')
    plt.show()
    

    plt.hist(HistVx, bins=50, color='blue', density=True)
    plt.xlabel('Velocity (V)')
    plt.ylabel('Final dist. of frequencies')
    plt.show()
