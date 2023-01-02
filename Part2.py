import pickle 
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import time 


#Function to initialize the velocities

def InitialVelocities(Temp):
    
    v=np.zeros([N,3]) 
    
    for i in range(0,N,1):
        
        v[i,0]=np.random.choice([1,-1])*np.random.normal(np.sqrt(Temp),1)
        v[i,1]=np.random.choice([1,-1])*np.random.normal(np.sqrt(Temp),1)
        v[i,2]=np.random.choice([1,-1])*np.random.normal(np.sqrt(Temp),1)
          
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
    pre=0
    
    for i in range(N):
        
        for j in range(i+1, N):
            
            rij = pbc(r[i]-r[j], L)
            d2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
            d4=d2*d2; d6=d4*d2; d8=d6*d2; d12=d6*d6; d14=d8*d6

            if d2 < cutoff2:
                
                aux = (48 / d14 - 24 /d8)*rij
             
                F[i] = F[i] + aux
                F[j] = F[j] - aux
                
                pot = pot + 4*( 1/ d12 - 1/d6)- 4*(1/cutoff**12 - 1/cutoff**6)
                
                pre += np.abs(np.sum(aux*rij))
                
   
    return np.array(F), pot, pre


#Euler algorithm

def time_step_Euler(r, vel, L):
    F, pot = find_force_LJ0(r, L)
    r = r + vel * dt + 0.5* F * dt*dt
    r = pbc(r, L)   
                           
    vel += F * dt

    kin = 0.5 * np.sum(vel**2)
    
    return r, vel,F , pot, kin

#Velocity Verlet algorithm

def time_step_vVerlet2(r, vel, F, L):
    
    r = r + vel * dt + 0.5* F * dt*dt
    r=pbc(r,L)
    vel += F* 0.5 * dt
    F, pot, pre = find_force_LJ0(r, L)
    vel += F* 0.5 * dt

    return r, vel, F, pot


#-----------------------------------------------------------------------------------

#Parameters

kb=1.38*10**-26
Na=6.022*10**23
m=40
M=5
N=M**3
Temp=1.2
nu=0.1
sigma = np.sqrt(Temp)
dt=0.0001
Ntimesteps=110000
rho=[0.05,0.1,0.2,0.4,0.6,0.8]
epsilon=0.998
sigma2=3.4

POTf_list=[]
KINf_list=[]
E_tf_list=[]
Pressiof_list=[]
np.random.seed(234)

for b in range(0,len(rho),1):  #Loop for the densities
    
    
    L=(N/rho[b])**(1/3)
    a=L/M
    cutoff=L/2
    POT_list=[]
    KIN_list=[]
    E_t_list=[]
    Pressio_list=[]

    r = lc_lattice(M, a)
    vel=np.zeros([N,3])
    F, pot, pre = find_force_LJ0(r, L)
    

    for t in range(Ntimesteps): #Loop for the steps
        
    
        r, vel, F, pot = time_step_vVerlet2(r, vel, F, L)
        
        #Changing to the units that we want
        
        pot2=pot*epsilon
        
        kin=0.5*np.sum(vel**2)*epsilon
        
        E_t=pot2+kin
        
        Tinst=2/(3*N)*kin
        
        
        pressio=(rho[b]*Tinst+1/L**3*pre)*epsilon*10**3*125/(Na*(sigma2*10**-10)**3)
            
        vel = therm_Andersen(vel, nu, sigma)
    
        
        if Ntimesteps>10000:  #We wait 10000 steps to reach equilibrium
        
            POT_list.append(pot)
            KIN_list.append(kin)
            E_t_list.append(E_t)
            Pressio_list.append(pressio)
        
    POTf_list.append(np.mean(POT_list))
    KINf_list.append(np.mean(KIN_list))
    E_tf_list.append(np.mean(E_t_list))
    Pressiof_list.append(np.mean(Pressio_list))
 
rho2=np.zeros(len(rho))
    
for kk in range(0,len(rho),1): 
    
    rho2[kk]=rho[kk]*m/((sigma2*10**-8)**3*Na)
    

#Plots

plt.plot(rho2, E_tf_list, color='red')
plt.ylabel('Total Energy (kJ/mol)')
plt.xlabel('Density (g/cm^3)')
plt.show()
plt.plot(rho2, POTf_list, color='blue')
plt.ylabel('Potential Energy (kJ/mol)')
plt.xlabel('Density (g/cm^3)')
plt.show()
plt.plot(rho2, KINf_list, color='green')
plt.ylabel('Kinetic Energy (kJ/mol)')
plt.ylim(150,250)
plt.xlabel('Density (g/cm^3)')
plt.show()
plt.plot(rho2, Pressiof_list, color='yellow')
plt.ylabel('Pressure (Pa)')
plt.xlabel('Density (g/cm^3)')
plt.show()









