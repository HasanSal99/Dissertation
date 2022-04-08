#!/usr/bin/python

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

class MSI():
    """Multi-species streaming instability class

    Class for calculating growth rates of the multi-species streaming instability. Follows notation of Benitez-Llambay et al. (2019, BLKP19).

    Args:
        eps: array of dust-to-gas ratios, one element for each species
        Ts: array of stopping times, one element for each species
        h0 (optional): aspect ratio of the disc. Defaults to 0.05.
        cs_over_eta (optional): sound speed in dimensionless units is h0/cs_over_eta. Defaults to unity.
    """
    def __init__(self, eps, Ts, h0=0.05, cs_over_eta=1.0):
        # Convert to arrays
        self.Ts = np.atleast_1d(Ts)
        self.eps = np.atleast_1d(eps)
        self.h0 = h0
        self.soundspeed = h0/cs_over_eta

        # Calculate equilibrium velocities using eqs (79)-(84) of BLKP19
        An = np.sum(self.eps*self.Ts/(1 + self.Ts*self.Ts))
        Bn = np.sum(self.eps/(1 + self.Ts*self.Ts)) + 1.0

        chi0 = 2*self.h0*self.h0
        psi = 1/(An*An + Bn*Bn)

        # Equilibrium gas velocity (units: Keplerian velocity box!)
        self.vgx = An*chi0*psi
        self.vgy = -0.5*Bn*chi0*psi

        # Equilibrium dust velocity (note: vectors!)
        self.vdx = (self.vgx + 2*self.Ts*self.vgy)/(1 + self.Ts*self.Ts)
        self.vdy = (self.vgy - 0.5*self.Ts*self.vgx)/(1 + self.Ts*self.Ts)

    def matrix(self, Kx, Kz):
        """Construct MSI matrix

        Args:
            Kx: non-dimensional wave number x
            Kz: non-dimensional wave number z
        """

        N = len(self.Ts)    # Number of dust species

        # Matrix: 4 equations per species (and gas!)
        M = np.zeros((4*N + 4, 4*N + 4), dtype=complex)

        # Scale velocities (see appendix E of BLKP19)
        vgx = self.vgx/self.h0**2
        vgy = self.vgy/self.h0**2
        vdx = self.vdx/self.h0**2
        vdy = self.vdy/self.h0**2

        # Relative velocities
        dvx = vgx - vdx
        dvy = vgy - vdy

        # Gas continuity equation
        M[0, 0] = 1j*Kx*vgx
        M[0, 1] = 1j*Kx
        M[0, 3] = 1j*Kz

        # Gas momentum x
        M[1, 0] = 1j*Kx/self.soundspeed**2 - np.sum(self.eps*vdx/self.Ts)
        M[1, 1] = 1j*Kx*vgx + np.sum(self.eps/self.Ts)
        M[1, 2] = -2
        M[1, 4:4*N+4:4] = dvx/self.Ts
        M[1, 5:4*N+4:4] = -self.eps/self.Ts

        # Gas momentum y
        M[2, 0] = -np.sum(self.eps*vdy/self.Ts)
        M[2, 1] = 0.5
        M[2, 2] = 1j*Kx*vgx + np.sum(self.eps/self.Ts)
        M[2, 4:4*N+4:4] = dvy/self.Ts
        M[2, 6:4*N+4:4] = -self.eps/self.Ts

        # Gas momentum z
        M[3, 0] = 1j*Kz/self.soundspeed**2
        M[3, 3] = 1j*Kx*vgx + np.sum(self.eps/self.Ts)
        M[3, 7:4*N+4:4] = -self.eps/self.Ts

        # Deal with all dust species
        for j in range(0, N):
            i = 4*j + 4

            # Dust continuity
            M[i, i] = 1j*Kx*vdx[j]
            M[i, i+1] = 1j*Kx*self.eps[j]
            M[i, i+3] = 1j*Kz*self.eps[j]

            # Dust momentum x
            M[i+1, 1] = -1/self.Ts[j]
            M[i+1, i+1] = 1j*Kx*vdx[j] + 1/self.Ts[j]
            M[i+1, i+2] = -2

            # Dust momentum y
            M[i+2, 2] = -1/self.Ts[j]
            M[i+2, i+1] = 0.5
            M[i+2, i+2] = 1j*Kx*vdx[j] + 1/self.Ts[j]

            # Dust momentum z
            M[i+3, 3] = -1/self.Ts[j]
            M[i+3, i+3] = 1j*Kx*vdx[j] + 1/self.Ts[j]

        # Return matrix
        return M

    def eigvals(self, Kx, Kz):
        """Calculate eigenvalues of MSI matrix

        Args:
            Kx: non-dimensional wave number x
            Kz: non-dimensional wave number z
        """
        return linalg.eigvals(self.matrix(Kx, Kz))

    def eig(self, Kx, Kz):
        """Calculate eigenvalues and eigenvectors of MSI matrix

        Args:
            Kx: non-dimensional wave number x
            Kz: non-dimensional wave number z
        """

        return linalg.eig(self.matrix(Kx, Kz), left=False, right=True)

    def max_growth(self, Kx, Kz, eigenvector=False):
        """Calculate maximum growth rate

        Args:
            Kx: non-dimensional wave number x (may be an array)
            Kz: non-dimensional wave number z (may be an array)
        """
        Kx = np.asarray(Kx)
        Kz = np.asarray(Kz)

        # Make sure we can handle both vector and scalar K's
        scalar_input = False
        if Kx.ndim == 0:
            Kx = Kx[None]  # Makes 1D
            Kz = Kz[None]  # Makes 1D
            scalar_input = True
        else:
            original_shape = np.shape(Kx)
            Kx = np.ravel(Kx)
            Kz = np.ravel(Kz)

        ret = np.zeros(len(Kx), dtype=complex)
        evector = []
        # Calculate maximum growth rate for each K
        for i in range(0, len(Kx)):
            ev = self.eigvals(Kx[i], Kz[i])
            ret[i] = ev[ev.real.argmin()]

            if eigenvector is True:
                evalue, evector = self.eig(Kx[i], Kz[i])
                evector = evector[:,ev.real.argmin()]/evector[4,ev.real.argmin()]

        # Return value of original shape
        if eigenvector is False:
            if scalar_input:
                return np.squeeze(ret)

            return np.reshape(ret, original_shape)

        if scalar_input:
            return np.squeeze(ret), evector

        return np.reshape(ret, original_shape), evector


#----------------
#----------------
    
def epsilons_calculator(Ts1,Ts2):
    epsilon1=(Ts2**0.5)/((Ts1**0.5)+(Ts2**0.5))
    epsilon2=(Ts1**0.5)/((Ts2**0.5)+(Ts1**0.5))
    return (epsilon1,epsilon2)


Ts1_Range=[10**-4,0.1] #Defines the range of Ts1, which we vary
Ts1s=np.logspace(np.log10(Ts1_Range[0]),np.log10(Ts1_Range[1]),1000) #Creates a list of evenly spaced out Ts1 values, in this case 1000.
Ts2=0.1 #constant we define
eps1,eps2=epsilons_calculator(Ts1s[0],Ts2) #calculating the other value
msi= MSI(eps=[eps1,eps2],Ts=[Ts1s[0],Ts2]) 
eigen=msi.max_growth(50,110) #calculates the eigen value

values=np.array([[Ts1s[0],Ts2,eps1,eps2,np.real(eigen),np.imag(eigen)]]) #creates an array. Final two elements are the real and imaginary part of the eigenvalue respectively.
"""We create the above array because the code does not allow an empty array to be created with a undefined dimensions. We calcualte all of the respected values, and then put them into an array with the appropriate amount of dimensions. After, we loop over the remaining 999 Ts1's."""
Ts1s=np.delete(Ts1s,0) #Removes the first element of the range.
for Ts1 in Ts1s:
    """This loop does the previous thing over and over again for every value of Ts1."""
    eps1,eps2=epsilons_calculator(Ts1,Ts2)
    msi= MSI(eps=[eps1,eps2],Ts=[Ts1,Ts2])
    eigen=msi.max_growth(50,110)
    values=np.append(values,[[Ts1,Ts2,eps1,eps2,np.real(eigen),np.imag(eigen)]],0)

stopping_times_1=[] 
imag_parts_1=[] 
real_parts_1=[] 
for event in values:
    if event[4]<0:
        real_part=event[4]*-1
        real_parts_1.append(real_part)
        stopping_times_1.append(event[0])
        imag_parts_1.append(event[5])

msi= MSI(eps=[eps1,eps2],Ts=[Ts1s[0],Ts2]) 
eigen=msi.max_growth(280,190) #calculates the eigen value

values_1=np.array([[Ts1s[0],Ts2,eps1,eps2,np.real(eigen),np.imag(eigen)]]) #creates an array. Final two elements are the real and imaginary part of the eigenvalue respectively.
Ts1s=np.delete(Ts1s,0) #Removes the first element of the range.
for Ts1 in Ts1s:
    """This loop does the previous thing over and over again for every value of Ts1."""
    eps1,eps2=epsilons_calculator(Ts1,Ts2)
    msi= MSI(eps=[eps1,eps2],Ts=[Ts1,Ts2])
    eigen=msi.max_growth(280,190)
    values_1=np.append(values_1,[[Ts1,Ts2,eps1,eps2,np.real(eigen),np.imag(eigen)]],0)

#print (values)
stopping_times_2=[]
imag_parts_2=[]
real_parts_2=[]
for event in values_1:
    if event[4]<0:
        real_part=event[4]*-1
        real_parts_2.append(real_part)
        stopping_times_2.append(event[0])
        imag_parts_2.append(event[5])

msi= MSI(eps=[eps1,eps2],Ts=[Ts1s[0],Ts2]) 
eigen=msi.max_growth(10,5) #calculates the eigen value

values_2=np.array([[Ts1s[0],Ts2,eps1,eps2,np.real(eigen),np.imag(eigen)]]) #creates an array. Final two elements are the real and imaginary part of the eigenvalue respectively.
Ts1s=np.delete(Ts1s,0) #Removes the first element of the range.
for Ts1 in Ts1s:
    """This loop does the previous thing over and over again for every value of Ts1."""
    eps1,eps2=epsilons_calculator(Ts1,Ts2)
    msi= MSI(eps=[eps1,eps2],Ts=[Ts1,Ts2])
    eigen=msi.max_growth(10,5)
    values_2=np.append(values_2,[[Ts1,Ts2,eps1,eps2,np.real(eigen),np.imag(eigen)]],0)
#print (values)
stopping_times_3=[] #20, 0.6
imag_parts_3=[] #20, 0.6
real_parts_3=[] #20, 0.6
for event in values_2:
    if event[4]<0:
        real_part=event[4]*-1
        real_parts_3.append(real_part)
        stopping_times_3.append(event[0])
        imag_parts_3.append(event[5])
#---
msi= MSI(eps=[eps1,eps2],Ts=[Ts1s[0],Ts2]) 
eigen=msi.max_growth(8,750) #calculates the eigen value

values_3=np.array([[Ts1s[0],Ts2,eps1,eps2,np.real(eigen),np.imag(eigen)]]) #creates an array. Final two elements are the real and imaginary part of the eigenvalue respectively.
Ts1s=np.delete(Ts1s,0) #Removes the first element of the range.
for Ts1 in Ts1s:
    """This loop does the previous thing over and over again for every value of Ts1."""
    eps1,eps2=epsilons_calculator(Ts1,Ts2)
    msi= MSI(eps=[eps1,eps2],Ts=[Ts1,Ts2])
    eigen=msi.max_growth(8,750)
    values_3=np.append(values_3,[[Ts1,Ts2,eps1,eps2,np.real(eigen),np.imag(eigen)]],0)
#print (values)
stopping_times_4=[]
imag_parts_4=[]
real_parts_4=[] 
for event in values_3:
    if event[4]<0:
        real_part=event[4]*-1
        real_parts_4.append(real_part)
        stopping_times_4.append(event[0])
        imag_parts_4.append(event[5])

#---

msi= MSI(eps=[eps1,eps2],Ts=[Ts1s[0],Ts2]) 
eigen=msi.max_growth(5,5) #calculates the eigen value

values_4=np.array([[Ts1s[0],Ts2,eps1,eps2,np.real(eigen),np.imag(eigen)]]) #creates an array. Final two elements are the real and imaginary part of the eigenvalue respectively.
Ts1s=np.delete(Ts1s,0) #Removes the first element of the range.
for Ts1 in Ts1s:
    """This loop does the previous thing over and over again for every value of Ts1."""
    eps1,eps2=epsilons_calculator(Ts1,Ts2)
    msi= MSI(eps=[eps1,eps2],Ts=[Ts1,Ts2])
    eigen=msi.max_growth(5,5)
    values_4=np.append(values_4,[[Ts1,Ts2,eps1,eps2,np.real(eigen),np.imag(eigen)]],0)
#print (values)
stopping_times_5=[]
imag_parts_5=[]
real_parts_5=[] 
for event in values_4:
    if event[4]<0:
        real_part=event[4]*-1
        real_parts_5.append(real_part)
        stopping_times_5.append(event[0])
        imag_parts_5.append(event[5])


fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_xscale('log')
#ax2.set_yscale('log')
fig.suptitle("Real and Imaginary parts of the the Eigen Value with varying stopping times")
ax1.plot(stopping_times_1,real_parts_1,label="15,7",)
#ax1.plot(stopping_times_2,real_parts_2,label="100,750")
ax1.plot(stopping_times_3,real_parts_3,label="8,750")
ax2.plot(stopping_times_1,imag_parts_1,label="15,7")
#ax2.plot(stopping_times_2,imag_parts_2,label="100,750")
ax2.plot(stopping_times_3,imag_parts_3,label="8,750")
ax1.set_xlabel('Stopping Time')
ax1.set_ylabel('Real part of Eigen Value')
ax2.set_xlabel('Stopping Time')
ax2.set_ylabel('Imaginary part of Eigen Value')
params = {'mathtext.default': 'regular' }
ax1.legend(title='$K_{x},K_{z}$')
ax2.legend(title='$K_{x},K_{z}$')
ax1.grid()
ax2.grid()
plt.show()

Ts1_Range=[10**-4,0.1] #Defines the range of Ts1, which we vary
Ts1s=np.logspace(np.log10(Ts1_Range[0]),np.log10(Ts1_Range[1]),1000) #Creates a list of evenly spaced out Ts1 values, in this case 1000.

stopping_times_6=[]  #Array for 5,5 Wave Number
real_parts_6=[]
for Ts1 in Ts1s:
    eps1,eps2=epsilons_calculator(Ts1,Ts2)
    msi= MSI(eps=[eps1,eps2],Ts=[Ts1,Ts2]) 
    eigenvalues=msi.eigvals(280,190) 
    for i in range(len(eigenvalues)):
        stopping_times_6.append(Ts1)
        real_parts_6.append(-1*np.real(eigenvalues[i]))

stopping_times_7=[]  #Array for 5,5 Wave Number
real_parts_7=[]
for Ts1 in Ts1s:
    eps1,eps2=epsilons_calculator(Ts1,Ts2)
    msi= MSI(eps=[eps1,eps2],Ts=[Ts1,Ts2]) 
    eigenvalues=msi.eigvals(50,110) 
    for i in range(len(eigenvalues)):
        stopping_times_7.append(Ts1)
        real_parts_7.append(-1*np.real(eigenvalues[i]))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3, figsize=(12, 6))
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(10**(-6),10**(0))
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(10**(-6),10**(0))
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_ylim(10**(-6),10**(0))
fig.suptitle("Growth Rates Two Particles each in two different regions")
ax1.plot(stopping_times_1,real_parts_1,label="50,110", color='blue')
ax1.plot(stopping_times_2,real_parts_2,label="280,190", color='red', linestyle="dashed")
#ax1.plot(stopping_times_3,real_parts_3,label="20,0.6", color='green')
#ax1.plot(stopping_times_4,real_parts_4,label="25,0.5", color='orange', linestyle='dashed')
ax2.plot(stopping_times_6,real_parts_6,label="280,190", linestyle='None', marker='.', markersize=3, color='Red')
ax3.plot(stopping_times_7,real_parts_7,label="50,110",linestyle='None', marker='.', markersize=3, color='Blue')
ax1.set_xlabel('Stopping Time')
ax1.set_ylabel('Real part')
ax2.set_xlabel('Stopping Time')
ax2.set_ylabel('Eigen Values')
ax3.set_xlabel('Stopping Time')
ax3.set_ylabel('Eigen Values')
params = {'mathtext.default': 'regular' }
ax1.legend(title='$K_{x},K_{z}$')
ax2.legend(title='$K_{x},K_{z}$')
ax3.legend(title='$K_{x},K_{z}$')
ax1.grid()
ax2.grid()
ax3.grid()
plt.show()


# Test Case 1: single dust fluid,
#LinA from Table 4 in BLKP19
msi = MSI(eps=[3.0], Ts=[0.1])
print(msi.max_growth(30,30))
# Test Case 2: single dust fluid, LinB from Table 4
msi = MSI(eps=[0.2], Ts=[0.1])
print(msi.max_growth(6,6))
# Test Case 3 : Two Dust fluid, LinC From Table 4
msi = MSI(eps=[1,0.5], Ts=[0.0425,0.1])
print (msi.max_growth(50,50))

#recorded_values=[-0.4190091323+0.3480181522i,-0.0154862262-0.4998787515i,-0.3027262829-0.3242790653i]
