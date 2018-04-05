import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
#import seaborn as sns
import matplotlib.colors as clr
import scipy.sparse.linalg

# For state a=(i_1,i_2,...,i_M), this function returns the "position" of this state in a list with all states lexicographically ordered. See Section 1.1 in Supplementary Material.
def Pos_R0(a,Ns,M,k):
	position=0
	for i in range(1,M+1):
		prod=1
		for h in range(i+1,M+1):
			if h!=k:
				prod=prod*(Ns[h-1]+1)
			else:
				prod=prod*Ns[h-1]
		if i!=k:	
			position+=a[i-1]*prod
		if i==k:
			position+=(a[i-1]-1)*prod
	return int(position)

# This is the construction of matrix D^{(j)} in Supplementary Material, Section 1.1
def Fill_Matrix_A(A,l,a,n,M,Ns,k,Gama,Betas,Lambdas):
	if l<M:
		if k==l:
			ini=1
		else:
			ini=0
		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			Fill_Matrix_A(A,l+1,a,n,M,Ns,k,Gama,Betas,Lambdas)
			a[l-1]=ini
	else:
		if k==l:
			ini=1
		else:
			ini=0
		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			#print(a)
			denominator=delta
			for h in range(1,M+1):
				if h!=k:
					denominator+=a[h-1]*Gama[h-1]+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
				else:
					denominator+=(a[h-1]-1)*Gama[h-1]+Gama[h-1]/(1.0-phi)+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
			for h in range(1,M+1):
				if h!=k:
					if a[h-1]>0:
						new=np.append([],a)
						new[h-1]-=1
						A[Pos_R0(a,Ns,M,k),Pos_R0(tuple(new),Ns,M,k)]+=Gama[h-1]*a[h-1]/denominator
				else:
					if a[h-1]>0:
						new=np.append([],a)
						new[h-1]-=1
						A[Pos_R0(a,Ns,M,k),Pos_R0(tuple(new),Ns,M,k)]+=Gama[h-1]*(a[h-1]-1)/denominator
				if a[h-1]<Ns[h-1]:
					new=np.append([],a)
					new[h-1]+=1
					A[Pos_R0(a,Ns,M,k),Pos_R0(tuple(new),Ns,M,k)]+=Lambdas[h-1]*(Ns[h-1]-a[h-1])/denominator
					for p in range(1,M+1):
						if p!=k:
							A[Pos_R0(a,Ns,M,k),Pos_R0(tuple(new),Ns,M,k)]+=Betas[p-1,h-1]*a[p-1]*(Ns[h-1]-a[h-1])/denominator
						else:
							A[Pos_R0(a,Ns,M,k),Pos_R0(tuple(new),Ns,M,k)]+=Betas[p-1,h-1]*(a[p-1]-1)*(Ns[h-1]-a[h-1])/denominator
			a[l-1]=ini

# This function constructs vector e^{(j)}(n) in Supplementary Material, Section 1.1
def Fill_Vector_b(b,l,a,n,M,Ns,k,Gama,Betas,Lambdas,Xis):
	if l<M:
		if k==l:
			ini=1
		else:
			ini=0

		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			Fill_Vector_b(b,l+1,a,n,M,Ns,k,Gama,Betas,Lambdas,Xis)
			a[l-1]=ini
	else:
		if k==l:
			ini=1
		else:
			ini=0

		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			denominator=delta
			for h in range(1,M+1):
				if h!=k:
					denominator+=a[h-1]*Gama[h-1]+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
				else:
					denominator+=(a[h-1]-1)*Gama[h-1]+Gama[h-1]/(1.0-phi)+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
						
			if n>0:
				for h in range(1,M+1):
					if a[h-1]<Ns[h-1]:
						new=np.append([],a)
						new[h-1]+=1
						b[Pos_R0(a,Ns,M,k),0]+=Betas[k-1,h-1]*(Ns[h-1]-a[h-1])*Xis[n-1][Pos_R0(tuple(new),Ns,M,k),0]/denominator
			else:
				b[Pos_R0(a,Ns,M,k),0]+=(Gama[k-1]/(1.0-phi))/denominator
				b[Pos_R0(a,Ns,M,k),0]+=delta/denominator
			a[l-1]=ini

# Case Study 4. 3 patients in Room 1; 2 patients in Rooms 2, 3 and 4
N=3+2+2+2
M=4
Ns=[3,2,2,2]

# Probability of admitted patient being colonized
phi = 0.01

# Transmission rate for patients within the same room
betaSR = 0.0366

# Transmission rate for patients at different rooms
betaDR = 0.0238

# Discharge rate
gamma = 0.1

# Spontaneous colonization rate
lamda = 0.0037

# Global detection rate. No detection considered, so delta=0
delta = 0

# This code computes the reproduction number of a Patient at Room 2. Figure 12 right

# Since patients at room 2 are at compartmental level 2, we set j=2. We are interested in computing R^{(j)}
j=2

# For initial state (0,...,0,1,0,...,0), where "1" corresponds to compartmental level j
init_state=[0]*M
init_state[j-1]=1

# This is the dimension of the system of equations to be solved. See Supplementary Material, Section 1.2
Num_States=1
for i in range(1,M+1):
	if i!=j:
		Num_States=Num_States*(Ns[i-1]+1)
	else:
		Num_States=Num_States*Ns[i-1]

# Since we are interested in computing the reproduction number R^{(j)}, we do not have any k.

# Since we can compute probabilities p(R^{(j)}=n), n=0,1,...., we can truncate the distribution so that total mass 0.9999 is accumulated
p_trunc=0.9999

precision=51

# The heatmap contains 51x51 values E[R^{(j)}], for different values of (betaSR,betaDR)
Gama=np.asmatrix(np.zeros((M,1)))
Betas=np.asmatrix(np.zeros((M,M)))
Lambdas=np.asmatrix(np.zeros((M,1)))

num_betaSRs=precision
num_betaDRs=precision

betaSRs=[]
betaDRs=[]

mean_R0_Patient_Room2=np.asmatrix(np.zeros((num_betaSRs,num_betaDRs)))

for betaSR_index in range(num_betaSRs):
	print(betaSR_index)
	betaSR=0.01+betaSR_index*(0.06-0.01)/(num_betaSRs-1.0)
	betaSRs=np.append(betaSRs,betaSR)
	
	for betaDR_index in range(num_betaDRs):
		print(betaDR_index)
		betaDR=0.01+betaDR_index*(0.06-0.01)/(num_betaDRs-1.0)
		betaDRs=np.append(betaDRs,betaDR)

		Gama[0]=(1-phi)*gamma
		Gama[1]=(1-phi)*gamma
		Gama[2]=(1-phi)*gamma
		Gama[3]=(1-phi)*gamma

		Lambdas[0]=lamda
		Lambdas[1]=lamda
		Lambdas[2]=lamda
		Lambdas[3]=lamda

		Betas[0,0]=betaSR
		Betas[0,1]=betaDR
		Betas[0,2]=betaDR
		Betas[0,3]=betaDR
		
		Betas[1,1]=betaSR
		Betas[1,0]=betaDR
		Betas[1,2]=betaDR
		Betas[1,3]=betaDR
		
		Betas[2,2]=betaSR
		Betas[2,1]=betaDR
		Betas[2,0]=betaDR
		Betas[2,3]=betaDR
		
		Betas[3,3]=betaSR
		Betas[3,1]=betaDR
		Betas[3,2]=betaDR
		Betas[3,0]=betaDR

		mean=0
		mean2=0

		suma=0

		Xis=[np.asmatrix(np.zeros((Num_States,1))) for n in range(501)]
		Identity=np.asmatrix(np.eye(Num_States))

		n=-1
		while(suma<p_trunc and n<500):
			n+=1
			A=np.asmatrix(np.zeros((Num_States,Num_States)))
			b=np.asmatrix(np.zeros((Num_States,1)))
			
			a=[0]*M
			
			l=1
			if j==l:
				ini=1
			else:
				ini=0

			for il in range(ini,Ns[l-1]+1):
				a[l-1]=il
				Fill_Matrix_A(A,l+1,a,n,M,Ns,j,Gama,Betas,Lambdas)
				Fill_Vector_b(b,l+1,a,n,M,Ns,j,Gama,Betas,Lambdas,Xis)
				a[l-1]=ini

			Xis[n]=np.linalg.solve(Identity-A,b)
			
			suma+=Xis[n][Pos_R0(tuple(init_state),Ns,M,j),0]
			mean+=n*Xis[n][Pos_R0(tuple(init_state),Ns,M,j),0]
			if suma>p_trunc:
				break
		mean_R0_Patient_Room2[num_betaDRs-1-betaDR_index,betaSR_index]=mean

fig = plt.figure()
ax = fig.add_subplot(111)
cax = plt.imshow(mean_R0_Patient_Room2, cmap='hot', interpolation='nearest',extent=[betaDRs[0],betaDRs[num_betaDRs-1],betaSRs[0],betaSRs[num_betaSRs-1]],aspect="auto")
barra = fig.colorbar(cax)
barra.set_label(r'$E[R_{(0,1,0,0)}^{(2)}]$', fontsize=15)

ax.set_xlabel(r'$\beta_{SR}$', fontsize=15)
ax.set_ylabel(r'$\beta_{DR}$', fontsize=15)

plt.savefig('Figure12_Right.png')
