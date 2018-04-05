import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
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


# This is the construction of matrix D^{(j)}(k) in Supplementary Material, Section 1.2
def Fill_Matrix_A_j_k(A,l,a,n,M,Ns,j,k,Gama,Betas,Lambdas):
	if l<M:
		if j==l:
			ini=1
		else:
			ini=0
		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			Fill_Matrix_A_j_k(A,l+1,a,n,M,Ns,j,k,Gama,Betas,Lambdas)
			a[l-1]=ini
	else:
		if j==l:
			ini=1
		else:
			ini=0
		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			denominator=delta
			for h in range(1,M+1):
				if h!=j:
					denominator+=a[h-1]*Gama[h-1]+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
				else:
					denominator+=(a[h-1]-1)*Gama[h-1]+Gama[h-1]/(1.0-phi)+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
			for h in range(1,M+1):
				if h!=j:
					if a[h-1]>0:
						new=np.append([],a)
						new[h-1]-=1
						A[Pos_R0(a,Ns,M,j),Pos_R0(tuple(new),Ns,M,j)]+=Gama[h-1]*a[h-1]/denominator
				else:
					if a[h-1]>0:
						new=np.append([],a)
						new[h-1]-=1
						A[Pos_R0(a,Ns,M,j),Pos_R0(tuple(new),Ns,M,j)]+=Gama[h-1]*(a[h-1]-1)/denominator
				if a[h-1]<Ns[h-1]:
					new=np.append([],a)
					new[h-1]+=1
					A[Pos_R0(a,Ns,M,j),Pos_R0(tuple(new),Ns,M,j)]+=Lambdas[h-1]*(Ns[h-1]-a[h-1])/denominator
					for p in range(1,M+1):
						if p!=j:
							A[Pos_R0(a,Ns,M,j),Pos_R0(tuple(new),Ns,M,j)]+=Betas[p-1,h-1]*a[p-1]*(Ns[h-1]-a[h-1])/denominator
						else:
							if h==k:
								A[Pos_R0(a,Ns,M,j),Pos_R0(tuple(new),Ns,M,j)]+=Betas[p-1,h-1]*(a[p-1]-1)*(Ns[h-1]-a[h-1])/denominator
							else:
								A[Pos_R0(a,Ns,M,j),Pos_R0(tuple(new),Ns,M,j)]+=Betas[p-1,h-1]*a[p-1]*(Ns[h-1]-a[h-1])/denominator
			a[l-1]=ini

# This function constructs vector e^{(j)}(k;n) in Supplementary Material, Section 1.2
def Fill_Vector_b_j_k(b,l,a,n,M,Ns,j,k,Gama,Betas,Lambdas,Xis):
	if l<M:
		if j==l:
			ini=1
		else:
			ini=0

		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			Fill_Vector_b_j_k(b,l+1,a,n,M,Ns,j,k,Gama,Betas,Lambdas,Xis)
			a[l-1]=ini
	else:
		if j==l:
			ini=1
		else:
			ini=0

		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			denominator=delta
			for h in range(1,M+1):
				if h!=j:
					denominator+=a[h-1]*Gama[h-1]+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
				else:
					denominator+=(a[h-1]-1)*Gama[h-1]+Gama[h-1]/(1.0-phi)+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
			if n>0:
				if a[k-1]<Ns[k-1]:
					new=np.append([],a)
					new[k-1]+=1
					b[Pos_R0(a,Ns,M,j),0]+=Betas[j-1,k-1]*(Ns[k-1]-a[k-1])*Xis[n-1][0,Pos_R0(tuple(new),Ns,M,j)]/denominator
			else:
				b[Pos_R0(a,Ns,M,j),0]+=(Gama[j-1]/(1.0-phi))/denominator
				b[Pos_R0(a,Ns,M,j),0]+=delta/denominator
			a[l-1]=ini

# Case Study 3. 20 patients, 5 HCWs and 100 surfaces
N=20+5+100
M=3
Ns=[20,5,100]

# Probability of admitted patient being colonized
phi = 0.1

# HCW hand-washing rate
mu = 24.0

# Surface cleaning rate
kappa = 1.0

# HCW-to-patient colonization rate
betasp = 0.3

# Patient-to-HCW contamination rate
betaps = 2.0

# HCW-to-surface contamination rate
betase = 2.0 

# Surface-to-HCW contamination rate
betaes = 2.0

# Surface-to-patient colonization rate
betaep = 0.3

# Patient-to-surface contamination rate
betape = 2.0

# Colonized patient discharge rate
gammaprime = 0.05

# Non-colonized patient discharge rate
gamma = 0.1

# Global detection rate. No detection, so delta=0
delta = 0

# This code computes the reproduction number of a patient among Surfaces. Figure 8 right

# Since patients are at compartmental level 1, we set j=1. We are interested in computing R^{(j)}(k)
j=1

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

# Since we are interested in computing the reproduction number R^{(j)}(k) for a patient among Surfaces, we set k=3
k=3

# Since we can compute probabilities p(R^{(j)}(k)=n), n=0,1,...., we can truncate the distribution so that total mass 0.9999 is accumulated
p_trunc=0.9999

precision=21

# The heatmap contains 21x21 values E[R^{(j)}(k)], for different values of (beta_pe,gamma'^{-1})
num_betas_pe=precision
num_gammaprimes=precision

betas_pe=[]
gammaprimes=[]
inv_gammaprimes=[]

mean_R0_Patient_to_Surfaces=np.asmatrix(np.zeros((num_betas_pe,num_gammaprimes)))

for beta_pe_index in range(num_betas_pe):
	print(beta_pe_index)
	betape=1.5+beta_pe_index*(2.5-1.5)/(num_betas_pe-1.0)
	betas_pe=np.append(betas_pe,betape)
	
	for gammaprime_index in range(num_gammaprimes):
		print(gammaprime_index)
		inv_gammaprime=15.0+gammaprime_index*(25.0-15.0)/(num_gammaprimes-1.0)
		gammaprime=1.0/inv_gammaprime
		gammaprimes=np.append(gammaprimes,gammaprime)
		inv_gammaprimes=np.append(inv_gammaprimes,inv_gammaprime)

		# "Gama" vector contains rates \mu_j
		Gama=np.asmatrix(np.zeros((M,1)))

		# "Betas" matrix contains rates \lambda_kj
		Betas=np.asmatrix(np.zeros((M,M)))

		# "Lambdas" vector contains rates \lambda_j
		Lambdas=np.asmatrix(np.zeros((M,1)))

		# These vectors are filled according to functions described in Figure 7. See also Supplementary Material Table S6
		Gama[0]=gammaprime*(1-phi)
		Gama[1]=mu
		Gama[2]=kappa

		Lambdas[0]=phi*gamma

		Betas[0,1]=betaps/Ns[0]
		Betas[1,0]=betasp/Ns[1]
		Betas[0,2]=betape/Ns[0]
		Betas[2,0]=betaep/Ns[2]
		Betas[1,2]=betase/Ns[1]
		Betas[2,1]=betaes/Ns[2]

		mean_j_k=0
		mean2_j_k=0
		suma=0

		Xis_j_k=[np.asmatrix(np.zeros((Num_States,1))) for n in range(501)]
		Identity=np.asmatrix(np.eye(Num_States))

		A_j_k=np.zeros((Num_States,Num_States))
				
		a=[0]*M

		l=1
		if j==l:
			ini=1
		else:
			ini=0

		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			Fill_Matrix_A_j_k(A_j_k,l+1,a,0,M,Ns,j,k,Gama,Betas,Lambdas)
			a[l-1]=ini
				
		n=-1
		while(suma<p_trunc and n<500):
			n+=1
			
			b_j_k=np.asmatrix(np.zeros((Num_States,1)))
			
			a=[0]*M
			
			l=1
			if j==l:
				ini=1
			else:
				ini=0
				
			for il in range(ini,Ns[l-1]+1):
				a[l-1]=il
				Fill_Vector_b_j_k(b_j_k,l+1,a,n,M,Ns,j,k,Gama,Betas,Lambdas,Xis_j_k)
				a[l-1]=ini
			
			Xis_j_k[n]=np.asmatrix(scipy.sparse.linalg.spsolve(Identity-A_j_k,b_j_k))
			
			suma+=Xis_j_k[n][0,Pos_R0(tuple(init_state),Ns,M,j)]
			mean_j_k+=n*Xis_j_k[n][0,Pos_R0(tuple(init_state),Ns,M,j)]
			mean2_j_k+=n*n*Xis_j_k[n][0,Pos_R0(tuple(init_state),Ns,M,j)]
			
			print(n,suma)
			
			if suma>p_trunc:
				break
		mean_R0_Patient_to_Surfaces[num_betas_pe-1-beta_pe_index,gammaprime_index]=mean_j_k

fig = plt.figure()
ax = fig.add_subplot(111)
cax = plt.imshow(mean_R0_Patient_to_Surfaces, cmap='hot', interpolation='nearest',extent=[inv_gammaprimes[0],inv_gammaprimes[num_gammaprimes-1],betas_pe[0],betas_pe[num_betas_pe-1]],aspect="auto")
barra = fig.colorbar(cax)
barra.set_label('$E[R_{(1,0,0)}^{(1)}(3)]$', fontsize=15)

ax.set_xlabel('$\gamma^{-1}$', fontsize=15)
ax.set_ylabel(r'$\beta_{pe}$', fontsize=15)

plt.savefig('Figure8_Right.png')
