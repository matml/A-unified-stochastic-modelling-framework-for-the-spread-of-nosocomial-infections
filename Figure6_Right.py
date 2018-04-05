import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import random, math
import numpy as np
#import seaborn as sns
import matplotlib.colors as clr

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
				denominator+=a[h-1]*Gama[h-1]+(Ns[h-1]-a[h-1])*Lambdas[h-1]
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
				denominator+=a[h-1]*Gama[h-1]+(Ns[h-1]-a[h-1])*Lambdas[h-1]
				for p in range(1,M+1):
					denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
			if n>0:
				if a[k-1]<Ns[k-1]:
					new=np.append([],a)
					new[k-1]+=1
					b[Pos_R0(a,Ns,M,j),0]+=Betas[j-1,k-1]*(Ns[k-1]-a[k-1])*Xis[n-1][Pos_R0(tuple(new),Ns,M,j),0]/denominator
			else:
				b[Pos_R0(a,Ns,M,j),0]+=Gama[j-1]/denominator
				b[Pos_R0(a,Ns,M,j),0]+=delta/denominator
			a[l-1]=ini

# Case Study 2. 7 patients, 14 HCWs and 2 volunteers
N=7+14+2
M=3
Ns=[7,14,2]

# Probability of admitted patient being colonized
phi = 0.165

# HCW hand-washing rate
gammaH = 24.0

# Volunteer hand-washing rate
gammaV = 12.0

# Beta for HCW-patient
betaHp = 0.72

# Beta for Volunteer-patient
betaVp = 0.20

# HCW hygienic level
eta = 0.46

# Volunteer hygienic level
xi = 0.23

# Colonized patient discharge rate
deltaC = 1.0/13.0

# Non-colonized patient discharge rate
deltaU = 1.0/7.0

# Global detection rate. No detection considered in this model
delta = 0

# This code computes the reproduction number of a Volunteer among patients. Figure 6 left

# Since Volunteers are at compartmental level 3, we set j=3. We are interested in computing R^{(j)}(k)
j=3

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

# Since we are interested in computing the reproduction number R^{(j)}(k) for a Volunteer among patients, we set k=1
k=1

# Since we can compute probabilities p(R^{(j)}(k)=n), n=0,1,...., we can truncate the distribution so that total mass 0.9999 is accumulated
p_trunc=0.9999

precision=51

# The heatmap contains 51x51 values E[R^{(j)}(k)], for different values of (gamma_V,xi)
num_xi=precision
num_gammaV=precision

xis=[]
gammaVs=[]

mean_R0_Volunteer=np.asmatrix(np.zeros((num_xi,num_gammaV)))

for xi_index in range(num_xi):
	print(xi_index)
	xi=0.1+xi_index*(0.9-0.1)/(num_xi-1.0)
	xis=np.append(xis,xi)
	
	for gammaV_index in range(num_gammaV):
		print(gammaV_index)
		gammaV=1.0+gammaV_index*(24.0-1.0)/(num_gammaV-1.0)
		gammaVs=np.append(gammaVs,gammaV)

		# "Gama" vector contains rates \mu_j
		Gama=np.asmatrix(np.zeros((M,1)))

		# "Betas" matrix contains rates \lambda_kj
		Betas=np.asmatrix(np.zeros((M,M)))

		# "Lambdas" vector contains rates \lambda_j
		Lambdas=np.asmatrix(np.zeros((M,1)))

		# These vectors are filled according to functions described in Figure 4. See also Supplementary Material Table S6
		Gama[0]=(1-phi)*deltaC
		Gama[1]=gammaH
		Gama[2]=gammaV
		Lambdas[0]=phi*deltaU
		Lambdas[1]=0
		Lambdas[2]=0
		Betas[0,1]=betaHp*(1.0-eta)/Ns[0]
		Betas[1,0]=betaHp*(1.0-eta)/Ns[0]
		Betas[0,2]=betaVp*(1.0-xi)/Ns[0]
		Betas[2,0]=betaVp*(1.0-xi)/Ns[0]

		mean_j_k=0
		suma=0

		Xis_j_k=[np.asmatrix(np.zeros((Num_States,1))) for n in range(501)]
		Identity=np.asmatrix(np.eye(Num_States))

		n=-1
		while(suma<p_trunc and n<500):
			n+=1
			A_j_k=np.asmatrix(np.zeros((Num_States,Num_States)))
			b_j_k=np.asmatrix(np.zeros((Num_States,1)))
			
			a=[0]*M
			
			l=1
			if j==l:
				ini=1
			else:
				ini=0

			for il in range(ini,Ns[l-1]+1):
				a[l-1]=il
				Fill_Matrix_A_j_k(A_j_k,l+1,a,n,M,Ns,j,k,Gama,Betas,Lambdas)
				Fill_Vector_b_j_k(b_j_k,l+1,a,n,M,Ns,j,k,Gama,Betas,Lambdas,Xis_j_k)
				a[l-1]=ini
			
			Xis_j_k[n]=np.linalg.solve(Identity-A_j_k,b_j_k)
			
			suma+=Xis_j_k[n][Pos_R0(tuple(init_state),Ns,M,j),0]
			mean_j_k+=n*Xis_j_k[n][Pos_R0(tuple(init_state),Ns,M,j),0]
			if suma>p_trunc:
				break

		mean_R0_Volunteer[num_xi-1-xi_index,gammaV_index]=mean_j_k

fig = plt.figure()
ax = fig.add_subplot(111)
cax = plt.imshow(mean_R0_Volunteer, cmap='hot', interpolation='nearest',extent=[gammaVs[0],gammaVs[num_gammaV-1],xis[0],xis[num_xi-1]],aspect="auto")
barra = fig.colorbar(cax)
barra.set_label('$E[R_{(0,0,1)}^{(3)}]$', fontsize=15)

ax.set_xlabel(r'$\gamma_V$', fontsize=15)
ax.set_ylabel(r'$\xi$', fontsize=15)

plt.savefig('Figure6_Right.png')
