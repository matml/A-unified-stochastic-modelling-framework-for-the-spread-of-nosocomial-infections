import matplotlib.pyplot as plt
import numpy as np

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
def Fill_Matrix_A(A,l,a,n,M,Ns,j,Gama,Betas,Lambdas):
	if l<M:
		if j==l:
			ini=1
		else:
			ini=0
		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			Fill_Matrix_A(A,l+1,a,n,M,Ns,j,Gama,Betas,Lambdas)
			a[l-1]=ini
	else:
		if j==l:
			ini=1
		else:
			ini=0
		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			denominator=Deltas[0]*a[0]
			for h in range(1,M+1):
				if h!=j:
					denominator+=a[h-1]*Gama[h-1]+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
				else:
					denominator+=(a[h-1]-1)*Gama[h-1]+Gama[h-1]/(1.0-sigma)+(Ns[h-1]-a[h-1])*Lambdas[h-1]
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
							A[Pos_R0(a,Ns,M,j),Pos_R0(tuple(new),Ns,M,j)]+=Betas[p-1,h-1]*(a[p-1]-1)*(Ns[h-1]-a[h-1])/denominator
			a[l-1]=ini

# This function constructs vector e^{(j)}(n) in Supplementary Material, Section 1.1
def Fill_Vector_b(b,l,a,n,M,Ns,j,Gama,Betas,Lambdas,Xis):
	if l<M:
		if j==l:
			ini=1
		else:
			ini=0

		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			Fill_Vector_b(b,l+1,a,n,M,Ns,j,Gama,Betas,Lambdas,Xis)
			a[l-1]=ini
	else:
		if j==l:
			ini=1
		else:
			ini=0

		for il in range(ini,Ns[l-1]+1):
			a[l-1]=il
			denominator=Deltas[0]*a[0]
			for h in range(1,M+1):
				if h!=j:
					denominator+=a[h-1]*Gama[h-1]+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
				else:
					denominator+=(a[h-1]-1)*Gama[h-1]+Gama[h-1]/(1.0-sigma)+(Ns[h-1]-a[h-1])*Lambdas[h-1]
					for p in range(1,M+1):
						denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
			if n>0:
				for h in range(1,M+1):
					if a[h-1]<Ns[h-1]:
						new=np.append([],a)
						new[h-1]+=1
						b[Pos_R0(a,Ns,M,j),0]+=Betas[j-1,h-1]*(Ns[h-1]-a[h-1])*Xis[n-1][Pos_R0(tuple(new),Ns,M,j),0]/denominator
			else:
				b[Pos_R0(a,Ns,M,j),0]+=(Gama[j-1]/(1.0-sigma))/denominator
				b[Pos_R0(a,Ns,M,j),0]+=Deltas[0]*a[0]/denominator
			a[l-1]=ini

# Case Study 1. 20 patients and 3 HCWs

N=20+3
M=2
Ns=[20,3]

# Probability of admitted patient being colonized

sigma=0.01

# HCW hand-washing rate

muprime = 14.0

# HCW-to-patient infection rate 

beta=1.0/6.0

# Patient-to-HCW contamination rate

betaprime = 1.0/6.0

# Discharge rate

mu=1.0/10.0


# In Figure 3 left, we explore 4 potential detection rates (gamma^{-1}=1,2,3,4 days)
for detection in range(1,5):
	
	# Detection rate
	gamma=1.0/detection

	# We compute in Figure 3 probabilities p(R_{(1,0)}^{(1)}=n) for n=0,1,...,10
	n_max=10

	# Since we are computing the reproduction number of a patient, who belongs to compartmental level 1, we set j=1
	j=1
	
	# "Gama" vector contains rates \mu_j
	Gama=np.asmatrix(np.zeros((M,1)))
	Gama[0]=(1-sigma)*mu
	Gama[1]=muprime
	
	# "Betas" matrix contains rates \lambda_kj
	Betas=np.asmatrix(np.zeros((M,M)))
	Betas[0,1]=betaprime
	Betas[1,0]=beta
	
	# "Lambdas" vector contains rates \lambda_j
	Lambdas=np.asmatrix(np.zeros((M,1)))
	Lambdas[0]=sigma*mu
	Lambdas[1]=0
	
	# "Deltas" vector contains rates \delta_j, such that \delta(i_1,i_2)=\delta_1i_1+\delta_2i_2
	Deltas=np.asmatrix(np.zeros((M,1)))
	Deltas[0]=gamma
	Deltas[1]=0

	# For initial state (0,...,0,1,0,...,0), where "1" corresponds to compartmental level j
	init_state=[0]*M
	init_state[j-1]=1

	# This is the dimension of the system of equations to be solved. See Supplementary Material
	Num_States=1
	for i in range(1,M+1):
		if i!=j:
			Num_States=Num_States*(Ns[i-1]+1)
		else:
			Num_States=Num_States*Ns[i-1]
	
	mean=0
	suma=0

	Xis=[np.asmatrix(np.zeros((Num_States,1))) for n in range(n_max+1)]
	Identity=np.asmatrix(np.eye(Num_States))
	R0_probabilities=np.zeros((n_max+1,1))

	for n in range(n_max+1):
		
		# For each value of n, we solve the corresponding system of equations (see Supplementary Material)
		A=np.asmatrix(np.zeros((Num_States,Num_States)))
		b=np.asmatrix(np.zeros((Num_States,1)))
		
		a=[0]*M
		
		l=1
		if j==l:
			ini=1
		else:
			ini=0

		if l<M:
			for il in range(ini,Ns[l-1]+1):
				a[l-1]=il
				Fill_Matrix_A(A,l+1,a,n,M,Ns,j,Gama,Betas,Lambdas)
				Fill_Vector_b(b,l+1,a,n,M,Ns,j,Gama,Betas,Lambdas,Xis)
				a[l-1]=ini
		else:
			if j==l:
				ini=1
			else:
				ini=0
			for il in range(ini,Ns[l-1]+1):
				a[l-1]=il
				denominator=Deltas[0]*a[0]
				for h in range(1,M+1):
					if h!=k:
						denominator+=a[h-1]*Gama[h-1]+(Ns[h-1]-a[h-1])*Lambdas[h-1]
						for p in range(1,M+1):
							denominator+=(Ns[h-1]-a[h-1])*a[p-1]*Betas[p-1,h-1]
					else:
						denominator+=(a[h-1]-1)*Gama[h-1]+Gama[h-1]/(1.0-sigma)+(Ns[h-1]-a[h-1])*Lambdas[h-1]
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
								A[Pos_R0(a,Ns,M,j),Pos_R0(tuple(new),Ns,M,j)]+=Betas[p-1,h-1]*(a[p-1]-1)*(Ns[h-1]-a[h-1])/denominator
				if n>0:
					for h in range(1,M+1):
						if a[h-1]<Ns[h-1]:
							new=np.append([],a)
							new[h-1]+=1
							b[Pos_R0(a,Ns,M,j),0]+=Betas[j-1,h-1]*(Ns[h-1]-a[h-1])*Xis[n-1][Pos_R0(tuple(new),Ns,M,j),0]/denominator
				else:
					b[Pos_R0(a,Ns,M,j),0]+=(Gama[j-1]/(1.0-sigma))/denominator
					b[Pos_R0(a,Ns,M,j),0]+=Deltas[0]*a[0]/denominator
				a[l-1]=ini
		
		Xis[n]=np.linalg.solve(Identity-A,b)
		print(Xis[n][Pos_R0(tuple(init_state),Ns,M,j),0])
		R0_probabilities[n]=Xis[n][Pos_R0(tuple(init_state),Ns,M,j),0]
		suma+=Xis[n][Pos_R0(tuple(init_state),Ns,M,j),0]
		mean+=n*Xis[n][Pos_R0(tuple(init_state),Ns,M,j),0]

	# We plot the distribution. Different colors represent different detection rates
	if detection==1:
		plt.bar(np.arange(0, n_max+1, 1)-0.3, R0_probabilities, width=0.2, color="red",label='1 day')
	if detection==2:
		plt.bar(np.arange(0, n_max+1, 1)-0.1, R0_probabilities, width=0.2, color="blue",label='2 days')
	if detection==3:
		plt.bar(np.arange(0, n_max+1, 1)+0.1, R0_probabilities, width=0.2, color="green",label='3 days')
	if detection==4:
		plt.bar(np.arange(0, n_max+1, 1)+0.3, R0_probabilities, width=0.2, color="orange",label='4 days')


plt.xlabel("n")
plt.ylabel("Probability")
plt.xlim(-0.4,n_max)
plt.ylim(0,1.00)

plt.legend()
plt.show()
