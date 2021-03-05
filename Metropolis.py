#
# MCMC Generator
# The program will generate the target joint guassian distribution of higher
# dimension. It will plot the projection of selected two dimension joint. 
# The references include my past projects and a github page of MCMC on 2d. 
#


from   numpy.random import Generator, PCG64 
import numpy             as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns


def pgauss(xlist, mu, sigma):
    return st.multivariate_normal.pdf(xlist, mean=mu, cov=sigma)


# Input: 
#   pi --- target distribution 
#   dim --- dimension of the multivariate normal dist.
#   iter --- Numebr of iterations proformed 
#   mu --- the mean of the each xi
#   sigma --- the std for each xi
#
# Output:
#   sample 

def metroplis(pi, dim, iter, mu, sigma):
    xlist = np.zeros(dim)             # xi's for multi normal
    sample = np.zeros((iter, dim))    # samples of iterations
    for k in range(iter):
        # condidate ylist: yi's from desired dimension 
        ylist = xlist + np.random.normal(size=dim) 
        ylist[2] = ylist[2] + np.random.rand()

        # compute the ratio alpha; accept and reject
        # note that it is detailed balance since s^2*Id commutes 
        if np.random.rand() < pi(ylist, mu, sigma)/pi(xlist, mu, sigma):
            xlist = ylist        # accept
        
        sample[k] = xlist

    return sample

###########################################################################
#####     Main     #####
# Set parameters 
dim = 6
mu = np.zeros(dim)
s = 4
sigma = s*np.identity(dim)
iter = 2500
pi = pgauss
d1 = 2
d2 = 3

# Run the above function 
sample = metroplis(pi, dim, iter, mu, sigma)

############################################################################
# Generate the histogram for guassian
nbins = 50 
d = d2
maxT = max(sample[:,d])
minT = min(sample[:,d])
DelT = (maxT-minT)/nbins              # bin width
binT = np.zeros(nbins)                # bin centers
binC = np.zeros(nbins)   

for ii in range(iter):
    Tk = sample[ii,d]
    if ( minT < Tk < maxT):           # count samples in the range of the histogram
        Bk = int( (Tk+minT)/DelT )           # truncate to identify the bin
        binC[Bk] += 1

# set the value for bin center 
for jj in range(nbins):    
    binT[jj] = minT + (jj+1)*DelT


########################################################################
fig, ax = plt.subplots()  
ax.plot(sample[:, d1],sample[:, d2], 'b*')
title = "Joint Guassian of {d1: 1.0f} and {d2: 2.0f} -th dimensions with total dimension {dim: 2.0f}"
title = title + "\n " + "sigma = {s:2.2f}" + "  iteration = {iter:6.0f}"
title = title.format(dim=dim, d1=d1, d2=d2,s=s, iter=iter)
ax.set_title(title)

fg, bx = plt.subplots()
bx.plot(binT, binC)
title2 = "Guassian for the {d:1.0f}-th dimension"
title2 = title2.format(d=d)
bx.set_title(title2)

plt.show()
