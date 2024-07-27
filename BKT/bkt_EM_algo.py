#################### EM ALGO #########################
# EM works best when we have some data coming from two distributions and we don't know which one of them a point belongs to
# First step is to guess the initial parameter. 
    # Say we have two Gaussians so we need mu, sigma, and pi - pi is a mixing coef (prob that a point comes from first gaussian)
    # in case of two binomials, we have theta 
# Second step (E) is to compute the "responsibilities" - probs that a point comes from each distribution 
  # this is done by plugging the point into pdf for each Gaussian and computing the following:
  # r[ij] = pi[j]*f(x)/sum(pi[k]*f(x)). We multiply pi times the gaussian pdf for the first gaussian plugging in x. Then divide by the sum of the same using both gaussians.
  # repeat the above for point two. 
# for example suppose mu1=0, mu2=0, sigma1=1, sigma2=1, and pi=.5 are initial params. Say we have a point x=.1. We plug in this value into steps above and estimate r[ij]
# third step (M) is to update the parameters using the responsibility values.
  # mu update is sum(r[ij]*x[i])/sum(r[ij]), the sum is over all data points, and this is done for distribution
  # sigma update is sum(r[ij]*(x[i]-mu[j])/sum(r[ij]), same way for each distribution
  # pi update is sum(r[ij])/N
# Repeat for max iterations or until improvement is less than threshold. 
https://medium.com/@keruchen/expectation-maximization-step-by-step-example-a9bb42add37d

#################### BKT SETUP ######################################
# In BKT we have 4 latent parameters:
    # initial knowledge (prior), 
    # Learning rate (transition prob from not knowing to knowing the skill)
    # Guessing Factor
    # Slipping factor
# Data would contain - student_id, skill_id, correct. Similar to our inter_seq (for DKT), one row per student-item pair. 
