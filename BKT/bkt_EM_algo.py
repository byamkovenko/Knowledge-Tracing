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
    # Learning rate (transition prob from not knowing to knowing the skill) P(T) = P(Lt+1|Lt=0)
    # Guessing Factor P(G)=P(obst=1|Lt=0)
    # Slipping factor P(S) = P(obst=0|Lt=1)
# Data would contain - student_id, skill_id, correct. Similar to our inter_seq (for DKT), one row per student-item pair. 

################ BKT COMPUTATION EXAMPLE ##################################
# The prior update happens after each step in the sequence per student/per skill. 
# P(Lt|obst=1) = P(Lt)*(1-P(S))/(P(Lt)*(1-P(S))+(1-P(Lt))*P(G))  - is probability that student mastered skill at time t. 
# P(Lt|obst=0) = P(Lt)*(P(S))/(P(Lt)*(P(S))+(1-P(Lt))*(1-P(G))) 
# Then updated prior for next time step is P(Lt+1) = P(Lt|obst) + (1 − P(Lt|obst))P(T)

# Here is an example of working through calculations to compute P(L) after three questions in 1 skill. 
data={'user_id':[1,1,1],'skill_name':['plot','plot','plot'],'correct':[1,0,1]}
df=pd.DataFrame(data)
model.fit(data = df, skills = ".*plot.*") # using pyBKT - estimates are quite different than manual below, but likely a function of init params

# step 1 set priors for P(L0) prior knowledge, P(T) - transition to knowing in one step, P(G) and P(S)
# pl=.3
# pt=.2
# pg=.25
# ps=.1

# now for each step in sequence apply the update to the likelihood of knowing
# since first step in seq is correct 
# P(L₁|obs₁=1) = P(L₀)(1-P(S)) / (P(L₀)(1-P(S)) + (1-P(L₀))P(G))  
# P(L₁|obs₁=1) = 0.3*(1-0.1) / (0.3*(1-0.1) + (1-0.3)*0.25)  
# P(L₁|obs₁=1) ≈ 0.41

# # then update the prior 
# P(L2) = P(L1|obs1=1) + (1 - P(L1|obs1=0))P(T)   
# P(L2) = 0.41 + (1 - 0.41) * 0.2  
# P(L2) ≈ 0.528

# # since second step is incorrect
#  P(Lt|obst=0) = P(Lt)*(P(S))/(P(Lt)*(P(S))+(1-P(Lt))*(1-P(G))) 
#  P(L2|obs2=0) ≈ (0.528 * .1)/(.528 * .1 + (1-.528)*(1-.25))
#  P(L2|obs2=0) ≈ .129

#  # we then update the prior again
# P(L3) = P(L2|obs2=0) + (1 - P(L2|obs2=0))P(T)   
# P(L3) = 0.129 + (1 - 0.129) * 0.2  
# P(L3) ≈ 0.303

# # since third step is correct
#  P(Lt|obst=1) = P(L3)(1-P(S)) / (P(L3)(1-P(S)) + (1-P(L3))P(G)) 
#  P(L3|obs3=1) ≈ (0.303 * (1-.1))/(.303 *(1-.1) + (1-.303)*.25)
#  P(L3|obs3=1) ≈ .61

#  # we then update the prior again
# P(L4) = P(L3|obs3=1) + (1 - P(L3|obs3=1))P(T)   
# P(L4) = 0.61 + (1 - 0.61) * 0.2  
# P(L4) ≈ 0.688

# So P(L) at the last step is the posterior probability of student knowing this skill. Computed on a skill/student level. 
# P(T) is a broad model parameter that is learned via EM and indicates overall transition prob from not knowing to knowing after a step for all students

################### EM ALGO #######################
# Expectation:
# start with initial params. Compute responsibility for each point - posterior prob that they knew the skill before answering, given they answered correctly
# P(L1|correct) = P(L0) * (1 - P(S)) / [ P(L0) * (1 - P(S)) + (1 - P(L0)) * P(G) ] - repeat for all points but update P(L) at each step as in the example above.
# NOTE: This is the same step as in manual calcs above. So responsibilities are computed for all points. 
# Maximization: 
# update each parameter based on the estimated probabilities in E step.
# P(G) update - p(G_new) = Σ [ (1 - P(Ln))*p(G)/( (1 - P(Ln))*p(G) + P(Ln)*(1-p(S)) ) ] / |D| (where D is the number of data points with incorrect response). Summation over all incorrect.
# P(S) update - P(S_new) = Σ [ P(Ln)*p(S)/(P(Ln)*p(S) + (1 - P(Ln))*(1 - p(G)) ) ] / |C| (where C is the number of data points with correct responses).  Summation over all correct.
# P(T) update - p(T_new) = Σ [ (1 - P(Ln)) * P(Ln+1) ] / Σ [ (1 - P(Ln)) ]
    # NOTE: For P(T) we are looking at transitions between pairs of questions. 
    # 
################## PREDICTION ####################
# To predict the next question we use estimated P(L) along with S and G params
# P(correct | L) = P(L) * (1 - P(S)) + (1 - P(L)) * P(G)
# NOTE: If a student didn't work on a skill, we can't have a prediction, other than the prior. 
