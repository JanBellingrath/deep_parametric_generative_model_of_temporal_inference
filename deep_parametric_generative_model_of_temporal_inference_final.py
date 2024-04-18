#!/usr/bin/env python
# coding: utf-8

# # The emergence of the ‘width’ of subjective temporality: the self-simulational theory of temporal extension from the perspective of the free energy principle 
# 
# ##### The code is a modified version (permission granted) from Sandved-Smith et. al. (2021): https://colab.research.google.com/drive/1IiMWXRF3tGbVh9Ywm0LuD_Lmurvta04Q?usp=sharing

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# In[3]:


def softmax(X):                                                                 
    norm = np.sum(np.exp(X) + 10**-5)
    Y = (np.exp(X) + 10**-5) / norm
    return Y

def softmax_dim2(X):                                                            
    norm = np.sum(np.exp(X) + 10**-5, axis=0)
    Y = (np.exp(X) + 10**-5) / norm
    return Y


# In[42]:


####### Temporal Experience during an odd-ball perceptual task.

################################################################################
### Defining parameters
################################################################################

T = 100                     ### Number of time-steps
Pi2 = np.zeros((2,T))       ### prior attentional actions (stay, switch)

E2 = np.array([0.99,0.99])  ### prior over attentional policies
gammaG2 = 4.0               
C2 = np.array([0.99, 0.01]) ### preference over attentional outcomes (agent is trying to perform the task - ie. maintain attention on the stimulus)

X2 = np.zeros((2,T))        ### attentional states prior (high vs low precision)
X2bar = np.zeros((2,T))     ### attentional states posterior (high vs low precision)

x2 = np.zeros(T)            ### discrete generative process states --> x2 sets gammaA1
x2[0] = 0                   ### start in a focused state
u2 = np.zeros(T)            ### discrete generative process active states --> u2 sets transition probabilities for x2

X1 = np.zeros((2,T))        ### perception prior (standard vs oddball)
X1bar = np.zeros((2,T))     ### perception posterior (standard vs oddball)

O = np.zeros(T)             ### observations (standard vs oddball)

O[int(T/5)]=1;              ### generative process determined by experimenter
O[int(2*T/5)]=1;            ### generative process determined by experimenter
O[int(3*T/5)]=1;            ### generative process determined by experimenter
O[int(4*T/5)]=1;            ### generative process determined by experimenter

O1 = np.zeros((2,T))        ### observation prior (standard vs oddball)
O1bar = np.zeros((2,T))     ### observation posterior (standard vs oddball)
for t in range(T):
    O1bar[int(O[t]),t]=1

X2[:,0] = [0.5,0.5]         ### attentional state prior D2
X1[:,0] = [0.5,0.5]         ### perceptual state prior D1

######## Defining transition matrices  ##########

B2a = np.zeros((2,2))       ### maintain attentional state - "stay"
B2b = np.zeros((2,2))       ### switch atttentional state - "switch"

B2a[:,0]=[.8,0.2]           ###probability of focus/distracted, given focus+stay
B2a[:,1]=[0.0,1.0]          ###probability of focus/distracted, given distracted+stay

B2b[:,0]=[0.0,1.0]          ###probability of focus/distracted, given focus+switch
B2b[:,1]=[1.0,0.0]          ###probability of focus/distracted, given distracted+switch

B1 = np.zeros((2,2))
B1[:,0]=[0.8,0.2]          
B1[:,1]=[0.2,0.8] 

######## Defining likelihood matrices  ##########

A1 = np.zeros((2,2))
A1[:,0] = [0.75,0.25]
A1[:,1] = [0.25,0.75]
gammaA1 = np.zeros(T)

betaA1m = np.zeros(2)
betaA1m[:] = [0.5,2.0]

A2 = np.zeros((2,2))
A2[:,0] = [0.75,0.25]
A2[:,1] = [0.25,0.75]
gammaA2 = 1.0
A2 = softmax_dim2(np.log(A2)*gammaA2)

######## Setting up F & G calculations #######

H2 = np.zeros(2)
H2[0] = np.inner(A2,np.log(A2))[0,0]
H2[1] = np.inner(A2,np.log(A2))[1,1]

G2 = np.zeros((2,T))

######## Setting up the risk tracking and threshold #######
num_policies = 2
                 
action_tracker = []
policy_tracker = []
policy_EFEs = {}
EFEs_last_epoch = {}

risk_over_time = np.full(T, np.nan)
              
threshold = 2.2
skip_counter = 0

################################################################################################################

# Running the simulation
for t in range(T-2):
    betaA1 = np.sum(betaA1m*np.inner(A2,X2[:,t])) ### Bayesian model average (prior precision beliefs)
    gammaA1[t] = betaA1m[int(x2[t])]**-1          ### actual precision is based on generative process (earlier it was betaA1**-1)    
    A1bar = softmax_dim2(A1**gammaA1[t])          ### Precision weighted likelihood mapping
    O1[:,t] = np.inner(A1bar,X1[:,t])             ### Observation priors
    X1bar[:,t] = softmax(np.log(X1[:,t])+gammaA1[t]*np.log(A1[int(O[t]),:])) ### calculate perceptual state posterior

    AtC = 0                                       ### "attentional charge" - inverse precision updating term
    for i in range(2):                            ##loop over outcomes
        for j in range(2):                        ##loop over states
            AtC += (O1bar[i,t]-A1bar[i,j])*X1bar[j,t]*np.log(A1[i,j])  
            
    if AtC > betaA1m[0]:
        AtC = betaA1m[0]-10**-5
    betaA1bar = betaA1 - AtC                      ### inverse precision posterior
    X2bar[:,t] = softmax(np.log(X2[:,t])-1.0*np.log((betaA1m-AtC)/betaA1m*betaA1/betaA1bar))  ### calculate attentional state posterior given precision beliefs  
    
                                                  ### Policy evaluation and execution
    if skip_counter > 0:                          ### skip counter
        skip_counter -= 1
        
        for j in range(t - 1, -1, -1):            ### Loop backwards from the current time step to find last selected policy, and update according to that policy
            if np.all((Pi2[:, j] != 0) & ~np.isnan(Pi2[:, j])):
                last_valid_t = j
                break

        B2 = B2a * Pi2[0, last_valid_t] + B2b * Pi2[1, last_valid_t] ### update state transition matrix B2
        X2[:, t+1] = np.inner(B2, X2bar[:, t])    ### update X2 based on B2 and X2bar
        X1[:, t+1] = np.inner(B1, X1bar[:, t])    ### update X1 based on B1 and X1bar
        u2[t] = np.argmax(Pi2[:, last_valid_t])   ### select attentional action (generative process states)
        
        if u2[t] == 0:
            x2[t+1] = np.random.choice([0, 1], p=B2a[:, int(x2[t])]) ### set generative process state (x2 sets gammaA1), given "stay"
            action_tracker.append(0)
        else:
            x2[t+1] = np.random.choice([0, 1], p=B2b[:, int(x2[t])]) ### set generative process state (x2 sets gammaA1), given "switch"
            action_tracker.append(0)
            
        policy_tracker.append(2)
        continue
        
    elif skip_counter == 0:                       ### if the skip counter == 0, policy-simulation takes place
        current_policy = (t // 2 + t) % num_policies ### leads to a sequence of (0,1),(1,0),(0,1),(1,0), where the brackets can be understood as evaluation epochs
        
        if current_policy == 0:
            X2a = np.inner(B2a, X2bar[:, t])      
            O2a = np.inner(A2, X2a)    
            EFE = np.sum(O2a * (np.log(O2a) - np.log(C2)) - X2a * H2) ### calculating the EFE (in terms of risk and ambiguity)
            Risk = np.sum(O2a * (np.log(O2a) - np.log(C2))) ### caluculating the risk (part of the EFE) 
            
        elif current_policy == 1:
            X2b = np.inner(B2b, X2bar[:, t])
            O2b = np.inner(A2, X2b)               
            EFE = np.sum(O2b * (np.log(O2b) - np.log(C2)) - X2b * H2) ### calculating the EFE (in terms of risk and ambiguity)
            Risk = np.sum(O2b * (np.log(O2b)- np.log(C2))) ### caluculating the risk (part of the EFE)
            
        policy_EFEs[current_policy] = EFE      ### store the EFF for a given policy 
        risk_over_time[t] = Risk                
        
        if t > 1 and EFE < EFEs_last_epoch[0] and EFE < EFEs_last_epoch[1] and t % 2 == 0: #if the EFE of the first policy per epoch is smaller than the EFE of both policies from the previous epoch, immediately select it.
            immediate_selection = np.zeros(num_policies)
            immediate_selection[current_policy] = 1.0
            Pi2[:, t] = immediate_selection            ### deterministically set Pi2 to the current policy
            
            B2 = B2a * Pi2[0, t] + B2b * Pi2[1, t]     ### update state transition matrix B2
            X2[:, t+1] = np.inner(B2, X2bar[:, t])     ### update X2 based on B2 and X2bar
            X1[:, t+1] = np.inner(B1, X1bar[:, t])     ### update X1 based on B1 and X1bar
            u2[t] = np.argmax(Pi2[:, t])               ### select attentional action (generative process states)
        
            if u2[t] == 0:
                x2[t+1] = np.random.choice([0, 1], p=B2a[:, int(x2[t])]) ### set generative process state (x2 sets gammaA1), given "stay"
                action_tracker.append(1)
            else:
                x2[t+1] = np.random.choice([0, 1], p=B2b[:, int(x2[t])]) ### set generative process state (x2 sets gammaA1), given "switch"
                action_tracker.append(1)
  
            if EFE < threshold:           ### set skip counter based on "quality" (in terms of EFE)
                skip_counter = 1
            else:
                skip_counter = 0

        elif t == 0 or ((EFE >= EFEs_last_epoch[0] or EFE >= EFEs_last_epoch[1]) and t % 2 == 0): #if the first policy per epoch has a higher or equal EFE than one policy of the previous epoch 
                
                if t == 0:                             ### in the first, time-step, simply propagate X2 and u2 forward
                    X2[:, t+1] = X2[:, t]
                    X1[:, t+1] = np.inner(B1, X1bar[:, t])
                    u2[t] = 0
                    action_tracker.append(0)
                    
                else:                                  ### Loop backwards from the current time step to find last selected policy, and update according to that policy
                    for j in range(t - 1, -1, -1):
                        if np.all((Pi2[:, j] != 0) & ~np.isnan(Pi2[:, j])):
                            last_valid_t = j
                            break

                    B2 = B2a * Pi2[0, last_valid_t] + B2b * Pi2[1, last_valid_t] ### update state transition matrix B2
                    X2[:, t+1] = np.inner(B2, X2bar[:, t])  ### update X2 based on B2 and X2bar
                    X1[:, t+1] = np.inner(B1, X1bar[:, t])  ### update X1 based on B1 and X1bar
                    u2[t] = np.argmax(Pi2[:, last_valid_t]) ### select attentional action (generative process states)
                    
                if u2[t] == 0:
                    x2[t+1] = np.random.choice([0, 1], p=B2a[:, int(x2[t])]) ### set generative process state (x2 sets gammaA1), given "stay"
                    action_tracker.append(0)
                else:
                    x2[t+1] = np.random.choice([0, 1], p=B2b[:, int(x2[t])]) ### set generative process state (x2 sets gammaA1), given "switch"
                    action_tracker.append(0)                   
                
        elif t % 2 == 1:                                    ### if it is the second policy per epoch, finishing a given simulation epoch
            Pi2[:, t] = softmax(np.log(E2)-gammaG2 * np.array(list(policy_EFEs.values()))) ### set Pi2 based on EFE, E2 and gammaG2

            B2 = B2a * Pi2[0, t] + B2b * Pi2[1, t]          ### update state transition matrix B2
            X2[:, t+1] = np.inner(B2, X2bar[:, t])          ### update X2 based on B2 and X2bar
            X1[:, t+1] = np.inner(B1, X1bar[:, t])          ### update X1 based on B1 and X1bar
            u2[t] = np.argmax(Pi2[:, t])                    ### select attentional action (generative process states)

            if u2[t] == 0:
                x2[t+1] = np.random.choice([0, 1], p=B2a[:, int(x2[t])]) ### set generative process state (x2 sets gammaA1), given "stay"
                if current_policy == 0:
                    action_tracker.append(1)
                else:
                    action_tracker.append(2)
            else:
                x2[t+1] = np.random.choice([0, 1], p=B2b[:, int(x2[t])]) ### set generative process state (x2 sets gammaA1), given "switch"
                if current_policy == 1:
                    action_tracker.append(1)
                else:
                    action_tracker.append(2)
                
            if EFE < threshold:               ### set skip counter based on "quality" (in terms of EFE)
                skip_counter = 2
            else:
                skip_counter = 0
        
        policy_tracker.append(current_policy)
            
        if t > 1 and EFE < EFEs_last_epoch[0] and EFE < EFEs_last_epoch[1] and t % 2 == 0:
            EFEs_last_epoch = {0: float('inf'), 1: float('inf')} ### resetting "memory"
            EFEs_last_epoch[current_policy] = EFE          ### storing EFE in "memory"
        else:
            EFEs_last_epoch[current_policy] = EFE          ### storing EFE in "memory"
        
   
    
#Plotting results 
##############################################################################################################

# Create figure and axes with 5 subplots
fig, axs = plt.subplots(4, 1, figsize=(12, 10)) 
time_range = np.arange(1, T-1)  

# Plot for the first level: Perceptual State
axs[0].plot(time_range, X1bar[0, :T-2], label=r'${\bar{s}}^{(1)}$', color='royalblue')
axs[0].scatter(time_range, 1 - O[:T-2], label='true state', color='steelblue')
axs[0].set_yticks([0, 1])
axs[0].set_yticklabels(['deviant', 'standard'])
axs[0].set_xlabel(r'time ($\tau$)')
axs[0].set_title('First Level: Perceptual State')
axs[0].legend(loc='lower right')
axs[0].set_ylim([-0.1, 1.1])
axs[0].set_xlim([0.5, T - 1.5])

# Plot for the second level: Attentional State
axs[1].plot(time_range, X2bar[0, :T-2], label=r'${\bar{s}}^{(2)}$', color='darkgreen')
axs[1].scatter(time_range, 1 - x2[:T-2], label='true state', color='darkgreen')
axs[1].set_ylim([-0.1, 1.1])
axs[1].set_yticks([0, 1])
axs[1].set_yticklabels(['distracted', 'focused'])
axs[1].set_xlabel(r'time ($\tau$)')
axs[1].set_title('Second Level: Attentional State')
axs[1].legend(loc='lower right')
axs[1].set_xlim([0.5, T - 1.5])


color_map_policies = {0: 'orange', 1: 'lightblue', 2: 'gray'}  
policy_labels = {0: 'stay', 1: 'switch', 2: 'no policy simulated'}

for i, policy in enumerate(policy_tracker):
    if i < len(time_range):
        axs[2].scatter(time_range[i], 0, color=color_map_policies[policy], s=40)

for i in range(1, len(time_range), 2): 
    x_position = time_range[i] + .5
    axs[2].vlines(x=x_position, ymin=-0.02, ymax=0.02, colors='black', linewidth=1)

axs[2].set_title('Simulated Policies')
axs[2].set_xlim([0.5, T - 1.5])
axs[2].set_ylim([-0.1, 0.1])
axs[2].set_xlabel(r'time ($\tau$)')
axs[2].set_xticks([20, 40, 60, 80])
axs[2].set_yticks([])

legend_handles_policies = [Line2D([0], [0], marker='o', color='w', label=policy_labels[key], 
                                  markersize=10, markerfacecolor=color_map_policies[key]) for key in policy_labels]
legend_policies = axs[2].legend(handles=legend_handles_policies, loc='lower right')

normal_cycle_handle = Line2D([0], [0], color='black', label='normal action selection cycle', linewidth=1.5)
legend_cycle = axs[2].legend(handles=[normal_cycle_handle], loc='upper right')

axs[2].add_artist(legend_policies)

risk_over_time = pd.Series(risk_over_time)
risk_over_time = risk_over_time.ffill()

# Plotting subjective temporal Now
axs[3].plot(time_range, risk_over_time[:-2], color='purple')
axs[3].set_title('Width of the Subjective Temporal Now')
axs[3].set_xlabel(r'time ($\tau$)')
axs[3].set_xlim([0.5, T - 1.5])
axs[3].set_yticks([1.125, 2.5])
axs[3].set_yticklabels(["low", "high"]) 
axs[3].set_ylabel('Self-Simulational Dissimilarity')

color_map_axs3 = {0: 'grey', 1: 'yellow', 2: 'darkred'} 
congruency_labels = {0: 'no action selection', 1: 'congruency', 2: 'incongruency'}

y_value = axs[3].get_ylim()[0] 
for i in range(len(action_tracker)):
    color = color_map_axs3.get(action_tracker[i], 'default_color')
    axs[3].scatter(i, y_value, color=color, s=40)
    
legend_handles_congruency = [Line2D([0], [0], marker='o', color='w', label=congruency_labels[key], 
                                    markersize=10, markerfacecolor=color_map_axs3[key]) for key in congruency_labels]

legend_congruency = axs[3].legend(handles=legend_handles_congruency, loc='lower right')

axs[3].add_artist(legend_congruency)

plt.tight_layout()
plt.show()


# In[ ]:




