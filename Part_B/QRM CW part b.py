#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import chi2
from scipy.stats import norm
df1=pd.read_csv("/Users/kimizuo/Desktop/QRM-2025-cw1-data-b(1).csv")


# In[2]:


#Calculate the negative log return
df1["neg Log return"]=-np.log(df1["Adj Close"]).diff().fillna(0)


# In[3]:


#Historical Simulation with 500 windows
result=[]
for i in range(500, len(df1)-1):
    window_losses = df1["neg Log return"][i-500:i]
    #Calulate the VaR and ES in the window
    VaR_95 = np.quantile(window_losses, 0.95)
    VaR_99 = np.quantile(window_losses, 0.99)
    ES_95 = window_losses[window_losses >= VaR_95].mean()
    ES_99 = window_losses[window_losses >= VaR_99].mean()
    result.append({"HS_VaR_95": VaR_95,"HS_ES_95": ES_95,"HS_VaR_99": VaR_99,"HS_ES_99": ES_99,
                    "realized_loss": df1.loc[i, "neg Log return"]})
result = pd.DataFrame(result)


# In[4]:


#Plot of results using HS
plt.figure(figsize=(12,6))

# Plot actual realized losses
plt.plot(result.index, result["realized_loss"], label="Realized Loss", linewidth=1)

# Plot HS VaR forecasts
plt.plot(result.index, result["HS_VaR_95"], label="HS VaR 95%", linestyle="--")
plt.plot(result.index, result["HS_VaR_99"], label="HS VaR 99%", linestyle="--")

# Plot HS ES forecasts
plt.plot(result.index, result["HS_ES_95"], label="HS ES 95%", linestyle="-")
plt.plot(result.index, result["HS_ES_99"], label="HS ES 99%", linestyle="-")

plt.title("Historical Simulation (HS) VaR/ES Forecasts vs Realized Losses")
plt.xlabel("Time")
plt.ylabel("negative Log return")
plt.legend()
plt.grid(True)
plt.show()


# In[5]:


#Filtered Historical Simulation with EWMA with 500 windows
result1=[]
#Initial estimation of variance
variance=[df1["neg Log return"][:500].var()]
residuals=list(df1["neg Log return"][:500]/np.sqrt(variance[0]))
for i in range(500, len(df1)-1):
    window_losses = residuals[i-500:i]
    variance.append(0.06*df1["neg Log return"][i-1]**2+0.94*variance[-1])
    #The 95% quantile for standardised residual
    z95=np.quantile(window_losses, 0.95)
    #The 99% quantile for standardised residual
    z99=np.quantile(window_losses, 0.99)
    VaR_95 = np.sqrt(variance[-1])*z95
    VaR_99 = np.sqrt(variance[-1])*z99
    ES_95 = np.sqrt(variance[-1])*np.mean([x for x in window_losses if x >= z95])
    ES_99 = np.sqrt(variance[-1])*np.mean([x for x in window_losses if x >= z99])
    #Append the new standardised residual
    residuals.append(df1["neg Log return"][i]/np.sqrt(variance[-1]))
    result1.append({"FHS_VaR_95": VaR_95,"FHS_ES_95": ES_95,"FHS_VaR_99": VaR_99,"FHS_ES_99": ES_99,
                    "realized_loss": df1.loc[i, "neg Log return"]})
result1 = pd.DataFrame(result1)


# In[6]:


#Plot of results using HS
plt.figure(figsize=(12,6))

# Plot actual realized losses
plt.plot(result1.index, result1["realized_loss"], label="Realized Loss", linewidth=1)

# Plot HS VaR forecasts
plt.plot(result1.index, result1["FHS_VaR_95"], label="FHS VaR 95%", linestyle="--")
plt.plot(result1.index, result1["FHS_VaR_99"], label="FHS VaR 99%", linestyle="--")

# Plot HS ES forecasts
plt.plot(result1.index, result1["FHS_ES_95"], label="FHS ES 95%", linestyle="-")
plt.plot(result1.index, result1["FHS_ES_99"], label="FHS ES 99%", linestyle="-")

plt.title("FHS with EWMA VaR/ES Forecasts vs Realized Losses")
plt.xlabel("Time")
plt.ylabel("negative Log return")
plt.legend()
plt.grid(True)
plt.show()


# In[7]:


#Implementation of FHS with GARCH method
results2 = []

for i in range(500, len(df1)-1):
    # Use the previous 500 returns for estimation window
    window1 = df1["neg Log return"].iloc[i-500:i]
    # Fit GARCH(1,1) with constant mean, normal innovations
    am = arch_model(window1, mean="Constant", vol="Garch", p=1, q=1, dist="normal",rescale=False)
    res = am.fit(disp="off")
    # 1-step ahead forecast of conditional variance
    forecast = res.forecast(horizon=1, reindex=False)
    # Using the variance forecast for next day
    vol1 = forecast.variance.values[-1, 0]
    sigma1 = np.sqrt(vol1)
    # Find the last 500 standardized residuals Z_t
    Z_t2 = res.std_resid[-500:]
    # Compute quantiles of standardised residual
    z_95 =np.quantile(Z_t2, 0.95)
    z_99 = np.quantile(Z_t2, 0.99)
    VaR95 = sigma1*z_95
    VaR99 = sigma1*z_99
    ES95 = sigma1*Z_t2[Z_t2 >= z_95].mean()
    ES99 = sigma1*Z_t2[Z_t2 >= z_99].mean()
    results2.append({"FHS-GARCH_VaR_95": VaR95,"FHS-GARCH_ES_95": ES95,"FHS-GARCH_VaR_99": VaR99,
                     "FHS-GARCH_ES_99": ES99,"realized_loss": df1.loc[i, "neg Log return"]})
results2= pd.DataFrame(results2)


# In[8]:


#Plot of results using HS
plt.figure(figsize=(12,6))

# Plot actual realized losses
plt.plot(results2.index, results2["realized_loss"], label="Realized Loss", linewidth=1)

# Plot HS VaR forecasts
plt.plot(results2.index, results2["FHS-GARCH_VaR_95"], label="FHS-GARCH VaR 95%", linestyle="--")
plt.plot(results2.index, results2["FHS-GARCH_VaR_99"], label="FHS-GARCH VaR 99%", linestyle="--")

# Plot HS ES forecasts
plt.plot(results2.index, results2["FHS-GARCH_ES_95"], label="FHS-GARCH ES 95%", linestyle="-")
plt.plot(results2.index, results2["FHS-GARCH_ES_99"], label="FHS-GARCH ES 99%", linestyle="-")

plt.title("FHS with GARCH VaR/ES Forecasts vs Realized Losses")
plt.xlabel("Time")
plt.ylabel("negative Log return")
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


#The unconditional VaR test
def unconditional(violations, alpha):
    T = len(violations)
    #Find the number of violations
    x = sum(violations)
    pi = x / T
    # Likelihood ratio
    LRuc = -2 * (np.log((1 - alpha) ** (T - x) * alpha ** x) -np.log((1 - pi) ** (T - x) * pi ** x))
    p_value = 1 - chi2.cdf(LRuc, df=1)
    return LRuc, p_value, pi
#Test for independence of violations
def independence_test(violations):
    # Transition counts
    n00 = ((violations == False) & (violations.shift(-1) == False)).sum()
    n01 = ((violations == True) & (violations.shift(-1) == False)).sum()
    n10 = ((violations == True) & (violations.shift(-1) == False)).sum()
    n11 = ((violations == True) & (violations.shift(-1) == True)).sum()

    # Markov transition probabilities
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)
    pi  = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Christoffersen independence LR statistic
    LRind = 2 * np.log(((1 - pi0)**n00 *(pi0)**n01 *(1 - pi1)**n10 *(pi1)**n11) /
                        ((1 - pi)**(n00 + n10) * (pi)**(n01 + n11)))
    p_value = 1 - chi2.cdf(LRind, df=1)
    return LRind, p_value
#Function for joint test
def joint_test(violations, alpha):
    LRuc, p_uc, pi = unconditional(violations, alpha)
    LRind, p_ind = independence_test(violations)
    LRcc = LRuc + LRind
    p_cc = 1 - chi2.cdf(LRcc, df=2)
    return LRcc, p_cc


# In[10]:


#Backtest for the 95% VaR of HS method
violation_HS_95=result["realized_loss"]>result["HS_VaR_95"]
LRuc, p_uc, pi = unconditional(violation_HS_95, 0.05)
LRind, p_ind = independence_test(violation_HS_95)
LRcc, p_cc = joint_test(violation_HS_95, 0.05)

#Backtest for the 99% VaR of HS method
violation_HS_99=result["realized_loss"]>result["HS_VaR_99"]
LRuc1, p_uc1, pi1 = unconditional(violation_HS_99, 0.01)
LRind1, p_ind1 = independence_test(violation_HS_99)
LRcc1, p_cc1 = joint_test(violation_HS_99, 0.01)


# In[11]:


#Backtest for the 95% VaR of FHS with EWMA method
violation_FHS_95=result1["realized_loss"]>result1["FHS_VaR_95"]
LRuc2, p_uc2, pi2 = unconditional(violation_FHS_95, 0.05)
LRind2, p_ind2 = independence_test(violation_FHS_95)
LRcc2, p_cc2 = joint_test(violation_FHS_95, 0.05)

#Backtest for the 99% VaR of FHS with EWMA method
violation_FHS_99=result1["realized_loss"]>result1["FHS_VaR_99"]
LRuc3, p_uc3, pi3 = unconditional(violation_FHS_99, 0.01)
LRind3, p_ind3 = independence_test(violation_FHS_99)
LRcc3, p_cc3 = joint_test(violation_FHS_99, 0.01)


# In[12]:


#Backtest for the 95% VaR of FHS with GARCH method
violation_GFHS_95=results2["realized_loss"]>results2["FHS-GARCH_VaR_95"]
LRuc4, p_uc4, pi4 = unconditional(violation_GFHS_95, 0.05)
LRind4, p_ind4 = independence_test(violation_GFHS_95)
LRcc4, p_cc4 = joint_test(violation_GFHS_95, 0.05)

#Backtest for the 99% VaR of FHS with GARCH method
violation_GFHS_99=results2["realized_loss"]>results2["FHS-GARCH_VaR_99"]
LRuc5, p_uc5, pi5 = unconditional(violation_GFHS_99, 0.01)
LRind5, p_ind5 = independence_test(violation_GFHS_99)
LRcc5, p_cc5 = joint_test(violation_GFHS_99, 0.01)


# In[13]:


#Create a data frame for backtest VaR result
VaR_backtest = pd.DataFrame({"HS_VaR95": {"Uncon": LRuc, "p_uc": p_uc,"Independent": LRind,
                                          "p_it": p_ind,"Joint": LRcc, "p_jt": p_cc},
                          "HS_VaR99": {"Uncon": LRuc1, "p_uc": p_uc1,"Independent": LRind1,
                                          "p_it": p_ind1,"Joint": LRcc1, "p_jt": p_cc1},
                          "FHS_EWMA_VaR95": {"Uncon": LRuc2, "p_uc": p_uc2,"Independent": LRind2,
                                          "p_it": p_ind2,"Joint": LRcc2, "p_jt": p_cc2},
                          "FHS_EWMA_VaR99": {"Uncon": LRuc3, "p_uc": p_uc3,"Independent": LRind3, 
                                          "p_it": p_ind3,"Joint": LRcc3, "p_jt": p_cc3},
                          "FHS_GARCH_VaR95": {"Uncon": LRuc4, "p_uc": p_uc4,"Independent": LRind4,
                                          "p_it": p_ind4,"Joint": LRcc4, "p_jt": p_cc4},
                          "FHS_GARCH_VaR99": {"Uncon": LRuc5, "p_uc": p_uc5,"Independent": LRind5,
                                          "p_it": p_ind5,"Joint": LRcc5, "p_jt": p_cc5}}).T
print(VaR_backtest)


# In[14]:


#Function for backtest ES for different model
def es_backtest(loss, var, es):
    #Indicator function for whether the loss exceeding the VaR
    I = (loss > var).astype(float)
    # value of loss compared to ES forecast
    ex1 = (loss - es) * I
    sum_ex1  = np.sum(ex1)
    sum_ex2 = np.sum(ex1**2)
    #Standard normal test statistics
    Z = sum_ex1 / np.sqrt(sum_ex2)
    #  p-value
    p = 2 * (1 - norm.cdf(abs(Z)))
    return {'Test Statistics': Z,'p-value': p}


# In[15]:


#Create a data frame for backtest ES result
ES_backtest = pd.DataFrame({"HS_ES_95": es_backtest(result["realized_loss"],result["HS_VaR_95"],result["HS_ES_95"]),
                      "HS__ES_99": es_backtest(result["realized_loss"],result["HS_VaR_99"],result["HS_ES_99"]),
                      "FHS_EWMA_ES_95": es_backtest(result1["realized_loss"],result1["FHS_VaR_95"],result1["FHS_ES_95"]),
                      "FHS_EWMA_ES_99": es_backtest(result1["realized_loss"],result1["FHS_VaR_99"],result1["FHS_ES_99"]),
                      "FHS_GARCH_ES_95": es_backtest(results2["realized_loss"],results2["FHS-GARCH_VaR_95"],results2["FHS-GARCH_ES_95"]),
                      "FHS_GARCH_ES_99": es_backtest(results2["realized_loss"],results2["FHS-GARCH_VaR_99"],results2["FHS-GARCH_ES_99"])}).T
print(ES_backtest)

