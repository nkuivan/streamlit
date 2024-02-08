# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:11:26 2024

@author: nkuiv
"""
# Effect of new ARV drug on HIV transmission to infants in  pregnant women (Causal Inference)
# W = baseline log viral load N(3,1)
# A = ARV drug treatment: 0: Standard 1: New
# Y = Infant infection status at birth

# Libraries/packages

import streamlit as st
import numpy as np
import pandas as pd

#Generate data

n = 10000
np.random.seed(0)

# Generate W, A, and Y
W = np.random.normal(3, 1, n)
A = np.random.binomial(1, 0.4, n)
logit_Y = -2 - A + 0.1*W
Y = np.random.binomial(1, 1 / (1 + np.exp(-logit_Y)))

# Create a DataFrame
df = pd.DataFrame({
    'W': W,
    'A': A,
    'Y': Y
})

# Check randomization of treatment

mean_W_A_0 = np.mean(W[A == 0])
print(mean_W_A_0)
mean_W_A_1 = np.mean(W[A==1])
print(mean_W_A_1)

print(f"Distribution of covariates between treatment groups A=1 and A=0 are similar: {mean_W_A_1, mean_W_A_0}")


#Estimate causal relative risk and risk difference

Y_A_0 = df.loc[df['A'] == 0, 'Y']
Y_A_1 = df.loc[df['A'] == 1, 'Y']

relative_risk = Y_A_1.mean() / Y_A_0.mean()

risk_difference = Y_A_1.mean() - Y_A_0.mean()

print(f"Causal relative risk: {relative_risk}")

print(f"Risk difference: {risk_difference}")

import matplotlib.pyplot as plt

# Create a histogram of the patients' viral loads
plt.hist(df['W'], bins=30, color='skyblue', edgecolor='black')

plt.title('Histogram of Patients\'baseline log Viral Loads')

plt.xlabel('log Viral Load'); plt.ylabel('Frequency')

st.title('Practical Exercise: Antiretroviral Drug Study')

st.dataframe(df)

st.pyplot(plt.gcf()) 

st.write(f"Mean W for A=0: {mean_W_A_0}")

st.write(f"Mean W for A=1: {mean_W_A_1}")

st.write(f"Causal relative risk: {relative_risk}")

st.write(f"Risk difference: {risk_difference}")

