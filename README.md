# streamlit

# A new antiretroviral drug has just been developed. We are interested in the effectiveness of
the new drug in reducing HIV transmission to infant at birth compared to the current
standard of treatment. A randomized controlled trial is initiated where HIV positive
pregnant women are randomized to new drug with probability 0.4. Baseline viral load is also
assessed at baseline. The infant’s HIV infection status is assessed at birth using DNA PCR.
Simulate the study data and estimate the effect of the new drug using the following
assumptions:
Let W be the baseline log viral load from a normal distribution with mean=3 and standard
deviation=1.
A is the treatment indicator variable with 0=standard of care and 1=new drug
Y is the infant’s HIV infection status with 0=not infected and 1=infected.
Assume 𝑙𝑜𝑔𝑖𝑡 𝐸 𝑌! = 1|𝑊 = 𝛽! + 𝛽!𝑎 + 𝛽!𝑤
where 𝛽! = −2, 𝛽! = −1, 𝛽! = 0.1
Assume a sample size n=1000.
1. Generate the pair of potential outcomes given the information above
2. Generate the observed data from the study
3. Check that randomization of treatment was achieved
4. Estimate the causal relative risk and risk difference 
