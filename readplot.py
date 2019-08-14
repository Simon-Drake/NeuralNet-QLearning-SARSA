import matplotlib.pyplot as plt

# This file reads results from Qlearning and SARSA to plot in one figure

fs = list()
with open("results.txt", "r") as f:
    fs = f.readlines()

fq = list()
with open("qlearn.txt", "r") as f:
    fq = f.readlines()

mvav_m_sarsa = [float(k) for k in fs[0].split(",")[:-1]]
mvav_r_sarsa = [float(k) for k in fs[1].split(",")[:-1]]

mvav_m_q = [float(k) for k in fq[0].split(",")[:-1]]
mvav_r_q = [float(k) for k in fq[1].split(",")[:-1]]


f, axarr = plt.subplots(1,2, figsize=(20,10))

axarr[0].plot(range(0,len(mvav_m_sarsa)), mvav_m_sarsa)
axarr[0].plot(range(0,len(mvav_m_q)), mvav_m_q)
axarr[0].set_title("Moving average: Moves")
axarr[1].plot(range(0,len(mvav_r_sarsa)), mvav_r_sarsa)
axarr[1].plot(range(0,len(mvav_r_q)), mvav_r_q)

axarr[1].set_title("Moving average: Rewards")
plt.show()


