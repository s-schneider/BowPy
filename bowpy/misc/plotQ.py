nl = np.linspace(0, 1, 11)[1:8]

Q2 = np.loadtxt("Q_stn2.txt")
Q3 = np.loadtxt("Q_stn3.txt")
Q5 = np.loadtxt("Q_stn5.txt")

plt.plot(nl, Q2, "D", label="2 center traces missing", color="blue")
plt.plot(nl, Q2, "b--", label=None)

plt.plot(nl, Q5, "D", label="30 %% of center missing", color="green")
plt.plot(nl, Q5, "g--", label=None)

plt.plot(nl, Q3, "D", label="30 %% randomly missing", color="red")
plt.plot(nl, Q3, "r--", label=None)

plt.legend()

plt.xlabel("Noise Level (% of P)")
plt.ylabel("Q")
