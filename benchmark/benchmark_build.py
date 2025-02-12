import matplotlib.pyplot as plt

# Data from benchmark.py
x = [66, 106, 146, 186, 226, 266, 306, 346, 386]
y1 = [1.182673, 8.427423, 20.838155, 29.837385, 38.893159, 40.240555, 46.508006, 52.198892, 54.142618] # diag time
y2 = [0.454161, 4.781357, 10.551366, 11.909735, 17.000203, 34.111541, 39.305087, 45.733083, 69.308079] # HPCP purif time
y3 = [0.369637, 3.334863, 8.208294, 19.910882, 20.604009, 22.666646, 33.471929, 42.849892, 49.393655] # TC2 purif time
y4 = [0.191152, 3.793516, 12.244729, 13.669052, 20.132319, 24.809736, 29.008021, 46.385996, 58.311562] # TRS4 purif time

# Create the plot
plt.scatter(x, y1, label='diag time', color='b')
plt.scatter(x, y2, label='HPCP time', color='r')
plt.scatter(x, y3, label='TC2 time', color='g')
plt.scatter(x, y4, label='TRS4 time', color='yellow')

plt.xlabel('basis size of molecules')
plt.ylabel('time')
plt.title('time of diagonalization/purification according to the size of the basis')
plt.legend()
plt.show()
