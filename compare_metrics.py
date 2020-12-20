import pprint as pp
import matplotlib.pyplot as plt

logs = open("history/log2.txt", "r")
data = logs.readlines()

g_loss = []
d_loss = []

isTraining = True
isVal = True

g = []
d = []

for row in data:
    values = row.split(" ")
    print(g)
    if 'Step' in row:
        # print(values[6])
        # print(row)
        if float(values[8].replace("\n","")):
            g_loss.append(float(values[6].replace("\n",""))/1000000000000000)
            d_loss.append(float(values[8].replace("\n",""))/1000000000000000)
    # elif 'Epoch' in row and 'time' not in row:
    #     g_loss.append(sum(g)/len(g))
    #     d_loss.append(sum(d)/len(d))
    #     g = []
    #     d = []

plt.plot(g_loss, label="gen")
# plt.plot(d_loss, label="disc")
plt.ylabel('Loss')
plt.legend()
plt.savefig('gen.png', dpi=200)
plt.clf()

# plt.plot(g_loss, label="gen")
plt.plot(d_loss, label="disc")
plt.ylabel('Loss')
plt.legend()
plt.savefig('disc.png', dpi=200)
plt.clf()

plt.plot(g_loss, label="gen")
plt.plot(d_loss, label="disc")
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png', dpi=200)
plt.clf()

