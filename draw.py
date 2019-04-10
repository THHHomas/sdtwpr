import matplotlib.pyplot as plt


if __name__ == '__main__':
	f=open("./log.txt")
	data = []
	line = f.readline()
	while(line):
		data.append(eval(line.split()[1]))
		line = f.readline()
	x = [x for x in range(0,len(data),5)]
	data = [data[x] for x in range(0,len(data),5)]
	plt.plot(x[3000:],data[3000:])
	plt.show()
