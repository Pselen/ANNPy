from Network import  Network



layers = [4,1,3,4]
network1 = Network(layers)

input = [0.1, 0.5, 0.2, 0.9]
target = [0.5, 1, 0.7 , 0.2]

for i in range(100):
    network1.train(input, target, 0.3)



output = network1.calculate(input)

for i in range(4):
    print(str(output[i]))
    print("==")
    