import torch
from tqdm import tqdm
from sklearn.utils import shuffle # used for shuffling training set for each epoch
import numpy as np 
import torchvision.datasets as data # just used to load dataset

def sigma(x):
	exp = torch.exp(x)
	neg_exp = torch.exp(-1*x)
	return (exp-neg_exp)/(exp+neg_exp)

def dsigma(x):
  tanh = sigma(x)
  return 1 - torch.pow(tanh, 2)

def softmax(x):
  return torch.exp(x)/torch.sum(torch.exp(x))

def dsoftmax(x):
  return softmax(x)*(1-softmax(x))

def loss(v, t):
  loss = torch.sum(torch.pow(v-t, 2))
  return loss

def dloss(v, t):
  return 2*(v-t)

def dceloss(v,t):
	res = v-t
	return res

def forward_pass(w1, b1, w2, b2, x):  
  s1 = torch.mm(w1, x) + b1  # sum = weight1 (300, 784) * each_sample(784, 1) + bias1(300,1)
  x1 = sigma(s1) # after passing from the activation function
  s2 = torch.mm(w2, x1) + b2 # sum = weight2 (10, 300) * second_layers_input(300, 1) + bias1(10,1)
  x2 = sigma(s2) # after passing from the activation function
  return x, s1, x1, s2, x2

def backward_pass(w1, b1, w2, b2, t, x, s1, x1, s2, x2, dw1, db1, dw2, db2):
  dx2 = dloss(x2, t) # first derivative of mse where x2 is the prediction, t is the actual output
  ds2 = torch.mul(dx2, dsigma(s2)) # element wise multiplication after the backward activation of second layer
    									             # using the dsigma
  db2 += ds2	# bias is equal to sums (in each step it is added since updates for bias and weights)
    					# are done after iteration amount of batch size is completed
  dw2 += torch.mm(ds2, torch.t(x1))      # ds2= (10,1), x1=(300,10) -> dw2 = (10,300)
  dx1 = torch.mm(torch.t(w2), ds2)  	   # dw2 = (10, 300), ds2 = (10, 1) -> dx1 = (300,1) 
  ds1 = torch.mul(dx1, dsigma(s1))       # ds1 = (300, 1), s1 = (300, 1) -> element wise multiplication
  db1 += ds1							
  dw1 += torch.mm(ds1, torch.t(x))       # ds1 (300,1), x = (784,1) -> dw1 = (300, 784)
  return dw1, dw2, db1, db2

def compute_error(test_input, test_target, w1, b1, w2, b2):
  numb_of_error = 0 
  for i in range (0,len(test_target)):
    _, _, _, _, pred = forward_pass(w1, b1, w2, b2, torch.reshape(test_input[i], (len(test_input[i]), 1)))
    if torch.argmax(pred) != torch.argmax(test_target[i]):
      numb_of_error += 1
  return numb_of_error/100 # instead of dividing by 10000 and multiplying with 100, I just simply divide by 100

def two_layered_MLP_784_300_10(train_input, train_target, test_input, test_target):
  epsilon = 1*(10**(-4)) #for weight initialization
  w1 = torch.zeros(300, 784).normal_(0,epsilon).double()
  b1 = torch.zeros(300,1).normal_(0,epsilon).double()
  w2 = torch.zeros(10, 300).normal_(0,epsilon).double()
  b2 = torch.zeros(10,1).normal_(0,epsilon).double()
  
  eta = 0.001  # learning rate
  counter = 0	 # just a counter to handle with batches
  for i in tqdm(range (0,30000)):  # at total, 100 epoch 
    dw1 = torch.zeros(300,784).double()	            # differential weights and biases
    dw2 = torch.zeros(10, 300).double()
    db1 = torch.zeros(300,1).double()
    db2 = torch.zeros(10,1).double()
    for j in range(counter*100, (counter+1)*100):	# batch size = 100
      x = torch.reshape(train_input[j], (len(train_input[j]),1))	     # just for size matches for the arrays
      t = torch.reshape(train_target[j], (len(train_target[j]), 1))    # just for size matches for the arrays
      x0, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2, x)             # forward propagation for 2 layered architecture
      dw1, dw2, db1, db2 = backward_pass(w1, b1, w2, b2, t,  
                                         x0, s1, x1, s2, x2, 
                                         dw1, db1, dw2, db2)  # backward propagation for 2 layered architecture
      w1 -= eta * dw1		# updates of the weights and biases with learning rate 
      w2 -= eta * dw2
      b1 -= eta * db1
      b2 -= eta * db2
      if counter < 599:   # controls the completion of 1 epoch, batch size (100) * 600 iteration = 60000 iteration = 1 epoch
        counter += 1
      else:
        counter = 0
        train_input, train_target = shuffle(train_input, train_target)  # shuffle training set and training input
      
    if (i % 600 == 0 and i > 0) or i == 29999:	
      a = compute_error(test_input, test_target, w1, b1, w2, b2)
      print("Test error: ", a, i)
      #if i % 30000 == 0 and a > 5 and i != 0: # update the learning rate after 50th epoch where the error does not converge
      #  eta = eta / 2.5
print(torch.cuda.is_available())

train_input = torch.from_numpy(np.load('data/train_input.npy'))
train_target = torch.from_numpy(np.load('data/train_target.npy'))
test_input = torch.from_numpy(np.load('data/test_input.npy'))
test_target = torch.from_numpy(np.load('data/test_target.npy'))

#normalize both train and test dataset by dividing them 255
train_input =  train_input / 255
test_input = test_input / 255

two_layered_MLP_784_300_10(train_input, train_target, test_input, test_target) #training for 2 layer MLP configuration

#mnist_train_set = data.MNIST(root='./data', train=True, download=True, transform=None)
#mnist_test_set = data.MNIST(root='./data', train=False, download=True, transform=None)
#train_input = mnist_train_set.data.view(-1, 784).double()
#train_t = mnist_train_set.targets
#test_input = mnist_test_set.data.view(-1, 784).double()
#test_t = mnist_test_set.targets
#test_target = torch.zeros((len(test_t), 10), dtype=float)
#train_target = torch.zeros((len(train_t), 10), dtype=float)
#for i in range (0, 60000):
#    train_target[i][train_t[i]] = 1
#for i in range (0, 10000):
#    test_target[i][test_t[i]] = 1    
  #np.save('2_layered_arch/2_layered_W1_784_300',w1) # save weights and biases as numpy arrays
  #np.save('2_layered_arch/2_layered_W2_300_10',w2)
  #np.save('2_layered_arch/2_layered_B1_300_1',b1)
  #np.save('2_layered_arch/2_layered_B2_10_1',b2)
#def part_1(test_input, test_target):
#	w1 = torch.from_numpy(np.load('2_layered_arch/2_layered_W1_784_300.npy'))
#	w2 = torch.from_numpy(np.load('2_layered_arch/2_layered_W2_300_10.npy'))
#	b1 = torch.from_numpy(np.load('2_layered_arch/2_layered_B1_300_1.npy'))
#	b2 = torch.from_numpy(np.load('2_layered_arch/2_layered_B2_10_1.npy'))	
#	print('The total error for 2 layered architecture configuration: ', compute_error(test_input, test_target, w1, b1, w2, b2))
#part_1(test_input, test_target)
#from pyJoules.energy_meter import measure_energy
#from pyJoules.device.rapl_device import RaplPackageDomain
#from pyJoules.device.rapl_device import RaplDramDomain
#from pyJoules.device.nvidia_device import NvidiaGPUDomain
#from pyJoules.handler.csv_handler import CSVHandler

#csv_handler = CSVHandler('result.csv')
#@measure_energy(handler=csv_handler)
#
#def foo():
#    part_1(test_input, test_target)
#
#for _ in range(1):
#    foo()

#timestamp  : monitored function launching time
#tag        : tag of the measure, if nothing is specified, this will be the function name
#duration   : function execution duration
#device_name: power consumption of the device device_name in uJ
#csv_handler.save_data()

#flatened_trace = csv_handler._flaten_trace()
#attributes = flatened_trace[0].energy.keys()
#print(csv_handler._gen_sample_line(flatened_trace[0], attributes))
#for sample in flatened_trace:
#    if sample in ["core_0", "uncore_0", "nvidia_gpu_0"]:


#np.save('train_input.npy', train_input)
#np.save('test_input.npy', test_input)
#np.save('train_target.npy', train_target)
#np.save('test_target.npy', test_target)

#def part_2(test_input, test_target):
#	w1 = torch.from_numpy(np.load('3_layered_arch/3_layered_W1_784_300.npy'))
#	w2 = torch.from_numpy(np.load('3_layered_arch/3_layered_W2_300_100.npy'))
#	w3 = torch.from_numpy(np.load('3_layered_arch/3_layered_W3_100_10.npy'))
#	b1 = torch.from_numpy(np.load('3_layered_arch/3_layered_B1_300_1.npy'))
#	b2 = torch.from_numpy(np.load('3_layered_arch/3_layered_B2_100_1.npy'))	
#	b3 = torch.from_numpy(np.load('3_layered_arch/3_layered_B3_10_1.npy'))
#	print('The total error for 3 layered architecture configuration: ', compute_error_2(test_input, test_target, w1, b1, w2, b2, w3, b3))
