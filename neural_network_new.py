import numpy
import scipy.special

class neuralNetwork:

    def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate):

        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

        print(self.wih)
        print(self.who)

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, input_list,targets_list):

        inputs = numpy.array(input_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)


        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_erros)

        self.who += self.lr * numpy.dot((
            output_errors * final_outputs * (1.0 - final_outputs)
        ),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((
            hidden_errors * hidden_outputs * (1.0 - hidden_outputs)
        ),numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        print(inputs)
        print(final_outputs)

        pass

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
n.query([[6,9,2],[4,3,2],[3,2,1]])