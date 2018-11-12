
import numpy as np

np.random.seed(1)

class mlp():
    """
    Multi layer perceptron for logistic regression, ordinary least squares and classification.
    ####FUNCTIONS####:
    activation functions; sigmoid. Used in hidden layer and output on logreg and classification
    include_bias: adds bias vector on last index on input vector 
    forward, forward_classy: feeds inputs vector forward in network returning an output
    backward, backward_classy: sends error back and updates weights
    train, train_classy: trains model with entire training data
    error, crossentropyerror: calculates error on entire validation set i.e one epoch
    earlystopping: setting a number of epochs and tracking error for each epoch. takes care of overfitting
    accuracy_acore: calculates accuracy on testdata
    """
    def __init__(self, inputs, targets, nhidden, eta):
        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden
        self.eta = eta
        self.beta = 1.0
        self.nvectors = inputs.shape[0]
        self.ntargets = targets.shape[1]
        self.ninputs = inputs.shape[1]
        self.hiddenacc = np.zeros(self.nhidden)
        self.output = np.zeros(targets.shape[1])
        self.v = np.random.randn(*(self.ninputs + 1, self.nhidden))*0.001
        self.w = np.random.randn(*(self.nhidden + 1, self.ntargets))*0.001

    ##### activation functions
    def sigmoid(self, x):
        return 1./(1 + np.exp(-self.beta*x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def sigmoid_out(self, x):
        if x < 0.5:
            return 0
        else:
            return 1

    def include_bias(self, array):
        """
        Adds a bias vector on last index of array 
        """
        bias = -1
        return np.append(array, bias)

    def forward(self, inputs):
        """
        Takes in a single vector. 
        Use for regression case
        """
        inputs_tot = self.include_bias(inputs)
        h_chi = np.zeros(self.nhidden)
        a_chi = np.zeros(self.nhidden)
        h_kappa = np.zeros(self.ntargets)
        y_kappa = np.zeros(self.ntargets)
        
        # activation on 1st layer  
        h_chi = np.dot(inputs_tot, self.v)
        a_chi = self.sigmoid(h_chi)
        self.hiddenacc = a_chi                     
        
        a_chi_tot = self.include_bias(a_chi)   
        h_kappa = np.dot(a_chi_tot, self.w)
        
        # output
        y_kappa = h_kappa
        self.output = y_kappa
        return y_kappa
    
    def backward(self, inputs, targets):
        """
        Use for regression case
        Feeds error back and updates weights
        """
        updateV = np.zeros(np.shape(self.v))
        updateW = np.zeros(np.shape(self.w))
        inputs_tot = self.include_bias(inputs)

        hiddenacc_tot = self.include_bias(self.hiddenacc)

        #Error in outputlayer
        delO = (self.output - targets)                
        updateW = np.outer(hiddenacc_tot.T, delO)
        updateW = - self.eta*updateW
        
        #Error in hiddenlayer            
        delH = self.hiddenacc*(1.0 - self.hiddenacc)*np.outer(delO, self.w.T[:,:-1])

        updateV = np.outer(inputs_tot.T, delH)
        updateV = - self.eta*updateV

        #updateV and updateW are one smaller than self.v and self.w because of bias.
        self.v += updateV
        self.w += updateW  

    def forward_classy(self, inputs):
        """
        Takes in a single vector. 
        Use for classification case
        """
        inputs_tot = self.include_bias(inputs)
        h_chi = np.zeros(self.nhidden)
        a_chi = np.zeros(self.nhidden)
        h_kappa = np.zeros(self.ntargets)
        y_kappa = np.zeros(self.ntargets)
        
        # activation on 1st layer  
        h_chi = np.dot(inputs_tot, self.v)
        a_chi = self.sigmoid(h_chi)
        self.hiddenacc = a_chi                     

        # activation on 2nd layer
        a_chi_tot = self.include_bias(a_chi)   
        h_kappa = np.dot(a_chi_tot, self.w)
        
        # output
        y_kappa = self.sigmoid_out(h_kappa)
        self.output = y_kappa
        return y_kappa

    def backward_classy(self, inputs, targets):
        """
        Feeds error back and updates weights
        Use for classification case
        """
        updateV = np.zeros(np.shape(self.v))
        updateW = np.zeros(np.shape(self.w))
        inputs_tot = self.include_bias(inputs)

        hiddenacc_tot = self.include_bias(self.hiddenacc)
        
        #Error in outputlayer
        delO = self.sigmoid_derivative(self.output)*(self.output - targets)*sum(inputs.T)                
        updateW = np.outer(hiddenacc_tot.T, delO)
        updateW = - self.eta*updateW
        
        #Error in hiddenlayer            
        delH = self.hiddenacc*(1.0 - self.hiddenacc)*np.outer(delO, self.w.T[:,:-1])

        updateV = np.outer(inputs_tot.T, delH)
        updateV = - self.eta*updateV

        #updateV and updateW are one smaller than self.v and self.w because of bias.
        self.v += updateV
        self.w += updateW        

    def train(self, inputs, targets):
        """
        Train MLP with inputs. Use training data
        Used for regression case
        """
        for n in range(inputs.shape[0]):
            self.forward(inputs[n])
            self.backward(inputs[n], targets[n])

    def train_classy(self, inputs, targets):
        """
        Train MLP with inputs. Use training data
        Used for classification case
        """
        for n in range(inputs.shape[0]):
            self.forward_classy(inputs[n])
            self.backward_classy(inputs[n], targets[n])

    def error(self, validationset, validationstargets):
        """
        cost function for regression case
        returns error on one set of input vector. Hold out validation data 
        """
        error = np.zeros(validationset.shape[0])
        for i in range(validationstargets.shape[0]):
            predicted = self.forward(validationset[i])
            error[i] = np.linalg.norm(validationstargets[i] - predicted)**2
        print('------------------------------------------------------------------------------------------------')
        print(f'Error for Ising energies are {sum(error)/validationstargets.shape[0]} with {self.nhidden} hidden nodes and a {self.eta} learning rate')
        print('------------------------------------------------------------------------------------------------') 
        return sum(error)/validationstargets.shape[0]

    def crossentropyerror(self, validationset, validationstargets):
        """
        cost function for classification case
        returns error on one set of input vectors. Hold out validation data
        """
        crosserror = np.zeros(validationset.shape[0])
        for i in range(validationstargets.shape[0]):
            predicted = self.forward(validationset[i])
            crosserror[i] = np.linalg.norm(-validationstargets[i]*np.log10(predicted) - (1 - validationstargets[i])*np.log10(1 - predicted))
        return sum(crosserror)/validationstargets.shape[0]

    def earlystopping(self, inputs, targets, validationset, validationstargets):
        """
        This is the function initating the MLP calling functions above. 
        Trains network for a number of epochs
        Calculate the error for a set errorfunction (Can be changed)
        If error increases for more than 10 iterations, STOP
        """
        # increase epochs for more training. set to 400 - 500 for ising energies and to 4-10 for determining phase 
        epochs = 4 + 1    
        count = 0
        MLP_error = np.zeros(epochs - 1)
        epochs_final = 0
        for i in range(epochs - 1):
            self.train(inputs, targets)
            """
            BELOW change MLP_error for a suitable error function:
            For logistic regression and classification use: "self.crossentropyerror()"
            For regular regression use: "self.error()"
            """
            MLP_error[i] = self.crossentropyerror(validationset, validationstargets) 
            #MLP_error[i] = self.error(validationset, validationstargets)      
            if MLP_error[i - 1] < MLP_error[i]:
                count += 1
            else:
                count = 0
            if count == 10:
                print('Error increasing %d times in a row. STOP' % count)
                print('Final epoch is:', i)
                epochs_final = i
                indices = np.linspace(i + 1, epochs, epochs - i)
                MLP_error = np.delete(MLP_error, indices)
                break
        return MLP_error, epochs

    def accuracy_score(self, testset, testtargets):
        """
        return accuracy on testdata after training model in % 
        """
        accuracy = 0
        for n in range(testtargets.shape[0]):
            predictedoutput = self.forward_classy(testset[n])
            if predictedoutput == testtargets[n]:
                accuracy += 1
        accuracy = accuracy/testtargets.shape[0]*100
        print('------------------------------------------------------------------------------------------------')
        print(f'Accuracy for classification MLP calculated to {accuracy} % with {self.nhidden} hidden nodes and a {self.eta} learning rate')
        print('------------------------------------------------------------------------------------------------') 
        return accuracy
        
