import numpy as np
class EinsteinSolid:
    '''
    EinsteinSolid that is able to perform MonteCarlo simulations

    ATTRIBUTES:

    oscillators: an array of oscillators that store
    N: total number of quanta in Einstein Solid
    q: average # of quanta per oscillator
    exchanges_per_update: self-explanatory
    '''

    #Constructor
    def __init__(self, N, q, exchanges_per_update):
        
        if(q < 1):
            self.oscillators = np.zeros(N)
            self.oscillators[int((1-q)*len(self.oscillators)):] += 1
        else:
            self.oscillators = np.ones( N, dtype=int) * q
        self.N = N
        self.q = q
        self.exchanges_per_update = exchanges_per_update

        
    #Methods:
    def updateEinMC(self):
        '''
        Updates an Einstein-Solid by performing several exchanges
        of quanta between oscillators.

        Returns: None. Updates the "solid" array in place.

        Parameters:
        solid: the Einstein solid - int NumPy array
            * solid.size = # of oscillators
            * solid[i] = n_i - number of quanta of the ith oscillator 
        num_exchanges: # of quanta to exchange - int
        '''
        # Perform num_exchanges exchanges
        for i in range(int(self.exchanges_per_update)):
            
            # Choose a random recipient atom
            recipient = np.random.randint(0, self.oscillators.size)
            
            # Find a random donor atom with at least one quantum
            donor = np.random.randint(0, self.oscillators.size)
            while self.oscillators[donor] == 0:
                donor = np.random.randint(0, self.oscillators.size)
            
            # Exchange enery
            self.oscillators[donor] -= 1
            self.oscillators[recipient] += 1
            self.Q = self.N * self.q


    def sampleDist(self, num_samples):
        '''
        Parameters:

        num_times: the number of samples

        returns a list containing the samples of each update
        '''
        #Creating a list to store sample arrays
        samples = []

        #Iterating over the samples
        for i in range(num_samples):
            self.updateEinMC()
            samples.append(self.oscillators)
        return samples
    
    def sampleProbDist(self, num_samples):
        '''
        Parameters:

        num_times: the number of samples
        
        returns a DataFrame containing the mean probabiliy distribution of n values to N_max
        alongside the mean^2 and uncertainty
        '''
        import pandas as pd
        
        samples = []

        for i in range(num_samples):
            self.updateEinMC()
            samples.append(((np.histogram(self.oscillators, bins=[i for i in range(int(self.Q) + 2)])[0]) / self.N))
        
        #Creating DataFrame with data
        df = pd.DataFrame(samples)
        
        #Creating mean, mean squared, uncertainty, and probability  rows
        df.loc['Mean'] = df.mean()
        df.loc['Mean Sq.'] = (df ** 2).mean()
        df.loc["Uncertainty"] = ((df.loc['Mean Sq.'] - (df.loc['Mean'] ** 2)) ** 0.5) / (num_samples ** 0.5)
        df = df.loc[{'Mean', 'Mean Sq.', 'Uncertainty'}]
        df.loc[:, ~(df == 0.0).any()]

        #Returning just the mean and standard deviation rows
        return df.loc[{'Mean', 'Mean Sq.', "Uncertainty"}].where(df != 0.0).dropna(axis = 1)


    def plotProbDist(self, df, errorbar = True, custom_graph = False, set_graph = None):
        '''
        Parameters:
        df: dataframe of probability distribution of n
        errorbar: sets graph as an errobar. If false, just a regular plot. default = True
        custom_graph: boolean if user wants to implement their own graph. If not, function creates its own graph. Default = False
        set_graph: if custom_graph == True, then allows user to input their own graph. Default = None

        returns a plot of the probability distribution of n
        '''
        #Create plot
        if(custom_graph == False):
            fig1, graph1 = plt.subplots(figsize = (10,10))
            #Setting up graph
            graph1.set_title("P(n) vs. n", fontsize = 20)
            graph1.set_ylabel("P(n)", fontsize = 16)
            graph1.set_xlabel("n", fontsize = 16)
        else:
            graph1 = set_graph

        if(errorbar == True):
            graph1.errorbar(df.columns, df.loc["Mean"], yerr = df.loc['Uncertainty'], linestyle='None', marker='o', label = 'q = ' + str(self.q))
        else:
            graph1.plot(df.columns, df.loc["Mean"], label = 'q = ' + str(self.q))
    
        return graph1
        
    def calcEntropy(self, df):
        '''
        Parameters:
        df: dataframe of probability distribution of n
    
        returns tuple of the entropy of the oscillator and the corresponding temperature
        '''
  
        #calculating tempterature values:
        t_val = temp(self.N, (self.N * self.q))

        #Creating a new row for s/kb:
        df.loc['s'] = ((np.log(df.loc['Mean']) * df.loc['Mean']) * -1)

        #Creating new column that sums rows(specifically for s)
        df['sigma_s'] = df.sum(axis = 1)
        
        return (df['sigma_s'].loc['s'], t_val)