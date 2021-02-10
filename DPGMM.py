import numpy as np
from collections import Counter
import pickle, grapheme

np.random.seed(17)


class DPGMM:
    def __init__(self, A, R, base_distribution):
        '''
          Dirichlet Process Geometric Mixture Model
          Parameters:
            A = Parameter of Dirichlet distribution
            R = No. of restaurants
            base_distribution = Base distribution
            data_dir = Directory of data
        '''
        self.K = {}
        self.A = A
        self.R = R
        self.N = 0
        self.X = []
        self.base_distribution = base_distribution
        self.iterations = 0
        self.current_iteration = 0

        self.cluster_dict = {}
        self.all_clusters = {}
        self.assigned_clusters = {}
        self.unassigned_clusters = {}
        self.Z = {}

        self.performance = {}
        self.cluster_size_per_iter = {}

        for r in range(self.R):
            self.K[r] = 20

            self.cluster_dict[r] = []
            self.all_clusters[r] = []
            self.assigned_clusters[r] = []
            self.unassigned_clusters[r] = []
            self.Z[r] = []

            self.cluster_size_per_iter[r] = []
            self.performance[r] = []

        self.location = "models/"

    @classmethod
    def withmodel(cls, model):
        return model

    ############################################################
    ################### UTILITY FUNCTIONS ######################
    ############################################################

    def normalize(self, prob):
        '''
          Normalize the given probabilities so that sum = 1
          Parameters:
            prob = Probabilites
          Output:
            norm = Normalized Probability
        '''
        norm = [p / np.sum(prob) for p in prob]
        return norm

    ############################################################

    ############################################################
    ################### DPGMM Features #########################
    ############################################################

    def get_estimated_mixture_proportions(self):
        '''
          Return estimated mixture proportions calculated using Gibbs sampling
          Output:
            es_mixture_proportions = Estimated mixture proportions for each cluster k
        '''
        es_mixture_proportions = {}
        for r in range(self.R):
            c = Counter(self.Z[r])
            es_mixture_proportions[r] = [c[k] / self.N for k in self.assigned_clusters[r]]
        return es_mixture_proportions

    ############################################################

    ############################################################
    ####################### INFERENCE ##########################
    ############################################################

    def generate_splits(self, word):
        '''
          Generate all possible splits
          Parameter:
            word = Word to be split
          Output:
            splits = List of all possible splits
        '''
        splits = []
        for s in range(grapheme.length(word) + 1):
            stem = grapheme.slice(word, 0, s)
            stem = stem if (grapheme.length(stem) > 0) else '$'
            suffix = grapheme.slice(word, s)
            suffix = suffix if (grapheme.length(suffix) > 0) else '$'
            splits.append((stem, suffix))
        return splits

    def split(self, word):
        '''
          Split the given word using Gibbs Sampling inference
          Parameter:
            word  = Word to split
          Output:
            split = Split with max probability
        '''
        splits = self.generate_splits(word)

        probability = []

        Z = {}
        P = {}

        for split in splits:
            p_k = {}

            for r in range(self.R):
                prob = self.calculate_cluster_assignment_probability_for_x(split[r], r)
                p_k[r] = self.normalize(prob)
                Z[r] = np.random.choice(len(p_k[r]), 1, p=p_k[r])[0]
                P[r] = p_k[r][Z[r]]

            probability.append(P[0] * P[1])

        max_probability_index = np.argmax(probability)

        split = splits[max_probability_index]

        return split

    def calculate_cluster_assignment_probability_for_x(self, x, r):
        '''
          Calculate the probability of data x belonging to each cluster k
          Parameter:
            x = Value of data
            r = Restaurant r
          Output:
            probability = Probability of data x belonging to each cluster
        '''

        probability = []

        # For each cluster k
        for k in self.assigned_clusters[r]:
            # Calculate prior * likelihood
            c_k = Counter(self.Z[r])[k]
            p = (c_k / (self.N + self.A))

            probability.append(p)

        # Calculate likelihood for new cluster
        p_k_new = self.calculate_probability(x, [])

        # Calculate prior * likelihood for new cluster
        p_new = (self.A / (self.N + self.A)) * p_k_new

        probability.append(p_new)

        return probability

    ############################################################
    ################### CLUSTER MANAGEMENT #####################
    ############################################################

    def initialize(self):
        '''
          Assign random clusters to each data point using Uniform Distribution
          Output:
            Z = List of assigned clusters to each data point x
        '''
        # Create a table dictionary for each restaurant
        for r in range(self.R):

            all_clusters_dict = {}
            all_clusters_set = set([x[r] for x in self.X])

            for index, cluster in enumerate(all_clusters_set):
                all_clusters_dict[index] = cluster

            self.cluster_dict[r] = all_clusters_dict
            self.all_clusters[r] = [k for k in all_clusters_dict.keys()]

        # Use Z as a latent variable to denote cluster assignments
        Z = {}
        for r in range(self.R):
            Z[r] = []

            total_clusters = len(self.all_clusters[r])
            uniform_probability = [1 / total_clusters for k in range(total_clusters)]
            selected_clusters = np.random.choice(self.all_clusters[r], self.K[r], p=uniform_probability)

            uniform_probability = [1 / self.K[r] for i in range(self.K[r])]

            for i in range(self.N):
                Z[r].append(np.random.choice(selected_clusters, 1, p=uniform_probability)[0])

            self.update_current_clusters(Z[r], r)

        return Z

    def update_current_clusters(self, Z, r):
        '''
          Update the current cluster assignments
        '''
        C = Counter(Z)
        self.assigned_clusters[r] = [k for k in C.keys()]
        self.K[r] = len(self.assigned_clusters[r])
        self.unassigned_clusters[r] = list(set(self.all_clusters[r]) - set(self.assigned_clusters[r]))

    def remove_empty_cluster(self, cluster_data, r):
        '''
          Remove empty clusters
          Paramter:
            cluster_data = Data assigned to clusters
        '''
        for k in self.assigned_clusters[r]:
            if len(cluster_data[k]) < 1:
                self.assigned_clusters[r].remove(k)
                self.unassigned_clusters[r].append(k)

        self.K[r] = len(Counter(self.assigned_clusters[r]))

    def get_new_cluster(self, r):
        '''
          Return a new cluster index from unassigned clusters
          Parameter:
            r = Restaurant r
        '''
        count = len(self.unassigned_clusters[r])
        uniform_prob = [1 / count for k in self.unassigned_clusters[r]]
        k = np.random.choice(self.unassigned_clusters[r], 1, p=uniform_prob)[0]

        return k

    ###########################################################

    ############################################################
    ################### GIBSS SAMPLING #########################
    ############################################################

    def remove_data_point(self, i, r):
        '''
          Remove data point xi from data
          Parameters:
            i = Index of data point to removed
            r = Restaurant r
          Output:
            cluster_data_without_xi = Dictionary consisting of data assigned to each cluster without data xi
        '''
        cluster_data_without_xi = {}

        for k in self.assigned_clusters[r]:
            cluster_data_without_xi[k] = []

        for index, value in enumerate(self.X):
            if index == i:
                continue
            k = self.Z[r][index]
            cluster_data_without_xi[k].append(value[r])

        return cluster_data_without_xi

    def calculate_probability(self, data_i, cluster_data):
        '''
          Calculate posterior predictive probability of new data x
          Parameters:
            data_i        = New data
            cluster_data  = Data of cluster k
          Output:
            p = Calculated probability
        '''
        beta_geom = self.base_distribution
        beta_geom.update_parameters(cluster_data)

        x = grapheme.length(data_i)
        p = beta_geom.probability(x)

        return p

    def calculate_cluster_assignment_probability(self, i, r):
        '''
          Calculate the posterior probability of data i belonging to each cluster k
          Parameter:
            i = Index of data
            r = Restaurant r
          Output:
            posterior_probability = Probability of data i belonging to each cluster
        '''

        # Remove data (xi, zi)
        cluster_data_without_x_i = self.remove_data_point(i, r)

        posterior_probability = []

        # Remove empty clusters for restaurant r
        self.remove_empty_cluster(cluster_data_without_x_i, r)

        # For each cluster k
        for k in self.assigned_clusters[r]:
            x_i = self.X[i][r]

            # Calculate likelihood
            p_k = self.calculate_probability(x_i, cluster_data_without_x_i[k])

            # Calculate prior * likelihood
            c_k = Counter(self.Z[r])[k]
            p = (c_k / (self.N + self.A - 1)) * p_k

            posterior_probability.append(p)

        # Calculate likelihood for new cluster
        p_k_new = self.calculate_probability(x_i, [])

        # Calculate prior * likelihood for new cluster
        p_new = (self.A / (self.N + self.A - 1)) * p_k_new

        posterior_probability.append(p_new)

        return posterior_probability

    def resume_fit(self):
        '''
          Resume fitting of data from last iteration
        '''
        self.fit(self.X, self.iterations, self.current_iteration + 1)

    def fit(self, data, iterations, start_iteration_at=1):
        '''
          Fit a Dirichlet Process Geometric Mixture Model using Gibbs Sampling
          parameters:
            data       = Data to fit to GMM
            iterations = No. of iterations
        '''
        self.X = data
        self.N = len(self.X)
        self.iterations = iterations
        if start_iteration_at == 1: self.Z = self.initialize()

        iteration = start_iteration_at

        print("Iteration \t Current clusters")

        while iteration < (self.iterations + 1):

            print(iteration, "\t\t", self.K)

            performance = {}

            for r in range(self.R):
                performance[r] = []

            # For each data xi
            for i in range(self.N):

                # For each restaurant
                for r in range(self.R):
                    # Calculate the posterior probability of data i belonging to each cluster k
                    posterior_probability = self.calculate_cluster_assignment_probability(i, r)

                    # Normalize
                    normalized_probability = self.normalize(posterior_probability)

                    new_cluster_index = self.get_new_cluster(r)

                    # Sample using posterior probability
                    self.Z[r][i] = \
                    np.random.choice(self.assigned_clusters[r] + [new_cluster_index], 1, p=normalized_probability)[0]

                    # Update clusters
                    self.update_current_clusters(self.Z[r], r)

                    # Calculate performance
                    performance[r].append(np.log(np.sum(posterior_probability)))

            for r in range(self.R):
                self.performance[r].append(np.sum(performance[r]))
                self.cluster_size_per_iter[r].append(self.K[r])

            self.current_iteration = iteration

            if (iteration % 10 == 0):
                current_model = DPGMM.withmodel(self)
                file_name = "model_" + str(int(self.N / 2)) + "_" + str(iteration) + ".p"
                pickle.dump(current_model, open(self.location + file_name, 'wb'))
                print("Saved " + file_name)

            iteration = iteration + 1