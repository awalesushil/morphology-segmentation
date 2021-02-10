from Utilities import Utilities
from BetaGeometric import BetaGeometric
from DPGMM import DPGMM
import matplotlib.pyplot as plt

# No. of restaurants == No. of splits

base_distribution = BetaGeometric(alpha = 1.0, beta = 2.0)
model = DPGMM(A = 0.5, R = 2, base_distribution = base_distribution)

utils = Utilities()

data_file = "data/raw_text.txt"

tokens = utils.tokenize(data_file)
splits = utils.generate_splits(no_of_splits=2, tokens=tokens)

model.fit(splits, iterations = 5)

es_proportions = model.get_estimated_mixture_proportions()

print('Estimated Proprtions')
print('----------------------')

for r in range(model.R):
    print('\n')
    print('For Restaurant', r)
    print('----------------------')
    print('Estimated no. of clusters:', model.K[r])
    print('Proportions:', [round(k, 3) for k in es_proportions[r]])


for r in range(model.R):
  plt.figure(figsize=(20, 5))
  plt.plot(model.performance[r])
  plt.title("Log-Likelihood of Restaurant " + str(r))
  plt.xlabel("Number of Iteration")
  plt.ylabel("log p(x_i,z_i|X_-i, Z_-i, theta)")
  plt.xticks([i for i in range(1, model.iterations + 1)])

  plt.show()


for r in range(model.R):
  plt.figure(figsize=(20, 5))
  plt.plot(model.cluster_size_per_iter[r])
  plt.title("Cluster size per iteration for restaraunt " + str(r))
  plt.xlabel("Number of Iteration")
  plt.ylabel("Cluster Size")
  plt.xticks([i for i in range(1, model.iterations + 1)])

  plt.show()