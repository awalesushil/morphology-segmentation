from BetaGeometric import BetaGeometric
from DPGMM import DPGMM

base_distribution = BetaGeometric(alpha = 1.0, beta = 2.0)
model = DPGMM(A = 0.5, R = 2, base_distribution = base_distribution)

# Inference
print(model.split("नेपालका"))