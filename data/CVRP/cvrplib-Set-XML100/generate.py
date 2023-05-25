import os, random
import numpy as np


def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed)


seed_everything(2023)
size = [125, 150, 175, 200]
A = [1, 2, 3]
B = [1, 2, 3]
C = [1, 2, 3, 4, 5, 6, 7]
D = [1, 2, 3, 4, 5, 6]
for s in size:
    for i in range(5):
    	a = random.sample(A, 1)[0]
    	b = random.sample(B, 1)[0]
    	c = random.sample(C, 1)[0]
    	d = random.sample(D, 1)[0]
    	os.system("python generator.py {} {} {} {} {} 1 1".format(s, a, b, c, d))
