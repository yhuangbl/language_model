import os
import random

num_models = 10
seeds = random.sample(range(1, 100), num_models)
dir_path = "models/ensemble/"
command1 = "python main.py -mode train -saved_model "
command2 = ".h5 -student_id 20213421 -batch_size 32 -embedding_dim 750 -hidden_size 500 -drop 0.5 -gpu 2,3 "\
           + "-model lstm1 -seed "

for i in range(num_models):
    random_seed = seeds[i]
    cmd = command1 + dir_path + "lstm1." + str(i) + command2 + str(random_seed)
    print("Execute " + cmd)
    os.system(cmd)
