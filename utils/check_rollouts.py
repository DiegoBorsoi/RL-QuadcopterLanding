from imitation.data import types
import sys

args = sys.argv

file_name = "saved-rollout" if len(args) < 2 else args[1]

if len(file_name.split('.')) < 2:
    file_name += ".npz"

l = types.load(file_name)

num_l = len(l)
print("Num. rollouts: " + str(num_l))

tot_rew = 0
for a in l:
    tot_rew += sum(a.rews)

print("Rew mean: " + str(tot_rew/num_l))

c = 0
for a in l:
    if sum(a.rews) > -1000:
        c += 1

print("Rew giusti (over -1000): " + str(c))
print("Giusti perc: " + str(c / num_l * 100) + "%")