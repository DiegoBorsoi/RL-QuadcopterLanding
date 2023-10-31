from imitation.data import types

#base_name = "saved-rollout_dim-2_plat-0_n-100_PPO-3"
#base_name = "saved-rollout_dim-2_plat-1_n-100_PPO-11"
#base_name = "saved-rollout_dim-2_plat-1_n-100_PPO-12"
base_name = "saved-rollout_dim-2_plat-1_n-200_PPO-12"

files_names = []
file_count = 16

for i in range(1, file_count + 1):
    files_names.append(base_name + "_" + str(i) + ".npz")

output_file = base_name + "_tot.npz"


for i in range(len(files_names)):
    if len(files_names[i].split('.')) < 2:
        files_names[i] += ".npz"

right_rollouts = []

for f in files_names:
    l = types.load(f)
    for r in l:
        if sum(r.rews) > -1000:
            right_rollouts.append(r)

print("Num right rollouts: " + str(len(right_rollouts)))

types.save(output_file, right_rollouts)

print("Saved")