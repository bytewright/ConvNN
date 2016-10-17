import os
import json
snapshot_path = '/home/ellerch/caffeProject/auto_trainer_output/_gute_runs/my_conv3+4+5_to_1/job7/job7'
best_file = 1
best_file_name = ''
for f in os.listdir(snapshot_path):
    if f.endswith(".caffemodel"):
        iter_num = int(f.replace('_iter_', '').replace('.caffemodel'))
        if iter_num > best_file:
            best_file = iter_num
            best_file_name = f
print os.path.join(snapshot_path, best_file_name)
