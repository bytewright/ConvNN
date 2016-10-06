import os
import json

output_dir = '/home/ellerch/caffeProject/auto_trainer_output'
stats_dict = {'0':'1', 'abc':['a','b','c'], '123':{'x':'y','g':'h'}}
log_path = os.path.join(output_dir, "stats.json")
json.dump(stats_dict, open(log_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))

a=json.load(open(log_path, 'r'))
print a.keys()
