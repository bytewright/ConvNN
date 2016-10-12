import os
import json

#output_dir = '/home/ellerch/caffeProject/auto_trainer_output'
output_dir = 'H:\\Entwicklung\\ConvNN\\auto_trainer\\my_tools'
stats_dict = {'0':'1', 'abc':['a','b','c'], '123':{'x':'y','g':'h'}}
stats_dict = {'0':{'name':'bla', 'solver_path':'bla', 'model_path':'bla'}, '1':{'name':'bla2', 'solver_path':'bla', 'model_path':'bla'}}
log_path = os.path.join(output_dir, "jobs.json")
#json.dump(stats_dict, open(log_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
log_path = 'H:\\Entwicklung\\ConvNN\\jobs\\jobs.json'
a=json.load(open(log_path, 'r'))
for job in a:
    print a[job]
