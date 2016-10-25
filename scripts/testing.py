import os
import json
import time

jobs_path = 'H:\\Entwicklung\\ConvNN\\auto_trainer\\my_tools\\jobs.json'


def get_next_job():
    jobs_dict = json.load(open(jobs_path, 'r'))
    for tmp_job in jobs_dict:
        if jobs_dict[tmp_job]['ignore']:
            continue
        if jobs_dict[tmp_job]['name'] not in finished_job_names:
            print jobs_dict[tmp_job]['name']+' not in:'
            print finished_job_names
            return jobs_dict[tmp_job], True
    return None, False


if __name__ == '__main__':
    finished_job_names = []
    while True:
        job, do_work = get_next_job()
        if not do_work:
            break
        print 'doing job '+job['name']
        time.sleep(10)
        finished_job_names.append(job['name'])
        print 'adding name, new list:'
        print finished_job_names

