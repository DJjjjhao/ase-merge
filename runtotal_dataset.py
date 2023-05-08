import json
import math
import time
import subprocess
import os
import sys 
import pickle
if __name__ == '__main__':
    file_name = sys.argv[1]

    total_raw_data_path = 'RAW_DATA'
    all_raw_base, all_raw_a, all_raw_b, all_raw_res = json.load(open('%s/raw_data'%(total_raw_data_path)))
    all_num = len(all_raw_base)
    jobs = []
    max_num = 100
    each_num = 1000
    for i in range(math.ceil(all_num / each_num)):
        while True:
            run_num = 0 
            for x in jobs:
                if x.poll() is None:
                    run_num += 1
            if run_num < max_num:
                break
            time.sleep(1)
        start = i * each_num
        end = min((i + 1) * each_num, all_num)
        p = subprocess.Popen("python %s %s %s"%(file_name, start, end), shell=True)
        jobs.append(p)
        time.sleep(1)
    for job in jobs:
        job.wait()