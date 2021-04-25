import datetime
import os

def log(string, file_name, need_time=False):
    save_str = ''
    if need_time:
        save_str = str(datetime.datetime.now().replace(microsecond=0)) + ' |'
    save_str = save_str + string
    print(save_str)
    log_path = os.path.join("./log", file_name + ".log")
    f = open(log_path,'a')
    f.write(save_str)
    f.write('\n')
    f.close()

def check_log_file(file_name):
    file_name = os.path.join("./log", file_name + ".log")
    if os.path.exists(file_name):
        os.remove(file_name)