import os
import json
import glob
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Unload")
    parser.add_argument("cfg_file", help = "config file", type = str)
    args = parser.parse_args()
    return args

def load_json(args):
    cfg_file = os.path.join("./config/", args + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)
    return config

def move_file(old_path, new_path):
    _, file_name = os.path.split(old_path)
    new_path = os.path.join(new_path, file_name)
    shutil.move(old_path, new_path)

def unload():
    args = parse_args()
    config = load_json(args.cfg_file)
    model_name = config["system"]["model_name"]
    save_path = './' + model_name + '/'
    if not os.path.exists(save_path):
        os.mkdir(model_name)
    folder_list = ['./trained_model/', './test_result/', './log/', './demo_output/']
    for folder in folder_list:
        if folder != './log/':
            model_folder = folder + model_name + '/'
            new_path = './' + model_name + folder.replace('.','')
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            if not os.path.exists(model_folder):
                continue
            file_list = []
            file_list.extend(glob.glob(os.path.join(model_folder, "*.*")))
            for f in file_list:
                move_file(f, new_path)
            os.rmdir(model_folder)
            print(folder.replace('.', '').replace('/', '') + ' success!')
        else:
            log_file = folder + model_name + '.log'
            if not os.path.exists(log_file):
                continue
            move_file(log_file, save_path)
            print("Log file success!")
    move_file('./config/' + args.cfg_file + '.json', save_path)
    print('Config file success!')

if __name__ == "__main__":
    unload()