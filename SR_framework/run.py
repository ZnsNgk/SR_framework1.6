import subprocess
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Auto run")
    parser.add_argument("cfg_file", help = "config file", type = str)
    parser.add_argument("--py3", action = "store_true")
    parser.add_argument("--train", action = "store_true")
    parser.add_argument("--test", action = "store_true")
    parser.add_argument("--unload", action = "store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = args.cfg_file
    is_python3 = args.py3
    is_train = args.train
    is_test = args.test
    is_unload = args.unload
    if is_train:
        cmd = ""
        if is_python3:
            cmd += "python3 "
        else:
            cmd += "python "
        cmd = cmd + "train.py " + model
        print(cmd)
        try:
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode != 0:
                return
        except:
            return
    if is_test:
        cmd = ""
        if is_python3:
            cmd += "python3 "
        else:
            cmd += "python "
        cmd = cmd + "test.py " + model + " --all"
        print(cmd)
        try:
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode != 0:
                return
        except:
            return
    if is_unload:
        cmd = ""
        if is_python3:
            cmd += "python3 "
        else:
            cmd += "python "
        cmd = cmd + "unload.py " + model
        print(cmd)
        try:
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode != 0:
                return
        except:
            return

if __name__ == "__main__":
    main()