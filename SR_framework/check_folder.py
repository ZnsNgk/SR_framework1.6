import os

dir_path = './'

dir_list = ['config', 'data', 'data/train', 'data/test', 'demo_input', 'demo_output', 'log', 'test_result', 'trained_model']

def check():
    for dir_name in dir_list:
        dir_name = dir_path + dir_name
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

if __name__ == "__main__":
    check()