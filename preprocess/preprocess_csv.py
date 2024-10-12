from os.path import dirname, abspath
import sys
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
from src.preprocessing import *
from preprocess.config import cfg_csv as cfg

def main():
    preprocess_csv(cfg)

if __name__=='__main__':
    main()