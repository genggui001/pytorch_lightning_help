#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: ./nb/template.ipynb
import sys
if __name__ == '__main__': sys.path.append('..')
from jupyter_dev_template.common import export_notebook

def main(name):
    print(f"hello {name}")

assert 1==1

import fire
import xs_lib
import os
if not xs_lib.common.IN_JUPYTER and __name__=="__main__" and os.environ.get("STOP_FIRE", "") != "true":
    fire.Fire(main)