import os
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='Path to the folder containing the images')
parser.add_argument('--project_name', required=True, help='Name of Project folder')
args=parser.parse_args()

if os.path.exists('project'):
    os.system('rm -rf project')


os.system('mkdir project')
os.system('mkdir project/images')

os.system('cp -r ' + args.path + '/* project/images')


# sdcs
os.system('colmap automatic_reconstructor \
        --workspace_path project \
        --image_path project/images')
