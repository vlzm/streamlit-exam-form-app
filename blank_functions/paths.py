import os

# get path to repo
path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# get path to data
path_to_data = os.path.join(path_to_repo, "data")

# get path to ref_pics
path_to_ref_pics = os.path.join(path_to_data, "ref_pics")
