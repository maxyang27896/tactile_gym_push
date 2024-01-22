import argparse
import os
import shutil
import json
import numpy as np
import torch


def check_dir(dir):

    # check save dir exists
    if os.path.isdir(dir):
        str_input = input('Save Directory already exists, would you like to continue (y,n)? ')
        if not str2bool(str_input):
            exit()
        else:
            # clear out existing files
            empty_dir(dir)

def empty_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_json_obj(obj, name):
    with open(name + '.json', 'w') as fp:
        json.dump(obj, fp)

def load_json_obj(name):
    with open(name + '.json', 'r') as fp:
        return json.load(fp)

def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False

def print_sorted_dict(dict):
    for key in sorted(iter(dict.keys())):
        print('{}:{}'.format(key, dict[key]) )

def quaternion_multiply(quaternion0, quaternion1):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1

    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0])

def get_inverse_quaternion(orn):
    conj_orn = orn.copy()
    conj_orn[:3] = -conj_orn[:3] 
    norm_orn = np.linalg.norm(orn)

    return conj_orn/norm_orn

def get_orn_diff(orn_1, orn_2):
    """
    Calculate the difference between two orientaion quaternion in the same frame 
    of reference
    """
    return quaternion_multiply(orn_1, get_inverse_quaternion(orn_2))
