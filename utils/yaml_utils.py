import yaml
from addict import Dict
import shutil
import os


def _parse_option_value(value):
    return yaml.safe_load(value)


def read_yaml(fpath=None):
    with open(fpath, mode="r", encoding="utf-8") as file:
        yml = yaml.safe_load(file)
        return Dict(yml)
    
def update_config_from_options(config, options):
    for option in options:
        key, value = option.split('=')
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d or d[k] is None:
                d[k] = Dict()
            d = d[k]
        d[keys[-1]] = _parse_option_value(value)
    return config

def change_yaml_by_options(yaml_path, options):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    for option in options:
        key, value = option.split('=')
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d or d[k] is None:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = _parse_option_value(value)

    with open(yaml_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(config, file, sort_keys=False)



def save_yaml(args,yaml_path,options):
    dst_path = os.path.join(args.Logs.now_log_dir,os.path.basename(yaml_path))
    shutil.copyfile(yaml_path,dst_path)
    if options != None:
        change_yaml_by_options(dst_path,options)
