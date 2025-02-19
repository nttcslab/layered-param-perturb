import pandas as pd
import io
import ast


class FromDict:
    def __init__(self, arg):
        for k, v in arg.items():
            if isinstance(v, dict):
                self.__dict__[k] = FromDict(v)
            else:
                self.__dict__[k] = v

    def print(self):
        print(self.__dict__)
        for k, v in self.__dict__.items():
            if isinstance(v, FromDict):
                print(k, end='')
                v.print()


def read_log(logfile, epochs):
    with open(logfile, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line[0] == '{':
            cfg_dict = ast.literal_eval(line)
            cfg = FromDict(cfg_dict)
        if line[0] == '#':
            last_i = i
    region_start = last_i+1
    region_end = last_i+1+epochs+1
    df = pd.read_table(io.StringIO(''.join(lines[region_start:region_end])))
    return cfg, df
