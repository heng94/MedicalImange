import yaml
import types


class Cfg:
    def __init__(self, cfg_file):
        self.cfg = types.DynamicClassAttribute()
        with open(cfg_file) as f:
            self.cfg_data = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    def get_cfg(self,):
        for key, value in self.cfg_data.items():
            if not hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
        return self.cfg


config = Cfg(cfg_file='./configs/baseline.yaml')
cfg = config.get_cfg()
print(cfg.exp)
print('1')

