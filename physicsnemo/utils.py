import importlib
from types import SimpleNamespace

def build_from_module(cfg):
    modname = cfg['module']
    classname = cfg['class']
    args = cfg['args']
    module = importlib.import_module(modname)
    model_cls = getattr(module, classname)
    cfg_obj = SimpleNamespace(**args)
    return model_cls(cfg_obj)