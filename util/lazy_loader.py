"""
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py
"""

import types
import importlib

class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies."""
    def __init__(self, local_name, parent_module_globals, name, warning=None):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._warning = warning
        self.__module__ = name.rsplit(".", 1)[0]
        self.__wrapped__ = None
        super(LazyLoader, self).__init__(name)
    def _load(self):
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        if self._warning:
            self._warning = None
        self.__dict__.update(module.__dict__)
        return module
    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)
    def __repr__(self):
        return f"<LazyLoader {self.__name__} as {self._local_name}>"
    def __dir__(self):
        module = self._load()
        return dir(module)
