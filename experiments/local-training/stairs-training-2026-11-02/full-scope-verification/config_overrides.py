"""Apply existing configuration paths, preserving types and recording overrides."""
import math


def check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite configuration value')
    if isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_finite(item)


def apply_paths(root, updates):
    for path, value in updates.items():
        check_finite(value)
        parts = path.split('.')
        if any(not key or key.startswith('_') for key in parts):
            raise ValueError('invalid configuration path')
        obj = root
        for key in parts[:-1]:
            obj = obj[key] if isinstance(obj, dict) else getattr(obj, key)
        key = parts[-1]
        old = obj[key] if isinstance(obj, dict) else getattr(obj, key)
        if callable(old):
            raise ValueError('functions must not be replaced by configuration values')
        if isinstance(old, bool) and not isinstance(value, bool):
            raise ValueError('expected boolean')
        if isinstance(old, int) and not isinstance(old, bool) and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError('expected integer')
        if isinstance(old, (tuple, list)):
            if not isinstance(value, (tuple, list)):
                raise ValueError('expected sequence')
            value = type(old)(value)
        if isinstance(obj, dict):
            obj[key] = value
        else:
            setattr(obj, key, value)
