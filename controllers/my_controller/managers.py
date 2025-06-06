# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.


import sys
import os
if os.name == 'nt' and sys.version_info >= (3, 8):  # we need to explicitly list the folders containing the DLLs
    robotis_libraries = os.path.join(os.environ['WEBOTS_HOME'], 'projects', 'robots', 'robotis', 'darwin-op', 'libraries')
    os.add_dll_directory(os.path.join(robotis_libraries, 'managers'))
    os.add_dll_directory(os.path.join(robotis_libraries, 'robotis-op2'))



from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from .test import _managers
else:
    import _managers

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__ # type: ignore

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


import controller
DGM_NMOTORS = _managers.DGM_NMOTORS
class RobotisOp2GaitManager(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, robot, iniFilename):
        _managers.RobotisOp2GaitManager_swiginit(self, _managers.new_RobotisOp2GaitManager(robot, iniFilename))
    __swig_destroy__ = _managers.delete_RobotisOp2GaitManager

    def isCorrectlyInitialized(self):
        return _managers.RobotisOp2GaitManager_isCorrectlyInitialized(self)

    def setXAmplitude(self, x):
        return _managers.RobotisOp2GaitManager_setXAmplitude(self, x)

    def setYAmplitude(self, y):
        return _managers.RobotisOp2GaitManager_setYAmplitude(self, y)

    def setAAmplitude(self, a):
        return _managers.RobotisOp2GaitManager_setAAmplitude(self, a)

    def setMoveAimOn(self, q):
        return _managers.RobotisOp2GaitManager_setMoveAimOn(self, q)

    def setBalanceEnable(self, q):
        return _managers.RobotisOp2GaitManager_setBalanceEnable(self, q)

    def start(self):
        return _managers.RobotisOp2GaitManager_start(self)

    def step(self, duration):
        return _managers.RobotisOp2GaitManager_step(self, duration)

    def stop(self):
        return _managers.RobotisOp2GaitManager_stop(self)

# Register RobotisOp2GaitManager in _managers:
_managers.RobotisOp2GaitManager_swigregister(RobotisOp2GaitManager)

DMM_NMOTORS = _managers.DMM_NMOTORS
class RobotisOp2MotionManager(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _managers.RobotisOp2MotionManager_swiginit(self, _managers.new_RobotisOp2MotionManager(*args))
    __swig_destroy__ = _managers.delete_RobotisOp2MotionManager

    def isCorrectlyInitialized(self):
        return _managers.RobotisOp2MotionManager_isCorrectlyInitialized(self)

    def playPage(self, id, sync=True):
        return _managers.RobotisOp2MotionManager_playPage(self, id, sync)

    def step(self, duration):
        return _managers.RobotisOp2MotionManager_step(self, duration)

    def isMotionPlaying(self):
        return _managers.RobotisOp2MotionManager_isMotionPlaying(self)

# Register RobotisOp2MotionManager in _managers:
_managers.RobotisOp2MotionManager_swigregister(RobotisOp2MotionManager)



