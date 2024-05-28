from functools import partial
import cv2
import tensorflow as tf
import numpy as np

import mxnet as mx
from generate_anchor import generate_anchors_fpn, nonlinear_pred, generate_runtime_anchors
from numpy import frombuffer, uint8, concatenate, float32, block, maximum, minimum, prod
from mxnet.ndarray import waitall, concat
from functools import partial
import time
