from ..builder import DETECTORS
from .sparse_rcnn import SparseRCNN


@DETECTORS.register_module()
class FuMA(SparseRCNN):
    '''
    We hack and build our model into Sparse RCNN framework implementation
    in mmdetection.
    '''
