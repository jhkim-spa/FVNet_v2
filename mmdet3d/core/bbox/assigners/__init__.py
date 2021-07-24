from mmdet.core.bbox import AssignResult, BaseAssigner, MaxIoUAssigner
from .inbox_assigner import InBoxAssigner

__all__ = ['BaseAssigner', 'MaxIoUAssigner', 'AssignResult',
	   'InBoxAssigner']
