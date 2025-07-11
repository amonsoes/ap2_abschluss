from src.model.openclipnet import dict_pretrain
from src.model.openclipnet import OpenClipLinear

class ClipDetectorLoader:

    def __init__(self, weights_dir='./weights'):
        self.weights_dir = weights_dir