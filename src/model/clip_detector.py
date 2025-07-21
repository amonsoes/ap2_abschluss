import os
import yaml

from src.model.openclipnet import dict_pretrain
from src.model.openclipnet import OpenClipLinear
from src.model.resnetmod import resnet50

class ClipDetectorLoader:

    def __init__(self, device, weights_dir='./weights'):
        self.weights_dir = weights_dir
        self.device = device
    
    def load_model(self):
        model_name = 'clipdet_latent10k_plus'
        _, model_path, arch, norm_type, patch_size = self.get_config(model_name, weights_dir=self.weights_dir)
        model = self.load_weights(self.create_architecture(arch), model_path)
        model = model.to(self.device).eval()
        return model, norm_type, patch_size
    
    def get_config(self, model_name, weights_dir='./weights'):
        with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
            data = yaml.load(fid, Loader=yaml.FullLoader)
        model_path = os.path.join(weights_dir, model_name, data['weights_file'])
        return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

    def load_weights(self, model, model_path):
        from torch import load
        dat = load(model_path, map_location='cpu')
        if 'model' in dat:
            if ('module._conv_stem.weight' in dat['model']) or \
            ('module.fc.fc1.weight' in dat['model']) or \
            ('module.fc.weight' in dat['model']):
                model.load_state_dict(
                    {key[7:]: dat['model'][key] for key in dat['model']})
            else:
                model.load_state_dict(dat['model'])
        elif 'state_dict' in dat:
            model.load_state_dict(dat['state_dict'])
        elif 'net' in dat:
            model.load_state_dict(dat['net'])
        elif 'main.0.weight' in dat:
            model.load_state_dict(dat)
        elif '_fc.weight' in dat:
            model.load_state_dict(dat)
        elif 'conv1.weight' in dat:
            model.load_state_dict(dat)
        else:
            print(list(dat.keys()))
            assert False
        return model

    def create_architecture(self, name_arch, pretrained=False, num_classes=1):
        if name_arch == "res50nodown":

            if pretrained:
                model = resnet50(pretrained=True, stride0=1, dropout=0.5).change_output(num_classes)
            else:
                model = resnet50(num_classes=num_classes, stride0=1, dropout=0.5)
        elif name_arch == "res50":

            if pretrained:
                model = resnet50(pretrained=True, stride0=2).change_output(num_classes)
            else:
                model = resnet50(num_classes=num_classes, stride0=2)
        elif name_arch.startswith('opencliplinear_'):
            model = OpenClipLinear(num_classes=num_classes, pretrain=name_arch[15:], normalize=True)
        elif name_arch.startswith('opencliplinearnext_'):
            model = OpenClipLinear(num_classes=num_classes, pretrain=name_arch[19:], normalize=True, next_to_last=True)
        else:
            assert False
        return model