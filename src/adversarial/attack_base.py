####################################################
# This code is transferred from the torchattacks library
# https://adversarial-attacks-pytorch.readthedocs.io/en/latest/
#
# All credits go to the original author
# Adjusted by Amon Soares de Souza
####################################################

import time
from collections import OrderedDict
from collections.abc import Iterable
import random
import math

import torch
from torch.utils.data import DataLoader, TensorDataset

def wrapper_method(func):
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get('_attacks').values():
            eval("atk."+func.__name__+"(*args, **kwargs)")
        return result
    return wrapper_func


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_model_training_mode`.
    """
    def __init__(self, name, model, model_trms, write_protocol=False, protocol_file='', target_mode='default'):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self._attacks = OrderedDict()
        
        self.set_model(model)
        self.model_trms = model_trms
        self.device = next(model.parameters()).device
        print(f'device for attack used: {self.device}')
        self.return_type = 'float'

        # Controls attack mode.
        self.attack_mode = 'default'
        self.supported_mode = ['default', 'targeted']
        self.targeted = True if target_mode in ['least_likely', 'most_likely'] else False
        self._target_map_function = None
        self.set_target_mode(target_mode)

        # Controls when normalization is used.
        self.normalization_used = None
        self._normalization_applied = None

        # Controls model mode during attack.
        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False
        
        if write_protocol:
            self.protocol_dir = "/".join(protocol_file.split('/')[:-2]) + '/'
            self.protocol_file = protocol_file

    def forward(self, inputs, labels=None, *args, **kwargs):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_eps_in_range(self, l2_bound, method='high'):
        low, high = self.get_iq_range_from_l2_bound(l2_bound)
        if method == 'random':
            l2_bound = random.uniform(low, high)
        elif method == 'high':
            l2_bound = high
        eps = self.compute_eps_by_l2_bound(l2_bound)
        return eps

    def get_l2_from_bounds(self, l2_bound, method='high'):
        low, high = self.get_iq_range_from_l2_bound(l2_bound)
        if method == 'random':
            l2_bound = random.uniform(low, high)
        elif method == 'high':
            l2_bound = high
        return l2_bound
        

    def compute_eps_by_l2_bound(self, l2_bound):
        """
        Given that every pixel in the l_inf norm contributes equally to the l2 distance
        Assumes that the incoming image size is 3 x 224 x 224
        """
        return math.sqrt((l2_bound**2)/(3*224*224))
    
    def get_iq_range_from_l2_bound(self, l2_bound):
        if l2_bound <= 4.0:
            low, high = 0.0, 4.0
        elif l2_bound <= 8.0:
            low, high = 4.0, 8.0
        elif l2_bound <= 15.0:
            low, high = 8.0, 15.0
        elif l2_bound <= 23.0:
            low, high = 15.0, 23.0
        elif l2_bound <= 38.0:
            low, high = 23.0, 38.0
        else:
            low, high = 38.0, 134.0
        return low, high

    def get_l2(self, orig_img, adv_img):
        distance = (orig_img - adv_img).pow(2).sum(dim=(1,2,3)).sqrt()
        return distance

    def set_target_mode(self, mode):
        if mode == 'least_likely':
            self.set_mode_targeted_least_likely()
        elif mode == 'most_likely':
            self.set_mode_targeted_most_likely()
        else:
            print('WARNING: set_target_mode was set to random. If unwanted, change "target_mode" arg to either "least_likely" or "most_likely".')
            self.set_mode_default()
        
    @wrapper_method
    def set_model(self, model):
        self.model = model
        self.model_name = str(model).split("(")[0]

    
    def get_logits(self, inputs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        return logits

    @wrapper_method
    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag
    
    @wrapper_method
    def set_device(self, device):
        self.device = device

    @wrapper_method
    def set_normalization_used(self, mean, std):
        self.normalization_used = {}
        n_channels = len(mean)
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used['mean'] = mean
        self.normalization_used['std'] = std
        self._normalization_applied = True

    def normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return inputs*std + mean

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self.attack_mode

    @wrapper_method
    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self.attack_mode = 'default'
        self.targeted = False
        print("Attack mode is changed to 'default.'")

    @wrapper_method
    def _set_mode_targeted(self, mode):
        if "targeted" not in self.supported_mode:
            raise ValueError("Targeted mode is not supported.")
        self.targeted = True
        self.attack_mode = mode
        print("Attack mode is changed to '%s'."%mode)

    @wrapper_method
    def set_mode_targeted_by_function(self, target_map_function):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda inputs, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)

        """
        self._set_mode_targeted('targeted(custom)')
        self._target_map_function = target_map_function

    @wrapper_method
    def set_mode_targeted_random(self):
        r"""
        Set attack mode as targeted with random labels.
        Arguments:
            num_classses (str): number of classes.

        """
        self._set_mode_targeted('targeted(random)')
        self._target_map_function = self.get_random_target_label

    @wrapper_method
    def set_mode_targeted_least_likely(self, kth_min=1):
        r"""
        Set attack mode as targeted with least likely labels.
        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)

        """
        self._set_mode_targeted('targeted(least-likely)')
        assert (kth_min > 0)
        self._kth_min = kth_min
        self._target_map_function = self.get_least_likely_label

    @wrapper_method
    def set_mode_targeted_most_likely(self):
        r"""
        Set attack mode as targeted with most likely labels that is not the true label.
        """
        self._set_mode_targeted('targeted(most-likely)')
        self._target_map_function = self.get_most_likely_label

    @wrapper_method
    def set_return_type(self, type):
        r"""
        Set the return type of adversarial inputs: `int` or `float`.

        Arguments:
            type (str): 'float' or 'int'. (Default: 'float')

        .. note::
            If 'int' is used for the return type, the file size of 
            adversarial inputs can be reduced (about 1/4 for CIFAR10).
            However, if the attack originally outputs float adversarial inputs
            (e.g. using small step-size than 1/255), it might reduce the attack
            success rate of the attack.

        """
        if type == 'float':
            self.return_type = 'float'
        elif type == 'int':
            self.return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def get_return_type(self):
        r"""
        Get the return type of adversarial inputs: `int` or `float`.
        """
        return self.return_type

    @wrapper_method
    def set_model_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    @wrapper_method
    def _change_model_mode(self, given_training):
        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()
        else:
            self.model.eval()

    @wrapper_method
    def _recover_model_mode(self, given_training):
        if given_training:
            self.model.train()


    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False,
             save_predictions=False, save_clean_inputs=False, save_type='float'):
        r"""
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_inputs (bool): True for saving clean inputs (Default: False)

        """
        if save_path is not None:
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)
        given_training = self.model.training

        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    adv_inputs_type_changed = self.to_type(adv_inputs, 'float')
                    outputs = self.get_output_with_eval_nograd(adv_inputs_type_changed)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (pred == labels.to(self.device))
                    correct += right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate l2 distance
                    delta = (adv_inputs_type_changed - inputs.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))
                    l2 = torch.cat(l2_distance).mean().item()

                    # Calculate time computation
                    progress = (step+1)/total_batch*100
                    end = time.time()
                    elapsed_time = end-start

                    if verbose:
                        self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')

            if save_path is not None:
                adv_input_list.append(self.to_type(adv_inputs.detach().cpu(), save_type))
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)
                save_dict = {'adv_inputs':adv_input_list_cat, 'labels':label_list_cat}

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict['preds'] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(self.to_type(inputs.detach().cpu(), save_type))
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict['clean_inputs'] = input_list_cat

                save_dict['save_type'] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    @staticmethod
    def _save_print(progress, rob_acc, l2, elapsed_time, end):
        print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t' \
              % (progress, rob_acc, l2, elapsed_time), end=end)

    @staticmethod
    def load(load_path, batch_size=128, shuffle=False,
             load_predictions=False, load_clean_inputs=False):
        save_dict = torch.load(load_path)
        keys = ['adv_inputs', 'labels']

        if load_predictions:
            keys.append('preds')
        if load_clean_inputs:
            keys.append('clean_inputs')

        if save_dict['save_type'] == 'int':
            save_dict['adv_inputs'] = save_dict['adv_inputs'].float()/255
            if load_clean_inputs:
                save_dict['clean_inputs'] = save_dict['clean_inputs'].float()/255

        adv_data = TensorDataset(*[save_dict[key] for key in keys])
        adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=shuffle)
        print("Data is loaded in the following order: [%s]"%(", ".join(keys)))
        return adv_loader

    @torch.no_grad()
    def get_output_with_eval_nograd(self, inputs):
        given_training = self.model.training
        if given_training:
            self.model.eval()
        outputs = self.get_logits(self.model_trms(inputs))
        if given_training:
            self.model.train()
        return outputs

    def get_target_label(self, inputs, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function is None:
            raise ValueError('target_map_function is not initialized by set_mode_targeted.')
        target_labels = self._target_map_function(inputs, labels)
        return target_labels

    @torch.no_grad()
    def get_least_likely_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_most_likely_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        label_mask = torch.full_like(labels.view(-1,1), -1e+03).to(torch.float32)
        outputs.scatter_(1, labels.view(-1,1), label_mask)
        _, target_labels = torch.max(outputs, dim=1)
        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_random_target_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l)*torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @staticmethod
    def to_type(inputs, type):
        r"""
        Return inputs as int if float is given.
        """
        if type == 'int':
            if isinstance(inputs, torch.FloatTensor) or isinstance(inputs, torch.cuda.FloatTensor):
                return (inputs*255).type(torch.uint8)
        elif type == 'float':
            if isinstance(inputs, torch.ByteTensor) or isinstance(inputs, torch.cuda.ByteTensor):
                return inputs.float()/255
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")
        return inputs

    def __call__(self, inputs, labels, *args, **kwargs):
        given_training = self.model.training
        self._change_model_mode(given_training)

        if self._normalization_applied is True:
            inputs = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
            adv_inputs = self.to_type(adv_inputs, self.return_type)

            adv_inputs = self.normalize(adv_inputs)
            self._set_normalization_applied(True)
        else:
            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
            adv_inputs = self.to_type(adv_inputs, self.return_type)

        self._recover_model_mode(given_training)

        return adv_inputs

    def __repr__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack', 'supported_mode']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info['attack_mode'] = self.attack_mode
        info['return_type'] = self.return_type
        info['normalization_used'] = True if self.normalization_used is not None else False

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name != 'text_cond_dict':
            attacks = self.__dict__.get('_attacks')

            # Get all items in iterable items.
            def get_all_values(items, stack=[]):
                    if (items not in stack):
                        stack.append(items)
                        if isinstance(items, list) or isinstance(items, dict):
                            if isinstance(items, dict):
                                items = (list(items.keys())+list(items.values()))
                            for item in items:
                                yield from get_all_values(item, stack)
                        else:
                            if isinstance(items, Attack):
                                yield items
                    else:
                        if isinstance(items, Attack):
                            yield items

            for num, value in enumerate(get_all_values(value)):
                attacks[name+"."+str(num)] = value
                for subname, subvalue in value.__dict__.get('_attacks').items():
                    attacks[name+"."+subname] = subvalue
