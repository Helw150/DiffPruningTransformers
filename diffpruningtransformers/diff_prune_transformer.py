import math
import torch
from torchreparam import ReparamModule
from torch import nn
from torch.nn import functional as F


def hard_sigmoid(x):
    return x.clamp(0, 1000).clamp(-1000, 1)


class L0Norm(nn.Module):
    def __init__(self, origin_shape, alpha_init=5, l=-1.5, r=1.5):
        """
        Base class of layers using L0 Norm
        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(L0Norm, self).__init__()
        self._size = origin_shape
        print(self._size)
        self.alpha = nn.Parameter(torch.zeros(self._size) + alpha_init)
        self.register_buffer("uniform", torch.zeros(self._size))
        self.l = l
        self.r = r
        self.lower_upper_ratio = math.log(-self.l / self.r)

    def _get_mask(self):
        if self.training:
            self.uniform.uniform_()
            s = torch.autograd.Variable(self.uniform).clamp_(0.0001, 0.9999)
            s = F.sigmoid(torch.log(s) - torch.log(1 - s) + self.alpha)
            u = s * (self.r - self.l) + self.l
            penalty = F.sigmoid(self.alpha - self.lower_upper_ratio).sum()
        else:
            s = F.sigmoid(self.alpha) * (self.r - self.l) + self.l
            penalty = torch.tensor(0)
            print(hard_sigmoid(s))
        return hard_sigmoid(s), penalty


class DiffPruningTransformer(torch.nn.Module):
    def __init__(self, parent_model, device):
        super(ParamXLM, self).__init__()
        self.lm = ReparamModule(parent_model.base_model)
        self.add_module("l0_norm", L0Norm(self.lm.flat_param.shape))
        self.patch_weight = torch.zeros_like(self.lm.flat_param)
        self.patch_weight.requires_grad = True
        self.patch_weight = torch.nn.Parameter(self.patch_weight)
        self.device = device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        mask, penalty = self.l0_norm._get_mask()
        penalty = penalty
        patch = mask * self.patch_weight
        flat_params = self.lm.flat_param.detach() + patch
        flat_params = flat_params
        outputs = self.lm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            flat_param=flat_params,
        )
        sequence_output = outputs[0]

        outputs = outputs[:1] + (penalty,) + outputs[1:]
        return sequence_output

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def load_pretrained(self, path):
        model.load_state_dict(torch.load(path))
