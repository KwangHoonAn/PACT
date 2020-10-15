import torch
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from torch.autograd import Function
from torch import nn
from torch.autograd import Function, Variable

# k = 8
class ActFn(Function):
	@staticmethod
	def forward(ctx, x, alpha, k):
		ctx.save_for_backward(x, alpha)
		# y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
		y = torch.clamp(x, min = 0, max = alpha.item())
		scale = (2**k - 1) / alpha
		y_q = torch.round( y * scale) / scale
		return y_q

	@staticmethod
	def backward(ctx, dLdy_q):
		# Backward function, I borrowed code from
		# https://github.com/obilaniu/GradOverride/blob/master/functional.py
		# We get dL / dy_q as a gradient
		x, alpha, = ctx.saved_tensors
		# Weight gradient is only valid when [0, alpha]
		# Actual gradient for alpha,
		# By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
		# dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range 
		lower_bound      = x < 0
		upper_bound      = x > alpha
		# x_range       = 1.0-lower_bound-upper_bound
		x_range = ~(lower_bound|upper_bound)
		grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
		return dLdy_q * x_range.float(), grad_alpha, None

def quantize_k(r_i, k):
	scale = (2**k - 1)
	r_o = torch.round( scale * r_i ) / scale
	return r_o

class DoReFaQuant(Function):
	@staticmethod
	def forward(ctx, r_i, k):
		tanh = torch.tanh(r_i).float()
		# scale = 2**k - 1.
		# quantize_k = torch.round( scale * (tanh / 2*torch.abs(tanh).max() + 0.5 ) ) / scale
		r_o = 2*quantize_k( tanh / (2*torch.max(torch.abs(tanh)).detach()) + 0.5 , k) - 1
		# r_o = 2 * quantize_k - 1.
		return r_o

	@staticmethod
	def backward(ctx, dLdr_o):
		# due to STE, dr_o / d_r_i = 1 according to formula (5)
		return dLdr_o, None


class Conv2d(nn.Conv2d):
	def __init__(self, in_places, out_planes, kernel_size, stride=1, padding = 0, groups=1, dilation=1, bias = False, bitwidth = 8):
		super(Conv2d, self).__init__(in_places, out_planes, kernel_size, stride, padding, groups, dilation, bias)
		self.quantize = DoReFaQuant.apply
		self.bitwidth = bitwidth

	def forward(self, x):
		vhat = self.quantize(self.weight, self.bitwidth)
		y = F.conv2d(x, vhat, self.bias, self.stride, self.padding, self.dilation, self.groups)
		return y

class Linear(nn.Linear):
	def __init__(self, in_features, out_features, bias = True, bitwidth = 8):
		super(Linear, self).__init__(in_features, out_features, bias)
		self.quantize = DoReFaQuant.apply
		self.bitwidth = bitwidth
	def forward(self, x):
		vhat = self.quantize(self.weight, self.bitwidth)
		y = F.linear(x, vhat, self.bias)
		return y