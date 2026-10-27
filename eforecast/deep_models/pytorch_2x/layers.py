import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from eforecast.deep_models.pytorch_2x.ts_transformers.ts_transformer_models import get_transfromer_model
from eforecast.deep_models.pytorch_2x.conv_att.cbam import CBAM
from eforecast.deep_models.pytorch_2x.conv_att.cbam_ver2 import CBAM_ver2
from eforecast.deep_models.pytorch_2x.conv_att.GeoDANet import GeoDualAttentionNetwork

import torchvision.transforms as transforms
import timm
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from einops import rearrange
from einops import einsum

act_funcs = nn.ModuleDict({'elu': nn.ELU(),
                           'sigmoid': nn.Sigmoid(),
                           'relu': nn.ReLU(),
                           'gelu': nn.GELU(),
                           'tanh': nn.Tanh(),
                           'lrelu': nn.LeakyReLU(),
                           'prelu': nn.PReLU()
                           })


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if module.out_proj.weight is not None:
            nn.init.xavier_uniform_(module.out_proj.weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)


class cnn_attention(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(cnn_attention, self).__init__()
        self.act_func = act_funcs[params['act_func']]
        self.name_scope = name_scope
        self.layer_id = layer_id
        self.name = f'{name_scope}_conv_att_{layer_id}'
        self.params = params
        if params['attention'] in {'cbam', 'cbam1', 'att'}:
            shape, layer_id, kernels, pool_size = conv_util(shape, size, layer_id)
            self.shape = shape
            self.size = int(size)
            if params['attention'] == 'cbam1':
                self.conv1 = CBAM_ver2(shape[1], reduction_ratio=1)
            elif params['attention'] == 'cbam':
                self.conv1 = CBAM(shape[1], reduction_factor=params['reduction_factor'], kernel_size=3)
            elif params['attention'] == 'att':
                self.conv1 = cnn_att(shape, params['filters'], act_func=params['act_func'])
                self.apply(_init_weights)
            x = torch.rand(10, *shape[1:])
            x_shape1 = list(x.size())
            if len(x_shape1) == 3:
                x = torch.unsqueeze(x, dim=1)
            if len(x_shape1) == 5:
                x = rearrange(x, 'b c1 c2 h w -> b (c1 c2) h w')
            # transform the input

            # Perform the usual forward pass
            x = self.conv1(x)
            if len(x_shape1) == 3:
                x = torch.squeeze(x, dim=1)
            self.output_shape = x.shape
        elif params['attention'] in {'datt', 'geodatt', 'timegeodatt'}:
            if params['attention'] == 'geodatt':
                self.conv1 = GeoDualAttentionNetwork(shape)

            x = torch.rand(10, *shape[1:])
            x_shape1 = list(x.size())
            if len(x_shape1) == 3:
                x = torch.unsqueeze(x, dim=1)
                x = torch.unsqueeze(x, dim=1)

            # Perform the usual forward pass
            x = self.conv1(x)
            if len(x_shape1) == 3:
                x = torch.squeeze(x, dim=1)
                x = torch.squeeze(x, dim=1)
            self.output_shape = x.shape


    def forward(self, x):
        x_shape1 = list(x.size())
        if self.params['attention'] in {'cbam', 'cbam1', 'att'}:
            if len(x_shape1) == 3:
                x = torch.unsqueeze(x, dim=1)
            if len(x_shape1) == 5:
                x = rearrange(x, 'b c1 c2 h w -> b (c1 c2) h w')
        else:
            if len(x_shape1) == 3:
                x = torch.unsqueeze(x, dim=1)
                x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        return x

def conv_util(shape, size, layer_id):
    if not isinstance(shape, list):
        shape = list(shape)
    if len(shape) == 3:
        shape.insert(1, 1)
    if isinstance(layer_id, str):
        layer_id = layer_id.split('_')[0]
    if len(shape) == 5:
        shape = [shape[0], shape[1] * shape[2], shape[3], shape[4]]
    if isinstance(layer_id, str):
        layer_id = int(layer_id.split('_lag')[0])
    kernels = []
    pool_size = []
    for s in shape[-2:]:
        if (s / 2) > size:
            kernels.append(size)
            if s > 50:
                pool_size.append(3)
            elif s > 10:
                pool_size.append(2)
            else:
                pool_size.append(1)
        else:
            kernels.append(int(s / 2)) if int(s / 2) > 1 else kernels.append(1)
            if s > 12:
                pool_size.append(2) if int(s / 2) > 1 else pool_size.append(1)
            else:
                pool_size.append(1)
    return shape, layer_id, kernels, pool_size

class cnn_att(nn.Module):
    def __init__(self, shape, filters, act_func='relu'):
        super(cnn_att, self).__init__()

        self.act_func = act_funcs[act_func]
        self.conv1 = nn.Conv2d(shape[1], int(filters), kernel_size=3)
        self.conv2 = nn.Conv2d(int(filters), int(2 * filters), kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(2 * filters * 2 * 2), 256)
        self.fc2 = nn.Linear(256, np.prod(shape[1:]))

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(shape[1], int(filters), kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(int(filters), int(2 * filters), kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(int(2 * filters * 2 * 2), int(4 * filters)),
            nn.ReLU(True),
            nn.Linear(int(4 * filters), 3 * 2)
        )

        self.apply(_init_weights)
        # Initialize the weights/bias with identity transformation (must come after general init)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = rearrange(xs, 'b c h w -> b (c h w)')
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        shape = list(x.size())
        # transform the input
        x1 = self.stn(x)

        # Perform the usual forward pass
        x1 = self.act_func(F.max_pool2d(self.conv1(x1), 2))
        x1 = self.act_func(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        x1 = rearrange(x1, 'b c h w -> b (c h w)')
        x1 = self.act_func(self.fc1(x1))
        x1 = self.fc2(x1)
        x1 = F.sigmoid(x1)
        x1 = rearrange(x1, 'b (c h w) -> b c h w', c=shape[1], h=shape[2], w=shape[3])
        return x

class conv_2d(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(conv_2d, self).__init__()
        self.size = int(size)
        if self.size == 1:
            self.size = 2
            shape = [shape[0], shape[1], 5, 5]
            self.crop = True
        else:
            self.crop = False
        shape, layer_id, kernels, pool_size = conv_util(shape, self.size, layer_id)

        self.shape = shape
        self.act_func = act_funcs[params['act_func']]
        self.name_scope = name_scope
        self.layer_id = layer_id
        self.name = f'{name_scope}_conv_2d_{layer_id}'
        self.conv = nn.Conv2d(shape[1], int(params['filters']  * (int(layer_id) + 1)),
                              kernel_size=kernels,
                              stride=1,
                              padding="valid")
        if params['batch_norm']:
            self.bn = nn.BatchNorm2d(int(params['filters']  * (int(layer_id) + 1)))
        else:
            self.bn = None
        self.pool = nn.AvgPool2d(pool_size)
        x = torch.rand(10, *shape[1:])
        x_shape1 = list(x.size())
        if len(x_shape1) == 3:
            x = torch.unsqueeze(x, dim=1)
        if len(x_shape1) == 5:
            x = rearrange(x, 'b c1 c2 h w -> b (c1 c2) h w')
        x = self.conv(x) if self.bn is None else self.bn(self.conv(x))
        x = self.pool(self.act_func(x))
        if len(x_shape1) == 3:
            x = torch.squeeze(x, dim=1)

        self.output_shape = x.shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.conv.weight.size())}")

    def forward(self, x):
        x_shape1 = list(x.size())
        if len(x_shape1) == 3:
            x = torch.unsqueeze(x, dim=1)
        if len(x_shape1) == 5:
            x = rearrange(x, 'b c1 c2 h w -> b (c1 c2) h w')
        if self.crop:
            x = x[:, :, 7:12, 7:12]
        x = self.conv(x) if self.bn is None else self.bn(self.conv(x))
        x = self.pool(self.act_func(x))
        if len(x_shape1) == 3:
            x = torch.squeeze(x, dim=1)
        return x


class conv_3d(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(conv_3d, self).__init__()
        if not isinstance(shape, list):
            shape = list(shape)
        if len(shape) == 3:
            shape.insert(1, 1)
        if len(shape) == 4:
            shape.insert(1, 1)
        if not isinstance(shape, list):
            shape = list(shape)
        self.shape = shape
        self.size = int(size)
        self.act_func = act_funcs[params['act_func']]
        self.name_scope = name_scope
        self.layer_id = layer_id
        self.name = f'{name_scope}_conv_3d_{layer_id}'
        if isinstance(layer_id, str):
            layer_id = int(layer_id.split('_lag')[0])
        kernels = []
        pool_size = []
        for s in shape[-3:]:
            if (s / 2) > self.size:
                kernels.append(self.size)
                if s > 50:
                    pool_size.append(3)
                elif s > 10:
                    pool_size.append(2)
                else:
                    pool_size.append(1)
            else:
                kernels.append(int(s / 2)) if int(s / 2) > 1 else kernels.append(1)
                if s > 12:
                    pool_size.append(2) if int(s / 2) > 1 else pool_size.append(1)
                else:
                    pool_size.append(1)
        self.conv = nn.Conv3d(shape[1], int(params['filters']  * (int(layer_id) + 1)),
                              kernel_size=kernels,
                              stride=1,
                              padding="valid")
        if params['batch_norm']:
            self.bn = nn.BatchNorm3d(int(params['filters']  * (int(layer_id) + 1)))
        else:
            self.bn = None
        self.pool = nn.AvgPool3d(pool_size)
        x = torch.rand(10, *shape[1:])
        x_shape1 = list(x.size())
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, dim=-1)
            x = torch.unsqueeze(x, dim=1)
        if len(x.size()) == 4:
            x = torch.unsqueeze(x, dim=1)
        x = self.conv(x) if self.bn is None else self.bn(self.conv(x))
        x = self.pool(self.act_func(x))
        if len(x.size()) == 3:
            x = torch.squeeze(x, dim=-1)
            x = torch.squeeze(x, dim=1)
        if len(x.size()) == 4:
            x = torch.squeeze(x, dim=1)
        self.output_shape = x.shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.conv.weight.size())}")

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, dim=-1)
            x = torch.unsqueeze(x, dim=1)
        if len(x.size()) == 4:
            x = torch.unsqueeze(x, dim=1)
        x = self.conv(x) if self.bn is None else self.bn(self.conv(x))
        x = self.pool(self.act_func(x))
        if len(x.size()) == 3:
            x = torch.squeeze(x, dim=-1)
            x = torch.squeeze(x, dim=1)
        if len(x.size()) == 4:
            x = torch.squeeze(x, dim=1)
        return x


class time_distr_conv_2d(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(time_distr_conv_2d, self).__init__()
        if not isinstance(shape, list):
            shape = list(shape)
        if len(shape) == 3:
            shape.insert(1, 1)
        if len(shape) == 4:
            shape.insert(1, 1)
        self.shape = shape
        self.name_scope = name_scope
        B, T, C, H, W = shape
        self.T = T
        self.C = C
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_time_distr_{layer_id}'
        self.conv_list = nn.ModuleList()
        self.dense_list = nn.ModuleList()
        for i in range(T):
            self.conv_list.append(conv_2d([B, C, H, W], params, size, self.name_scope, f'{layer_id}_lag{i}'))
            self.dense_list.append(nn.Linear(int(np.prod(self.conv_list[i].output_shape[1:])),
                                             int(np.prod(self.conv_list[i].output_shape[2:]))))
        self.dense_output = int(np.prod(self.conv_list[0].output_shape[2:]))
        self.conv_output = self.conv_list[0].output_shape
        self.lstm = nn.LSTM(self.dense_output,
                            self.dense_output,
                            batch_first=True)
        self.flat = nn.Flatten()
        x = torch.rand([B, T, C, H, W])
        output = torch.tensor([])
        for i in range(self.T):
            output_t = self.conv_list[i](x[:, i, ...])
            output_t = self.flat(output_t)
            output_t = self.dense_list[i](output_t)
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        hidden_state = (torch.rand(1, B, self.dense_output), torch.rand(1, B, self.dense_output))
        lstm_out = self.lstm(output, hidden_state)[0]
        x = self.act_func(self.flat(lstm_out))
        x = rearrange(x, 'b (t w h) -> b t w h',
                      t=self.T, w=self.conv_output[-2], h=self.conv_output[-1])
        self.output_shape = x.shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(
            f"layer weights {self.name} has {self.shape[1]} conv of shape{list(self.conv_list[0].weight.size())}")

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, dim=1)
        if len(x.size()) == 4:
            x = torch.unsqueeze(x, dim=1)
        B, _, _, _, _ = x.size()
        output = torch.tensor([]).to(x.device)
        for i in range(self.T):
            output_t = self.conv_list[i](x[:, i, ...])
            output_t = self.flat(output_t)
            output_t = self.dense_list[i](output_t)
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        hidden_state = (torch.rand(1, B, self.dense_output).to(x.device),
                        torch.rand(1, B, self.dense_output).to(x.device))
        lstm_out = self.lstm(output, hidden_state)[0]
        x = self.act_func(self.flat(lstm_out))
        x = rearrange(x, 'b (t w h) -> b t w h',
                      t=self.T, w=self.conv_output[-2], h=self.conv_output[-1])
        return x


class time_distr_conv_3d(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(time_distr_conv_3d, self).__init__()
        if not isinstance(shape, list):
            shape = list(shape)
        if len(shape) == 3:
            shape.insert(1, 1)
        if len(shape) == 4:
            shape.insert(1, 1)
        self.shape = shape
        self.name_scope = name_scope
        B, T, C, H, W = shape
        self.size = size
        self.T = T
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_time_distr_{layer_id}'
        self.conv_list = nn.ModuleList()
        self.dense_list = nn.ModuleList()
        for i in range(T):
            self.conv_list.append(conv_3d([B, C, H, W], params, size, self.name_scope, f'{layer_id}_lag{i}'))
            self.dense_list.append(nn.Linear(int(np.prod(self.conv_list[i].output_shape[1:])),
                                             int(np.prod(self.conv_list[i].output_shape[2:]))))
        self.dense_output = int(np.prod(self.conv_list[0].output_shape[2:]))
        self.conv_output = self.conv_list[0].output_shape
        self.lstm = nn.LSTM(self.dense_output,
                            self.dense_output,
                            batch_first=True)

        self.flat = nn.Flatten()
        x = torch.rand([B, T, C, H, W])
        output = torch.tensor([])
        for i in range(self.T):
            output_t = self.conv_list[i](x[:, i, ...])
            output_t = self.flat(output_t)
            output_t = self.dense_list[i](output_t)
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        hidden_state = (torch.rand(1, B, self.dense_output), torch.rand(1, B, self.dense_output))
        lstm_out = self.lstm(output, hidden_state)[0]
        x = self.act_func(self.flat(lstm_out))
        x = rearrange(x, 'b (t c w h) -> b t c w h',
                      t=self.T, c=self.conv_output[-3], w=self.conv_output[-2], h=self.conv_output[-1])
        self.output_shape = x.shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(
            f"layer weights {self.name} has {self.shape[1]} conv of shape{list(self.conv_list[0].weight.size())}")

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, dim=1)
        if len(x.size()) == 4:
            x = torch.unsqueeze(x, dim=1)
        B = x.size()[0]
        output = torch.tensor([]).to(x.device)
        for i in range(self.T):
            output_t = self.conv_list[i](x[:, i, ...])
            output_t = self.flat(output_t)
            output_t = self.dense_list[i](output_t)
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        hidden_state = (torch.rand(1, B, self.dense_output).to(x.device),
                        torch.rand(1, B, self.dense_output).to(x.device))
        lstm_out = self.lstm(output, hidden_state)[0]
        x = self.act_func(self.flat(lstm_out))
        x = rearrange(x, 'b (t c w h) -> b t c w h',
                      t=self.T, c=self.conv_output[-3], w=self.conv_output[-2], h=self.conv_output[-1])
        return x


class lstm(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(lstm, self).__init__()
        self.shape = shape
        self.size = size
        self.is_lstm_output = params['is_lstm_output']
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_lstm_{layer_id}'
        if isinstance(shape, dict):
            try:
                B, Tpred, Ctime = shape['future_data' if 'future_data' in shape.keys() else 'future_calendar']
            except:
                B, Tpred, Ctime = shape['past_data' if 'past_data' in shape.keys() else 'past_calendar']
            self.obs_embed = None
            if 'past_calendar' in shape.keys() and 'future_calendar' in shape.keys():
                future_calendar = torch.cat([torch.rand(shape['past_calendar']),
                                             torch.rand(shape['future_calendar'])], 1)
            elif 'past_calendar' in shape.keys():
                future_calendar = torch.rand(shape['past_calendar'])
            elif 'future_calendar' in shape.keys():
                future_calendar = torch.rand(shape['future_calendar'])
            else:
                future_calendar = None


            if 'past_data' in shape.keys() and 'future_data' in shape.keys():
                future_data = torch.cat([torch.rand(shape['past_data']),
                                    torch.rand(shape['future_data'])], 1)
            elif 'past_data' in shape.keys():
                future_data = torch.rand(shape['past_data'])
            elif 'future_data' in shape.keys():
                future_data = torch.rand(shape['future_data'])
            else:
                future_data = None


            if 'past_obs' in shape.keys():
                past_obs = torch.rand(shape['past_obs'])
            else:
                past_obs = None


            past_obs_emd = self.obs_embed(past_obs) if self.obs_embed is not None else None
            past =[mat for mat in [past_obs, past_obs_emd] if mat is not None]
            future = [mat for mat in [future_data, future_calendar] if mat is not None]
            x_enc1 = torch.cat(past, -1)
            x_enc2 = torch.cat(future, -1)
            self.enc_in = max([x_enc1.shape[-1], x_enc2.shape[-1]])
            self.lstm = nn.ModuleList([nn.LSTM(x_enc1.shape[-1], int(size * self.enc_in), batch_first=True),
                         nn.LSTM(x_enc2.shape[-1], int(size * self.enc_in), batch_first=True),
                         nn.LSTM(int(size * self.enc_in), int(size * self.enc_in), num_layers=int(size), batch_first=True)])
            hidden_state = (torch.rand(1, B, int(size * self.enc_in)), torch.rand(1, B, int(size * self.enc_in)))
            lstm_out1, hidden_state = self.lstm[0](x_enc1, hidden_state)
            lstm_out1 = self.act_func(lstm_out1)
            lstm_out2, hidden_state = self.lstm[1](x_enc2, hidden_state)
            lstm_out2 = self.act_func(lstm_out2)
            hidden_state = (torch.rand(int(size), B, int(size * self.enc_in)), torch.rand(int(size), B, int(size * self.enc_in)))
            lstm_out, hidden_state = self.lstm[2](torch.cat([lstm_out1, lstm_out2], 1), hidden_state)
            self.output_shape = self.act_func(lstm_out)[:, -Tpred:, :].shape


        else:
            B = shape[0]
            Tpred = shape[1]
            x_dec = torch.rand(shape)
            self.enc_in = shape[-1]
            self.lstm = nn.LSTM(self.enc_in, int(size * self.enc_in), batch_first=True)
            hidden_state = (torch.rand(1, B, int(size * self.enc_in)), torch.rand(1, B, int(size * self.enc_in)))
            lstm_out = self.lstm(x_dec, hidden_state)[0]
            if self.is_lstm_output:
                self.output_shape = self.act_func(lstm_out)[:, -Tpred:, :].shape
            else:
                self.output_shape = self.act_func(lstm_out).shape
        self.apply(_init_weights)


    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.lstm.weight.size())}")


    def forward(self, x):
        if isinstance(x, dict):
            if 'past_calendar' in x.keys() and 'future_calendar' in x.keys():
                future_calendar = torch.cat([x['past_calendar'], x['future_calendar']], 1)
            elif 'past_calendar' in x.keys():
                future_calendar = x['past_calendar']
            elif 'future_calendar' in x.keys():
                future_calendar = x['future_calendar']
            else:
                future_calendar = None


            if 'past_data' in x.keys() and 'future_data' in x.keys():
                future_data = torch.cat([x['past_data'], x['future_data']], 1)
            elif 'past_data' in x.keys():
                future_data = torch.rand(x['past_data'])
            elif 'future_data' in x.keys():
                future_data = x['future_data']
            else:
                future_data = None




            past_obs = x['past_obs'] if 'past_obs' in x.keys() else None


            past_obs_emd = self.obs_embed(past_obs) if self.obs_embed is not None else None


            device = future_data.device if future_data is not None else future_calendar.device
            B, Tpred, Ctime = future_calendar.shape if future_calendar is not None else future_data.shape
            past =[mat for mat in [past_obs, past_obs_emd] if mat is not None]
            future = [mat for mat in [future_data, future_calendar] if mat is not None]
            x_enc1 = torch.cat(past, -1)
            x_enc2 = torch.cat(future, -1)
            hidden_state = (torch.rand(1, B, int(self.size * self.enc_in)).to(device),
                            torch.rand(1, B, int(self.size * self.enc_in)).to(device))
            lstm_out1, hidden_state = self.lstm[0](x_enc1, hidden_state)
            lstm_out1 = self.act_func(lstm_out1)
            lstm_out2, hidden_state = self.lstm[1](x_enc2, hidden_state)
            lstm_out2 = self.act_func(lstm_out2)
            hidden_state = (torch.rand(int(self.size), B, int(self.size * self.enc_in)).to(device),
                            torch.rand(int(self.size), B, int(self.size * self.enc_in)).to(device))
            lstm_out, hidden_state = self.lstm[2](torch.cat([lstm_out1, lstm_out2], 1), hidden_state)
            return self.act_func(lstm_out)[:, -Tpred:, :]
        else:
            B = x.shape[0]
            inp = x
            device = x.device
            hidden_state = (torch.rand(1, B, int(self.size * self.enc_in)).to(device),
                            torch.rand(1, B, int(self.size * self.enc_in)).to(device))
            lstm_out = self.lstm(inp, hidden_state)[0]
            if self.is_lstm_output:
                return self.act_func(lstm_out)[:, -1, :]
            else:
                return self.act_func(lstm_out)


class lstm_att(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(lstm_att, self).__init__()
        self.shape = shape
        self.size = size
        self.is_lstm_output = params['is_lstm_output']
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_lstm_{layer_id}'
        if isinstance(shape, dict):
            try:
                B, Tpred, Ctime = shape['future_data' if 'future_data' in shape.keys() else 'future_calendar']
            except:
                B, Tpred, Ctime = shape['past_data' if 'past_data' in shape.keys() else 'past_calendar']
            self.obs_embed = None
            if 'past_calendar' in shape.keys() and 'future_calendar' in shape.keys():
                future_calendar = torch.cat([torch.rand(shape['past_calendar']),
                                             torch.rand(shape['future_calendar'])], 1)
            elif 'past_calendar' in shape.keys():
                future_calendar = torch.rand(shape['past_calendar'])
            elif 'future_calendar' in shape.keys():
                future_calendar = torch.rand(shape['future_calendar'])
            else:
                future_calendar = None

            if 'past_data' in shape.keys() and 'future_data' in shape.keys():
                future_data = torch.cat([torch.rand(shape['past_data']),
                                    torch.rand(shape['future_data'])], 1)
            elif 'past_data' in shape.keys():
                future_data = torch.rand(shape['past_data'])
            elif 'future_data' in shape.keys():
                future_data = torch.rand(shape['future_data'])
            else:
                future_data = None

            if 'past_obs' in shape.keys():
                past_obs = torch.rand(shape['past_obs'])
            else:
                past_obs = None

            past_obs_emd = self.obs_embed(past_obs) if self.obs_embed is not None else None
            past =[mat for mat in [past_obs, past_obs_emd] if mat is not None]
            future = [mat for mat in [future_data, future_calendar] if mat is not None]
            x_enc1 = torch.cat(past, -1)
            x_enc2 = torch.cat(future, -1)
            self.enc_in = int(size * max([x_enc1.shape[-1], x_enc2.shape[-1]]))
            self.lstm = nn.ModuleList([nn.LSTM(x_enc1.shape[-1], self.enc_in, batch_first=True),
                         nn.LSTM(x_enc2.shape[-1], self.enc_in, batch_first=True)])

            self.cross_attn = CrossAttention({'lstm1': [10, int(x_enc1.shape[1] * self.enc_in)],
                                              'lstm2': [10, int(x_enc2.shape[1] * self.enc_in)]}, None, None, name_scope,
                                             layer_id, train=train, heads=5, dim_head=self.enc_in * self.enc_in, dropout=0.1)
            hidden_state = (torch.rand(1, B, self.enc_in), torch.rand(1, B, self.enc_in))
            lstm_out1, hidden_state = self.lstm[0](x_enc1, hidden_state)

            hidden_state = (torch.rand(1, B, self.enc_in), torch.rand(1, B, self.enc_in))
            lstm_out2, hidden_state1 = self.lstm[1](x_enc2, hidden_state)

            cross_attn_out = self.cross_attn([self.act_func(lstm_out1.flatten(start_dim=1)),
                                              self.act_func(lstm_out2.flatten(start_dim=1))])
            self.output_shape = cross_attn_out.shape

        else:
            B = shape[0]
            Tpred = shape[1]
            x_dec = torch.rand(shape)
            self.enc_in = shape[-1]
            self.lstm = nn.LSTM(self.enc_in, int(size * self.enc_in), batch_first=True)
            hidden_state = (torch.rand(1, B, int(size * self.enc_in)), torch.rand(1, B, int(size * self.enc_in)))
            lstm_out = self.lstm(x_dec, hidden_state)[0]
            if self.is_lstm_output:
                self.output_shape = self.act_func(lstm_out)[:, -Tpred:, :].shape
            else:
                self.output_shape = self.act_func(lstm_out).shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.lstm.weight.size())}")

    def forward(self, x):
        if isinstance(x, dict):
            if 'past_calendar' in x.keys() and 'future_calendar' in x.keys():
                future_calendar = torch.cat([x['past_calendar'], x['future_calendar']], 1)
            elif 'past_calendar' in x.keys():
                future_calendar = x['past_calendar']
            elif 'future_calendar' in x.keys():
                future_calendar = x['future_calendar']
            else:
                future_calendar = None

            if 'past_data' in x.keys() and 'future_data' in x.keys():
                future_data = torch.cat([x['past_data'], x['future_data']], 1)
            elif 'past_data' in x.keys():
                future_data = torch.rand(x['past_data'])
            elif 'future_data' in x.keys():
                future_data = x['future_data']
            else:
                future_data = None


            past_obs = x['past_obs'] if 'past_obs' in x.keys() else None

            past_obs_emd = self.obs_embed(past_obs) if self.obs_embed is not None else None

            device = future_data.device if future_data is not None else future_calendar.device
            B, Tpred, Ctime = future_calendar.shape if future_calendar is not None else future_data.shape
            past =[mat for mat in [past_obs, past_obs_emd] if mat is not None]
            future = [mat for mat in [future_data, future_calendar] if mat is not None]
            x_enc1 = torch.cat(past, -1)
            x_enc2 = torch.cat(future, -1)
            hidden_state = (torch.rand(1, B, self.enc_in).to(device),
                            torch.rand(1, B, self.enc_in).to(device))
            lstm_out1, hidden_state = self.lstm[0](x_enc1, hidden_state)

            hidden_state = (torch.rand(1, B, self.enc_in).to(device),
                            torch.rand(1, B, self.enc_in).to(device))
            lstm_out2, hidden_state1 = self.lstm[1](x_enc2, hidden_state)

            return self.cross_attn([self.act_func(lstm_out1.flatten(start_dim=1)),
                                              self.act_func(lstm_out2.flatten(start_dim=1))])
        else:
            B = x.shape[0]
            inp = x
            device = x.device
            hidden_state = (torch.rand(1, B, int(self.size * self.enc_in)).to(device),
                            torch.rand(1, B, int(self.size * self.enc_in)).to(device))
            lstm_out = self.lstm(inp, hidden_state)[0]
            if self.is_lstm_output:
                return self.act_func(lstm_out)[:, -1, :]
            else:
                return self.act_func(lstm_out)


class RnnEmbedding(nn.Module):
    def __init__(self, shape, embed_size, layer_id, use_embedding):
        super().__init__()
        self.embed_size = embed_size
        self.use_embedding = use_embedding
        self.name_embed = f'RnnEmbedding{layer_id}'
        if self.use_embedding:
            self.rnn = nn.LSTM(shape[-1], embed_size, bias= False, batch_first=True)
            output, (hn, cn) = self.rnn(torch.rand(shape))
            output = rearrange(output, 'b t c -> b c t')
            self.output_shape = output[:, :, -shape[-1]:].shape
        else:
            self.output_shape = torch.zeros([shape[0], embed_size, shape[-1]]).shape
        self.apply(_init_weights)


    def forward(self, x):
        shape = x.shape
        if self.use_embedding:
            output, (hn, cn) = self.rnn(x)
            output = rearrange(output, 'b t c -> b c t')
            return output[:, :, -shape[-1]:]
        else:
            return torch.zeros([shape[0], self.embed_size, shape[-1]])

class transformer(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(transformer, self).__init__()
        B, Tpast, Tpred, Cdata, Ctime, Cobs = 0, 0, 0, 0, 0, 0
        if 'past_data' in shape.keys() and 'future_data' in shape.keys():
            B, Tpred, Cdata = shape['future_data']
            _, Tpast, _ = shape['past_data']
            future_data = torch.cat([torch.rand(shape['past_data']),
                                     torch.rand(shape['future_data'])], 1)
        elif 'past_data' in shape.keys():
            B, Tpast, Cdata = shape['past_data']
            future_data = torch.rand(shape['past_data'])
        elif 'future_data' in shape.keys():
            B, Tpred, Cdata = shape['future_data']
            future_data = torch.rand(shape['future_data'])
        else:
            future_data = None

        if 'past_calendar' in shape.keys() and 'future_calendar' in shape.keys():
            B, Tpred, Ctime = shape['future_calendar']
            _, Tpast, _ = shape['past_calendar']
            future_calendar = torch.cat([torch.rand(shape['past_calendar']),
                                         torch.rand(shape['future_calendar'])], 1)
        elif 'past_calendar' in shape.keys():
            B, Tpast, Ctime = shape['past_calendar']
            future_calendar = torch.rand(shape['past_calendar'])
        elif 'future_calendar' in shape.keys():
            B, Tpred, Ctime = shape['future_calendar']
            future_calendar = torch.rand(shape['future_calendar'])
        else:
            future_calendar = None

        if 'past_obs' in shape.keys():
            past_obs = torch.rand(shape['past_obs'])
        else:
            past_obs = None

        if 'past_obs' in shape.keys():
            B, Tpast, Cobs = shape['past_obs']
            self.obs_embed = RnnEmbedding(shape['past_obs'], Tpred, layer_id, params['use_embedding'])
        else:
            self.obs_embed = None

        self.shape = shape
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_transformer_{layer_id}'
        self.transformer_name = params['transformer_name']
        enc_in = Cdata + Cobs
        dec_in = Cdata + Cobs

        self.size = size if self.transformer_name not in {'Autoformer'} else enc_in
        self.transfromer_model = get_transfromer_model(self.transformer_name, Tpast, Tpred, enc_in,
                                                       dec_in, int(size), params)

        past_calendar = torch.rand(shape['past_calendar']) if 'past_calendar' in shape.keys() else None
        past_data = torch.rand(shape['past_data']) if 'past_data' in shape.keys() else None
        past_obs_emd = self.obs_embed(past_obs) if self.obs_embed is not None else None
        past = [mat for mat in [past_obs, past_obs_emd] if mat is not None]
        future_obs = torch.cat(past, 1) if len(past) > 0 else None

        x_enc = torch.cat([mat for mat in [past_obs, past_data] if mat is not None], -1)
        x_mark_enc = past_calendar
        x_dec = torch.cat([mat for mat in [future_obs, future_data] if mat is not None], -1)
        x_mark_dec = future_calendar
        self.output_shape = self.transfromer_model(x_enc, x_mark_enc, x_dec, x_mark_dec).shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.lstm.weight.size())}")

    def forward(self, x):
        if 'past_data' in x.keys() and 'future_data' in x.keys():
            future_data = torch.cat([x['past_data'], x['future_data']], 1)
        elif 'past_data' in x.keys():
            future_data = x['past_data']
        elif 'future_data' in x.keys():
            future_data = x['future_data']
        else:
            future_data = None

        if 'past_calendar' in x.keys() and 'future_calendar' in x.keys():
            future_calendar = torch.cat([x['past_calendar'], x['future_calendar']], 1)
        elif 'past_calendar' in x.keys():
            future_calendar = x['past_calendar']
        elif 'future_calendar' in x.keys():
            future_calendar = x['future_calendar']
        else:
            future_calendar = None
        past_obs = x['past_obs'] if 'past_obs' in x.keys() else None
        past_obs_emd = self.obs_embed(past_obs) if self.obs_embed is not None else None

        past_calendar = x['past_calendar'] if 'past_calendar' in x.keys() else None
        past_data = x['past_data'] if 'past_data' in x.keys() else None
        past = [mat for mat in [past_obs, past_obs_emd] if mat is not None]
        future_obs = torch.cat(past, 1) if len(past) > 0 else None
        x_enc = torch.cat([mat for mat in [past_obs, past_data] if mat is not None], -1)
        x_mark_enc = past_calendar
        x_dec = torch.cat([mat for mat in [future_obs, future_data] if mat is not None], -1)
        x_mark_dec = future_calendar

        return self.act_func(self.transfromer_model(x_enc, x_mark_enc, x_dec, x_mark_dec))

class bi_lstm(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(bi_lstm, self).__init__()
        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_lstm_{layer_id}'
        if isinstance(shape, dict):
            B, Tpred, Ctime = shape['future_data' if 'future_data' in shape.keys() else 'future_calendar']
            try:
                self.obs_embed = RnnEmbedding(shape['past_obs'], Tpred, layer_id, params['use_embedding']) \
                    if 'past_obs' in shape.keys() else None
            except:
                self.obs_embed = None
            if 'past_calendar' in shape.keys():
                past_calendar = torch.rand(shape['past_calendar'])
            else:
                past_calendar = None
            if 'future_calendar' in shape.keys():
                future_calendar = torch.rand(shape['future_calendar'])
            else:
                future_calendar = None

            if 'past_data' in shape.keys():
                past_data = torch.rand(shape['past_data'])
            else:
                past_data = None

            if 'future_data' in shape.keys():
                future_data = torch.rand(shape['future_data'])
            else:
                future_data = None

            if 'past_obs' in shape.keys():
                past_obs = torch.rand(shape['past_obs'])
            else:
                past_obs = None

            future_obs = self.obs_embed(past_obs) if self.obs_embed is not None else None
            past = [mat for mat in [past_obs, past_data, past_calendar] if mat is not None]
            future = [mat for mat in [future_obs, future_data, future_calendar] if mat is not None]
            x_dec1 = torch.cat(past, -1)
            x_dec2 = torch.cat(future, -1)
            self.enc_in1 = x_dec1.shape[-1]
            self.enc_in2 = x_dec2.shape[-1]
            self.T1 = x_dec1.shape[1]
            self.T2 = x_dec2.shape[1]
        else:
            raise NotImplementedError(f"bi_lstm layer not implemented for input tensor. Only dict is supported")
        self.lstm1 = nn.LSTM(self.enc_in1, int(size * self.enc_in1), batch_first=True)
        self.lstm2 = nn.LSTM(self.enc_in2, int(size * self.enc_in2), batch_first=True)
        hidden_state1 = (torch.rand(1, B, int(size * self.enc_in1)), torch.rand(1, B, int(size * self.enc_in1)))
        hidden_state2 = (torch.rand(1, B, int(size * self.enc_in2)), torch.rand(1, B, int(size * self.enc_in2)))
        lstm_out1 = self.lstm1(x_dec1, hidden_state1)[0]
        lstm_out2 = self.lstm2(x_dec2, hidden_state2)[0]
        lstm_out = torch.cat([lstm_out1[:, -int(self.T1 / 2):, :], lstm_out2[:, -int(self.T2 / 2):, :]], 1)
        self.output_shape = self.act_func(lstm_out).shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.lstm.weight.size())}")

    def forward(self, x):
        if isinstance(x, dict):

            if 'past_calendar' in x.keys():
                past_calendar = x['past_calendar']
            else:
                past_calendar = None
            if 'future_calendar' in x.keys():
                future_calendar = x['future_calendar']
            else:
                future_calendar = None

            if 'past_data' in x.keys():
                past_data = x['past_data']
            else:
                past_data = None
            if 'future_data' in x.keys():
                future_data = x['future_data']
            else:
                future_data = None


            past_obs = x['past_obs'] if 'past_obs' in x.keys() else None

            future_obs = self.obs_embed(past_obs) if self.obs_embed is not None else None
            try:
                device = future_data.device if future_data is not None else future_calendar.device
                B, Tpred, Ctime = future_calendar.shape if future_calendar is not None else future_data.shape
            except:
                device = past_data.device if past_data is not None else past_calendar.device
                B, Tpred, Ctime = past_calendar.shape if past_calendar is not None else past_data.shape

            past = [mat for mat in [past_obs, past_data, past_calendar] if mat is not None]
            future = [mat for mat in [future_obs, future_data, future_calendar] if mat is not None]
            inp1 = torch.cat(past, -1)
            inp2 = torch.cat(future, -1)
        else:
            raise NotImplementedError(f"bi_lstm layer not implemented for input tensor. Only dict is supported")
        hidden_state1 = (torch.rand(1, B, int(self.size * self.enc_in1)).to(device),
                         torch.rand(1, B, int(self.size * self.enc_in1)).to(device))
        hidden_state2 = (torch.rand(1, B, int(self.size * self.enc_in2)).to(device),
                         torch.rand(1, B, int(self.size * self.enc_in2)).to(device))
        lstm_out1 = self.lstm1(inp1, hidden_state1)[0]
        lstm_out2 = self.lstm2(inp2, hidden_state2)[0]
        lstm_out = torch.cat([lstm_out1[:, -int(self.T1 / 2):, :], lstm_out2[:, -int(self.T2 / 2):, :]], 1)
        return self.act_func(lstm_out)

class Attention(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(Attention, self).__init__()
        self.shape = shape
        self.size = size
        self.name = f'{name_scope}_lstm_{layer_id}'
        if isinstance(shape, dict):
            B, Tpred, Ctime = shape['future_data' if 'future_data' in shape.keys() else 'future_calendar']
            try:
                self.obs_embed = RnnEmbedding(shape['past_obs'], Tpred, layer_id, params['use_embedding']) \
                    if 'past_obs' in shape.keys() else None
            except:
                self.obs_embed = None
            if 'past_calendar' in shape.keys() and 'future_calendar' in shape.keys():
                future_calendar = torch.cat([torch.rand(shape['past_calendar']),
                                             torch.rand(shape['future_calendar'])], 1)
            elif 'past_calendar' in shape.keys():
                future_calendar = torch.rand(shape['past_calendar'])
            elif 'future_calendar' in shape.keys():
                future_calendar = torch.rand(shape['future_calendar'])
            else:
                future_calendar = None

            if 'past_data' in shape.keys() and 'future_data' in shape.keys():
                future_data = torch.cat([torch.rand(shape['past_data']),
                                    torch.rand(shape['future_data'])], 1)
            elif 'past_data' in shape.keys():
                future_data = torch.rand(shape['past_data'])
            elif 'future_data' in shape.keys():
                future_data = torch.rand(shape['future_data'])
            else:
                future_data = None

            if 'past_obs' in shape.keys():
                past_obs = torch.rand(shape['past_obs'])
            else:
                past_obs = None

            past_obs_emd = self.obs_embed(past_obs) if self.obs_embed is not None else None
            past =[mat for mat in [past_obs, past_obs_emd] if mat is not None]
            future_obs = torch.cat(past, 1) if len(past) > 0 else None
            future = [mat for mat in [future_obs, future_data, future_calendar] if mat is not None]
            x_dec = torch.cat(future, -1)
            self.enc_in = x_dec.shape[1]
        else:
            B = shape[0]
            x_dec = torch.rand(shape)
            self.enc_in = shape[1]
        self.dense = nn.Linear(self.enc_in, int(self.enc_in * size))
        self.attn = MultiheadAttention(int(self.enc_in * size), int(size))
        lin_out = self.dense(rearrange(x_dec, 'b t c -> b c t'))
        attn_out, _ = self.attn(lin_out, lin_out, lin_out)
        attn_out = rearrange(attn_out, 'b c t -> b t c')
        self.output_shape = attn_out.shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.lstm.weight.size())}")

    def forward(self, x):
        if isinstance(x, dict):
            if 'past_calendar' in x.keys() and 'future_calendar' in x.keys():
                future_calendar = torch.cat([x['past_calendar'], x['future_calendar']], 1)
            elif 'past_calendar' in x.keys():
                future_calendar = x['past_calendar']
            elif 'future_calendar' in x.keys():
                future_calendar = x['future_calendar']
            else:
                future_calendar = None

            if 'past_data' in x.keys() and 'future_data' in x.keys():
                future_data = torch.cat([x['past_data'], x['future_data']], 1)
            elif 'past_data' in x.keys():
                future_data = torch.rand(x['past_data'])
            elif 'future_data' in x.keys():
                future_data = x['future_data']
            else:
                future_data = None


            past_obs = x['past_obs'] if 'past_obs' in x.keys() else None

            past_obs_emd = self.obs_embed(past_obs) if self.obs_embed is not None else None

            past =[mat for mat in [past_obs, past_obs_emd] if mat is not None]
            future_obs = torch.cat(past, 1) if len(past) > 0 else None
            future = [mat for mat in [future_obs, future_data, future_calendar] if mat is not None]
            inp = torch.cat(future, -1)
        else:
            inp = x
        lin_out = self.dense(rearrange(inp, 'b t c -> b c t'))
        attn_out, _ = self.attn(lin_out, lin_out, lin_out)
        attn_out = lin_out + attn_out if (attn_out.shape[-1] / inp.shape[1]) > 5 else attn_out
        return rearrange(attn_out, 'b c t -> b t c')


class CrossAttention(nn.Module):
    def __init__(
        self, shape, params, size, name_scope, layer_id, train=True,
        heads=4,
        dim_head=128,
        dropout=0.0,
    ):
        super(CrossAttention, self).__init__()
        if not isinstance(shape, dict):
            raise ValueError('shape must be dict of names of branches that should be cross-attention')
        if len(shape) > 2:
            tags = [k for k in shape.keys()]
            shape1 = {n:v for n, v in shape.items() if n != tags[0]}
            self.concat_layer = Concat_(shape1, layer_id)
            shape = {tags[0]: shape[tags[0]], '_'.join(tags[1:]): self.concat_layer .output_shape}
        names_branch = [k for k in shape.keys()]
        if len(shape) != 2:
            self.output_shape = shape[names_branch[0]]
            return
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        names_branch = [k for k in shape.keys()]
        self.to_q = nn.Linear(shape[names_branch[1]][1], inner_dim, bias=False)

        self.to_kv = nn.Linear(shape[names_branch[0]][1], inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, shape[names_branch[0]][1]), nn.Dropout(dropout))
        self.apply(_init_weights)
        self.output_shape = self.forward([torch.rand(10, shape[names_branch[0]][1]),
                                          torch.rand(10, shape[names_branch[1]][1])]).shape

    def forward(self, x_list):
        if not isinstance(x_list, list):
            return x_list
        if len(x_list) == 1:
            return x_list[0]
        if len(x_list) > 2:
            x1 = self.concat_layer(x_list[1:])
            x = [x_list[0], x1]
        else:
            x = x_list
        src, tgt = x
        q = self.to_q(tgt)

        qkv = (q, *self.to_kv(src).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, "b (h d) -> (b h) d", h=self.heads), qkv
        )

        dots = einsum(q, k, "b i, b i -> b i") * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum(attn, v, "b j, b j -> b j")
        out = rearrange(out, "(b h) d -> b (h d)", h=self.heads)
        return self.to_out(out)


class CrossAttentionBoth(nn.Module):
    def __init__(
        self, shape, params, size, name_scope, layer_id, train=True,
        heads=4,
        dim_head=128,
        dropout=0.0,
    ):
        super(CrossAttentionBoth, self).__init__()
        if not isinstance(shape, dict):
            raise ValueError('shape must be dict of names of branches that should be cross-attention')
        if len(shape) > 2:
            tags = [k for k in shape.keys()]
            shape1 = {n:v for n, v in shape.items() if n != tags[0]}
            self.concat_layer = Concat_(shape1, layer_id)
            shape = {tags[0]: shape[tags[0]], '_'.join(tags[1:]): self.concat_layer .output_shape}
        names_branch = [k for k in shape.keys()]
        if len(shape) != 2:
            self.output_shape = shape[names_branch[0]]
            return
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q_f = nn.Linear(shape[names_branch[1]][1], inner_dim, bias=False)
        self.to_q_b = nn.Linear(shape[names_branch[0]][1], inner_dim, bias=False)

        self.to_kv_f = nn.Linear(shape[names_branch[0]][1], inner_dim * 2, bias=False)
        self.to_kv_b = nn.Linear(shape[names_branch[1]][1], inner_dim * 2, bias=False)

        self.to_out_f = nn.Sequential(nn.Linear(inner_dim, shape[names_branch[0]][1]), nn.Dropout(dropout))
        self.to_out_b = nn.Sequential(nn.Linear(inner_dim, shape[names_branch[1]][1]), nn.Dropout(dropout))
        self.apply(_init_weights)
        self.output_shape = self.forward([torch.rand(10, shape[names_branch[0]][1]),
                                          torch.rand(10, shape[names_branch[1]][1])]).shape

    def forward(self, x_list):
        if not isinstance(x_list, list):
            return x_list
        if len(x_list) == 1:
            return x_list[0]
        if len(x_list) > 2:
            x1 = self.concat_layer(x_list[1:])
            x = [x_list[0], x1]
        else:
            x = x_list
        src_f, tgt_f = x
        q_f = self.to_q_f(tgt_f)

        qkv_f = (q_f, *self.to_kv_f(src_f).chunk(2, dim=-1))

        q_f, k_f, v_f = map(
            lambda t: rearrange(t, "b (h d) -> (b h) d", h=self.heads), qkv_f
        )

        dots_f = einsum(q_f, k_f, "b i, b i -> b i") * self.scale

        attn_f = self.attend(dots_f)
        attn_f = self.dropout(attn_f)

        out_f = einsum(attn_f, v_f, "b j, b j -> b j")
        out_f = rearrange(out_f, "(b h) d -> b (h d)", h=self.heads)
        out_f = self.to_out_f(out_f)

        src_b, tgt_b = tgt_f, src_f
        q_b = self.to_q_b(tgt_b)

        qkv_b = (q_b, *self.to_kv_b(src_b).chunk(2, dim=-1))

        q_b, k_b, v_b = map(
            lambda t: rearrange(t, "b (h d) -> (b h) d", h=self.heads), qkv_b
        )

        dots_b = einsum(q_b, k_b, "b i, b i -> b i") * self.scale

        attn_b = self.attend(dots_b)
        attn_b = self.dropout(attn_b)

        out_b = einsum(attn_b, v_b, "b j, b j -> b j")
        out_b = rearrange(out_b, "(b h) d -> b (h d)", h=self.heads)
        out_b = self.to_out_b(out_b)
        return torch.cat([out_f, out_b], -1)


# # class yolo(nn.Module):
# #     def __init__(self, shape, params, size, name_scope, layer_id, train=True):
# #         super(yolo, self).__init__()
# #
# #         self.shape = shape
# #         self.size = size
# #         self.act_func = act_funcs[params['act_func']]
# #
# #         self.resize = transforms.Resize((224, 224), antialias=False)
# #         if params['pretrained'] in {'tune', 'finetune'}:
# #             is_train = True
# #         else:
# #             is_train = False
# #
# #         model_trained = YOLO(model=os.path.basename(params['model_yaml']).replace('yolov8', 'yolov8m').
# #                              replace('.yaml', '.pt'))
# #         self.yolo_model = DetectionModel(cfg=params['model_yaml'], ch=self.shape[-1])
# #         state_dict = dict()
# #         for k in self.yolo_model.state_dict().keys():
# #             for k2 in model_trained.state_dict().keys():
# #                 if k in k2:
# #                     state_dict[k] = model_trained.state_dict()[k2]
# #                     break
# #         if is_train:
# #             try:
# #                 self.yolo_model.load_state_dict(state_dict)
# #             except:
# #                 pass
# #         if params['pretrained'] == 'finetune':
# #             self.yolo_model.requires_grad = False
# #         self.flat = nn.Flatten()
#
#
#     def string(self):
#         print(f"layer {self.name} has input shape {self.shape}")
#
#     def forward(self, x):
#         x = rearrange(x, 'b h w c -> b c h w')
#         if self.resize is not None:
#             x = self.resize(x)
#         return self.act_func(self.flat(self.yolo_model(x)))


class unet(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(unet, self).__init__()

        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_{params["unet_type"]}_{layer_id}'
        arch = eval(f'smp.{params["unet_type"]}')

        self.resize = transforms.Resize((128, 128), antialias=False)
        if params['pretrained'] in {'tune', 'finetune'}:
            is_train = True
        else:
            is_train = False
        self.unet_model = arch(encoder_name=params["encoder_name"],
                               in_channels=shape[1],
                               encoder_depth = 3,
                               decoder_channels= (64, 32, 16),
                               encoder_weights='imagenet' if is_train else None,
                               classes=1)
        if params['pretrained'] == 'finetune':
            self.unet_model.requires_grad = False
        self.flat = nn.Flatten()


    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")

    def forward(self, x):
        if self.resize is not None:
            x = self.resize(x)
        return self.act_func(self.flat(self.unet_model(x)))

class vit_net(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(vit_net, self).__init__()

        self.shape = shape
        if len(self.shape) == 5:
            B, T, C, H, W = self.shape
            C = int(T * C)
        elif len(self.shape) == 4:
            B, C, H, W = self.shape
        else:
            raise ValueError(f'Wrong input shape given dimension {len(self.shape)} expected 4 or 5')
        x_sample = torch.rand([B, C, H, W])
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_{params["model_name"]}_{layer_id}'
        self.model_name = params["model_name"]
        try:
            s = int(params["model_name"].split('_')[-1])
        except:
            s =  H
        if s > 600:
            raise ValueError('Too Large input size')
        self.img_size = s
        if self.model_name in params['timm_models'] or 'timm' in self.model_name:
            try:
                self.vit_model = timm.create_model(self.model_name, pretrained=True,
                                                    num_classes=0, global_pool='', in_chans=C)
            except:
                try:
                    self.vit_model = timm.create_model(self.model_name, pretrained=True,
                                                        num_classes=self.img_size * C, in_chans=C)
                except:
                    self.vit_model = timm.create_model(self.model_name, pretrained=False,
                                                       num_classes=self.img_size * C, in_chans=C)
            data_config = timm.data.resolve_model_data_config(self.vit_model)
            self.resize = transforms.Resize(data_config['input_size'][1:], antialias=False)

            img = self.resize(x_sample)
            outputs = self.vit_model(img)
            if len(outputs.shape) == 2:
                outputs = rearrange(outputs, 'b (c h) -> b c h', c=self.img_size)
            output_shape = outputs.shape
        else:
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.vit_model = AutoModel.from_pretrained(self.model_name)
                inputs = self.processor(images=x_sample, return_tensors="pt", input_data_format='channels_first',
                                        do_rescale=False)
                outputs = self.vit_model(**inputs)
            except:
                self.config = AutoConfig.from_pretrained(self.model_name)
                self.config.num_channels = C
                if hasattr(self.config, 'image_size'):
                    self.config.image_size = H if isinstance(self.config.image_size, int) else [H, W]
                self.vit_model = AutoModel.from_config(self.config)
                outputs = self.vit_model(x_sample)
            output_shape = outputs.last_hidden_state.shape
            outputs = outputs.last_hidden_state
        self.flat = nn.Flatten()
        self.pool = nn.AvgPool1d(2)
        self.output_shape = self.pool(self.flat(outputs)).shape
        if params['pretrained'] == 'finetune':
            self.vit_model.requires_grad = False


    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer {self.name} has output_shape {self.output_shape}")

    def forward(self, x):
        device = x.device
        if isinstance(x, dict):
            x = x['images']
        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
            C = int(T * C)
        elif len(x.shape) == 4:
            B, C, H, W = x.shape
        if len(x.size()) == 5:
            x = rearrange(x, 'b t c h w -> b (t c) h w')
        if hasattr(self, 'resize'):
            x = self.resize(x)
            output = self.vit_model(x)
            if len(output.shape) == 2:
                output = rearrange(output, 'b (c h) -> b c h',c=self.img_size)
            output = self.act_func(self.pool(self.flat(output)))
            if bool(torch.isnan(output).any().cpu().detach().numpy()):
                try:
                    self.vit_model = timm.create_model(self.model_name, pretrained=True,
                                                       num_classes=0, global_pool='', in_chans=C)
                except:
                    try:
                        self.vit_model = timm.create_model(self.model_name, pretrained=True,
                                                           num_classes=self.img_size * C, in_chans=C)
                    except:
                        self.vit_model = timm.create_model(self.model_name, pretrained=False,
                                                       num_classes=self.img_size * C, in_chans=C)
                self.vit_model.to(device)
                output = self.vit_model(x)
                if len(output.shape) == 2:
                    output = rearrange(output, 'b (c h) -> b c h', c=self.img_size)
                output = self.act_func(self.pool(self.flat(output)))
        elif hasattr(self, 'config'):
            output = self.act_func(self.pool(self.flat(self.vit_model(x).last_hidden_state)))
            if bool(torch.isnan(output).any().cpu().detach().numpy()):
                self.vit_model = AutoModel.from_config(self.config)
                self.vit_model.to(device)
                output = self.act_func(self.pool(self.flat(self.vit_model(x).last_hidden_state)))
        elif hasattr(self, 'processor') and not hasattr(self, 'config'):
            x = self.processor(images=x, return_tensors="pt", input_data_format='channels_first',
                                        do_rescale=False)
            x = x.to(device)
            output = self.act_func(self.pool(self.flat(self.vit_model(**x).last_hidden_state)))
            if bool(torch.isnan(output).any().cpu().detach().numpy()):
                self.vit_model = AutoModel.from_pretrained(self.model_name)
                self.vit_model.to(device)
                output = self.act_func(self.pool(self.flat(self.vit_model(**x).last_hidden_state)))
        return output

class time_distr_vit_net(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(time_distr_vit_net, self).__init__()
        self.shape = shape
        B, T, C, H, W = self.shape
        self.size = size
        self.act_func = act_funcs['sigmoid']
        self.name = f'{name_scope}_time_distr_vit_net_{layer_id}'
        self.vit_list = nn.ModuleList()
        for i in range(T):
            self.vit_list.append(vit_net([B, C, H, W], params, size, self.name, i))
        self.lstm = nn.LSTM(int(self.vit_list[0].output_shape[-1]), int(self.vit_list[0].output_shape[-1]),
                            batch_first=True)
        self.flat = nn.Flatten()
        x = torch.rand([10, T, C, H, W])
        time_steps = x.size()[1]
        output = torch.tensor([])
        for i in range(time_steps):
            output_t = self.vit_list[i](x[:, i, ...])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        hidden_state = (torch.rand(1, B, output.shape[-1]), torch.rand(1, B, output.shape[-1]))
        lstm_out = self.lstm(output, hidden_state)[0]
        self.output_shape = self.act_func(self.flat(lstm_out)).shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(
            f"layer weights {self.name} has {self.shape[1]} conv of shape{list(self.conv_list[0].weight.size())}")
        print(f"layer {self.name} has output_shape {self.output_shape}")

    def forward(self, x):
        if isinstance(x, dict):
            x = x['images']
        time_steps = x.size()[1]
        B = x.size()[0]
        output = torch.tensor([]).to(x.device)
        for i in range(time_steps):
            output_t = self.vit_list[i](x[:, i, ...])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        hidden_state = (torch.rand(1, B, output.shape[-1]).to(x.device),
                        torch.rand(1, B, output.shape[-1]).to(x.device))
        lstm_out = self.lstm(output, hidden_state)[0]
        return self.act_func(self.flat(lstm_out))

class dense(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(dense, self).__init__()
        self.shape = shape
        self.size = size
        if params['act_func'] is not None:
            self.act_func = act_funcs[params['act_func']]
        else:
            self.act_func = None
        self.name = f'{name_scope}_dense_{layer_id}'
        self.dense = nn.Linear(shape[1], int(size))
        self.output_shape = self.dense(torch.rand([10, *self.shape[1:]])).shape
        self.apply(_init_weights)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.dense.weight.size())}")

    def forward(self, x):
        if self.act_func is not None:
            return self.act_func(self.dense(x))
        else:
            return self.dense(x)


class Reshape(nn.Module):
    def __init__(self, shape, layer_id):
        super(Reshape, self).__init__()
        self.shape = shape
        self.name = f'Reshape_{layer_id}'
        self.output_shape = [10] + self.shape

    def forward(self, x):
        if (self.shape[0] * self.shape[1]) != x.shape[1]:
            x = x[:, :int(self.shape[0] * self.shape[1])]
        return rearrange(x, 'b (c h) -> b c h', c=self.shape[0], h=self.shape[1])


class Concat_(nn.Module):
    def __init__(self, shape, layer_id):
        super(Concat_, self).__init__()
        self.shape = shape
        shapes = [s for s in shape.values()]
        self.output_shape = [shapes[0][0], sum([s[-1] for s in shape.values()])]
        self.name = f'Concat_{layer_id}'

    def forward(self, x):
        if isinstance(x, list):
            return torch.cat(x, -1)
        else:
            return x
class Clip(nn.Module):
    def __init__(self, shape, layer_id=0):
        super(Clip, self).__init__()
        self.shape = shape
        self.name = f'Clip_{layer_id}'
        # Learnable log-temperature initialised to ln(1/0.07) ≈ 2.66 (CLIP default)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07)))
        self.output_shape = shape

    def forward(self, x):
        if isinstance(x, list):
            # L2-normalise embeddings → cosine similarity
            x_norm = [F.normalize(xi, dim=-1) for xi in x]
            # Exponentiate + clamp for numerical stability
            logit_scale = self.logit_scale.exp().clamp(max=100.0)
            out = []
            for i in range(len(x_norm)):
                for j in range(len(x_norm)):
                    if i >= j:
                        continue
                    similarity = logit_scale * x_norm[i] @ x_norm[j].T
                    out.append(similarity)
            return out
        else:
            return x

def layers_func():
    layers = {'conv_2d': conv_2d,
              'time_distr_conv_2d': time_distr_conv_2d,
              'conv_3d': conv_3d,
              'time_distr_conv_3d': time_distr_conv_3d,
              'lstm': lstm,
              'lstm_att': lstm_att,
              'cnn_att': cnn_attention,
              'bi_lstm': bi_lstm,
              'transformer': transformer,
              'attention': Attention,
              'Reshape': Reshape,
              'unet': unet,
              # 'yolo': yolo,
              'timm_net': vit_net,
              'cross_attention': CrossAttention,
              'cross_attention_both': CrossAttentionBoth,
              'clip': Clip,
              'concatenate': Concat_,
              'vit_net': vit_net,
              'time_distr_vit_net': time_distr_vit_net,
              'dense': dense
              }
    return layers
