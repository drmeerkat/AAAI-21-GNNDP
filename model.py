import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Bernoulli, Categorical, DiagGaussian
from utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs=None, masks=None, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 64, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(64, 510, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(510, 64, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(64 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class GNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(GNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 64, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(64, 510, 4, stride=2)), nn.ReLU(),)
            # init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            # init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
        self.gnn1 = MultiHeadAttention(512, 256, 256, 4)
        self.gnn2 = MultiHeadAttention(256, 128, 128, 4)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        # self.mapping_feature = init_(nn.Linear(256 * 9 * 9, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        # add object position as additional GNN feature
        b_sz, n_channel, height, width = x.size()
        entity_embed = x.reshape(b_sz, n_channel, -1).transpose(1, 2)
        coord = []
        for i in range(height*width):
            # add coordinate and normalize
            coord.append([float(i//width)/height, (i%width)/width])
        coord = torch.tensor(coord, device=entity_embed.device).view(1, -1, 2).repeat(b_sz, 1, 1)
        entity_embed = torch.cat((entity_embed, coord), dim=2)
        
        out = F.relu(self.gnn1(entity_embed))
        # out = F.relu(self.gnn2(out))
        x = torch.max(out, dim=1)[0]
        # x = F.relu(self.mapping_feature(out.view(b_sz, -1)))

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MultiHeadAttention(nn.Module):
    '''Multi-Head Attention module used by GConv'''
    
    def __init__(self, d_model, d_kq, d_v, n_heads, drop_prob=0.0):
        super(MultiHeadAttention, self).__init__()
        
        init_ = lambda m: self.param_init(m, nn.init.orthogonal_, nn.init.calculate_gain('relu'))

        self.d_model = d_model
        self.d_k = d_kq
        self.d_q = d_kq
        self.d_v = d_v
        self.n_heads = n_heads
        self.linear_k = init_(nn.Linear(self.d_model, self.d_k*n_heads, bias=False))
        self.linear_q = init_(nn.Linear(self.d_model, self.d_q*n_heads, bias=False))
        self.linear_v = init_(nn.Linear(self.d_model, self.d_v*n_heads, bias=False))
        self.normalize = np.sqrt(self.d_k)
        self.linear_output = nn.Sequential(
#                                 nn.Linear(self.d_v*n_heads, self.d_model*2, bias=False),
#                                 nn.LeakyReLU(),
#                                 nn.Linear(self.d_model*2, self.d_model, bias=False)
                                  nn.Linear(self.d_v*n_heads, self.d_model, bias=False),
                             )
        
        # Assume that the dimension of linear_k/q/v are all the same
        self.layer_norm_embed = nn.LayerNorm(self.d_k*n_heads, eps=1e-6)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.atten_dropout = nn.Dropout(drop_prob)
        
    def param_init(self, module, weigth_init, gain=1):
        weigth_init(module.weight.data, gain=gain)

        return module
    
    def forward(self, entity_embeds_raw):
        b_sz, num_entities = entity_embeds_raw.size(0), entity_embeds_raw.size(1)
        # (batch_size, num_entities, d_model) -> (batch_size*num_entities, d_model)
        entity_embeds = entity_embeds_raw.reshape(-1, self.d_model)
        # (batch_size*num_entities, d_k*n_heads) -> (batch_size, num_entities, n_heads, d_k)
        embed_q = self.layer_norm_embed(self.linear_q(entity_embeds)).view(b_sz, num_entities, self.n_heads, self.d_q)
        embed_k = self.layer_norm_embed(self.linear_k(entity_embeds)).view(b_sz, num_entities, self.n_heads, self.d_k)
        embed_v = self.layer_norm_embed(self.linear_v(entity_embeds)).view(b_sz, num_entities, self.n_heads, self.d_v)

        residual_v = embed_v
        # swap n_head dim with num_entities
        # ->(batch_size, n_heads, num_entities, d_k)
        embed_q2 = embed_q.transpose(1,2)
        embed_k2 = embed_k.transpose(1,2)
        embed_v2 = embed_v.transpose(1,2)
        
        # Scaled Dot Product Attention(for each head)
        tmp = torch.matmul(embed_q2, embed_k2.transpose(2, 3))/self.normalize
        # -> (batch_size, n_heads, num_entities, num_entities_prob)
        weights = self.atten_dropout(F.softmax(tmp, dim=-1))
        # weights = self.atten_dropout(F.softmax(tmp, dim=1)) #this is the previous old/wrong implementation
        new_v = torch.matmul(weights, embed_v2)
        
        # Concatenate over head dimensioins
        # (batch_size, n_heads, num_entities, d_k) -> (batch_size, num_entities, n_heads*d_k)
        new_v = new_v.transpose(1, 2).contiguous().view(b_sz, num_entities, -1)
        new_v = self.linear_output(new_v)
        
        # residual
        output = new_v + entity_embeds_raw
        # output = new_v + residual_v.view(b_sz, num_entities, -1)
        # output = self.layer_norm_embed(output).view(b_sz, num_entities, new_v.shape[-1])
        output = self.layer_norm(output).view(b_sz, num_entities, self.d_model)
        
        return output


class ImpalaCNN(NNBase):
    def __init__(self, num_inputs, hidden_size=256, recurrent=False, depths=[16, 32, 32]):
        super(ImpalaCNN, self).__init__(recurrent, num_inputs, hidden_size)

        init_cnn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        init_fc = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.num_inputs = num_inputs
        self.depths = depths

        self.blocks = []
        tmp_in = num_inputs
        for depth in depths:
            self.blocks.append(self.conv_block(tmp_in, depth, init_cnn))
            tmp_in = depth
        self.blocks = nn.Sequential(*self.blocks)
        # The output shape from blocks would be 32x11x11
        self.flatten_fc = init_fc(nn.Linear(11 * 11 * depths[-1], hidden_size))
        self.critic_linear = init_fc(nn.Linear(hidden_size, 1))
        self.relu = nn.ReLU()

        self.train()

    def conv_block(self, num_inputs, num_outputs, init_cnn):
        return nn.Sequential(
                    init_cnn(nn.Conv2d(num_inputs, num_outputs, 3, padding=1)),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    ResidualConv(num_outputs),
                    ResidualConv(num_outputs),
               )

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs / 255.0
        # for block in self.blocks:
        x = self.blocks(x)
        # print(x.shape)
        out = x.view(-1, x.shape[2]*x.shape[3]*x.shape[1])
        # print("x.shape out.shape", x.shape, out.shape)
        out = self.flatten_fc(self.relu(out))
        
        # if self.is_recurrent:
        #     x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(out), out, rnn_hxs
        

class ResidualConv(nn.Module):

    def __init__(self, channel):
        super(ResidualConv, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(channel, channel, 3, padding=1))
        self.conv2 = init_(nn.Conv2d(channel, channel, 3, padding=1))
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.relu(inputs)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return inputs + out


class ImpalaGNN(NNBase):
    def __init__(self, num_inputs, hidden_size=256, recurrent=False, depths=[16, 32, 32]):
        super(ImpalaGNN, self).__init__(recurrent, num_inputs, hidden_size)

        init_cnn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        init_fc = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.num_inputs = num_inputs
        self.depths = depths

        self.blocks = []
        tmp_in = num_inputs
        for i, depth in enumerate(depths):
            if i != len(depths)-1:
                self.blocks.append(self.conv_block(tmp_in, depth, init_cnn))
            else:
                self.blocks.append(self.mha_block(tmp_in, depth, init_cnn))
                
            tmp_in = depth
        self.blocks = nn.Sequential(*self.blocks)
        # The output shape from blocks would be 32x11x11
        self.flatten_fc = init_fc(nn.Linear(11 * 11 * depths[-1], hidden_size))
        self.critic_linear = init_fc(nn.Linear(hidden_size, 1))
        self.relu = nn.ReLU()

        self.train()

    def mha_block(self, num_inputs, num_outputs, init_cnn):
        return nn.Sequential(
                    init_cnn(nn.Conv2d(num_inputs, num_outputs-2, 3, padding=1)),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    ResidualMHA(num_outputs-2, loc_pad=True),
                    ResidualMHA(num_outputs, loc_pad=False),
               )

    def conv_block(self, num_inputs, num_outputs, init_cnn):
        return nn.Sequential(
                    init_cnn(nn.Conv2d(num_inputs, num_outputs, 3, padding=1)),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    ResidualConv(num_outputs),
                    ResidualConv(num_outputs),
               )

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs / 255.0
        # print('input shape:', x.shape)
        x = self.blocks(x).contiguous()
        # out = self.relu(torch.max(x, dim=1)[0])
        # actually the max version is faster and more stable
        out = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        # print("x.shape out.shape", x.shape, out.shape)
        out = self.relu(out)
        out = self.flatten_fc(out)
        out = self.relu(out)
        return self.critic_linear(out), out, rnn_hxs

class ResidualMHA(nn.Module):

    def __init__(self, channel, loc_pad=False):
        super(ResidualMHA, self).__init__()
        if loc_pad:
            self.channel = channel+2
        else:
            self.channel = channel

        self.gnn1 = MultiHeadAttention(self.channel, self.channel*8, self.channel*8, 4)
        self.gnn2 = MultiHeadAttention(self.channel, self.channel*8, self.channel*8, 4)
        self.relu = nn.ReLU()
        self.loc_pad = loc_pad

    def forward(self, inputs):
        x = self.relu(inputs)
        # print("block input shape: ", x.shape)
        # add object position as additional GNN feature
        b_sz, n_channel, height, width = x.size()
        entity_embed_raw = x.reshape(b_sz, n_channel, -1).transpose(1, 2)
        if self.loc_pad:
            coord = []
            for i in range(height*width):
                # add coordinate and normalize
                coord.append([float(i//width)/height, (i%width)/width])
            coord = torch.tensor(coord, device=entity_embed_raw.device).view(1, -1, 2).repeat(b_sz, 1, 1)
            entity_embed_raw = torch.cat((entity_embed_raw, coord), dim=2)

        
        # print('entity embed shape', entity_embed_raw.shape)
        entity_embed = F.relu(self.gnn1(entity_embed_raw))
        entity_embed = F.relu(self.gnn2(entity_embed))

        out = (entity_embed+entity_embed_raw).transpose(1, 2).view(b_sz, self.channel, height, width)
        # print("output.shape", out.shape)
        return out

