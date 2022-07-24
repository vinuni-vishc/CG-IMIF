import torch
import pdb
import torch.nn.functional as F

from torch import nn
from copy import deepcopy

class EdgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.pool = nn.MaxPool2d(8, 8)
        self.conv2 = nn.Conv2d(16, 32, 7)
        # self.conv3 = nn.Conv2d(16, 32, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return x


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        # create different branch feature extraction for different stream

        self.model = model

        # simple net for edge image

        self.model_edge = EdgeNet()

        # head_var_rgb = self.model.head_var
        # head_var_edge = self.model_edge.head_var

        # self.model = model
        # to do: add depth

        # last_layer = getattr(self.model_rgb, head_var)
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
                # setattr(self.model_edge, head_var, nn.Sequential())
                # setattr(self.model, head_var, nn.Sequential())

                # to do: add depth
        else:
            self.out_size = last_layer.out_features

        # create a bunch of module list
        self.heads = nn.ModuleList()
        self.fc_rgb_1_list = nn.ModuleList()
        self.fc_edge_1_list = nn.ModuleList()

        self.task_cls = []
        self.task_offset = []

        # create different fc for different branch before concatenating them
        # self.fc_rgb = nn.Linear(self.out_size, self.out_size)

        # self.fc_rgb_1 = nn.Linear(self.out_size, self.out_size)
        # self.fc_rgb_2 = nn.Linear(512, )
        

        # self.fc_edge = nn.Linear(self.out_size, self.out_size)

        # self.fc_histogram_1 = nn.Linear(512, 1024) # dim of histogram is 512
        # self.fc_histogram_2 = nn.Linear(1024, 2048)


        # modify this later
        # self.out_size_concat = self.out_size + 2048
        # self.out_size_concat = 64 + 64


        self._initialize_weights()


    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        # self.heads.append(nn.Linear(self.out_size, num_outputs))
        new_fc_rgb_1 = nn.Linear(self.out_size, self.out_size)
        new_fc_edge_1 = nn.Linear(288, 64)
        # new_fc_histogram_2 = nn.Linear(1024, 2048)
        
        self.fc_rgb_1_list.append(new_fc_rgb_1)
        self.fc_edge_1_list.append(new_fc_edge_1)
        # size of concatenated feature
        self.out_size_concat = self.out_size + 64
        self.heads.append(nn.Linear(self.out_size_concat, num_outputs))
        
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x_rgb, x_edge, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        
        # Extract feature of image
        x1_rgb = self.model(x_rgb)
        x1_edge = self.model_edge(x_edge)

        assert (len(self.heads) > 0), "Cannot access any head"
        y = []

        # for head in self.heads:
        #     y.append(head(x))

        for index in range(len(self.heads)):
            
            head_i = self.heads[index]
            fc_rgb_1i = self.fc_rgb_1_list[index]
            fc_edge_1i = self.fc_edge_1_list[index]

            feat_rgb = F.relu(fc_rgb_1i(x1_rgb))
            feat_edge= F.relu(fc_edge_1i(x1_edge))
            x_cat = torch.cat((feat_rgb, feat_edge), 1)
            
            y.append(head_i(x_cat))


        if return_features:
            return y, x1_rgb
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass
