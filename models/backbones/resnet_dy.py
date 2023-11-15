import os
import math
import torch
import torchvision
from torch import nn
from torch.utils import model_zoo
from core.layers import NonLocal, IBN
#----------------------------------EEA
import torch.nn.functional as F
import cv2
import numpy as np
extractFeatures = False
#----------------------------------EEA

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/semi_weakly_supervised_resnext101_32x8-b4712904.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
#-------------------------------------------------------EEA
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

class conv_dy(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(conv_dy, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dim = int(math.sqrt(inplanes))   #orjinali bu
        #self.dim = int(math.sqrt(inplanes))*2
        squeeze = max(inplanes, (self.dim) ** 2) // 16

        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        ####------------------------------------------EEA
        self.inplanes_p = inplanes
        self.planes_p = planes
        ####------------------------------------------EEA


        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()

        ####------------------------------------------EEA
        self.counter_1 = 0
        ####------------------------------------------EEA

    def forward(self, x):
        r = self.conv(x) #r girise gore degisiyor
        b, c, _, _= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        phi = self.fc_phi(y).view(b, self.dim, self.dim)
        scale = self.hs(self.fc_scale(y)).view(b,-1,1,1)
        r = scale.expand_as(r)*r

        out = self.bn1(self.q(x))
        _, _, h, w = out.size()

        out = out.view(b,self.dim,-1)
        out = self.bn2(torch.matmul(phi, out)) + out
        out = out.view(b,-1,h,w)
        out = self.p(out) + r

        #print('DIM: {}'.format(self.dim))
        #print('INPLANES: {}'.format(self.inplanes_p))
        #print('PLANES: {}'.format(self.planes_p))

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False,
                 stride=1, downsample=None, groups=1, base_width=64,
                 use_non_local=False, use_last_relu=True
                 ):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv_dy(inplanes, planes, 1, 1, 0)
        #self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)

        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv_dy(width, width, 3, stride, 1) #ilk layerda 1, diğerlerinde 2
        #self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
        #                       padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv_dy(width, planes * self.expansion, 1, 1, 0)
        #self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_last_relu = use_last_relu
        # GCNet

        self.use_non_local = use_non_local
        if self.use_non_local:
            self.non_local_block = NonLocal(planes * self.expansion,
                                            planes * self.expansion // 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_last_relu:
            out = self.relu(out)

        if self.use_non_local:
            out = self.non_local_block(out)

        return out

#-------------------------------------------------------EEA
""" #statik
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False,
                 stride=1, downsample=None, groups=1, base_width=64,
                 use_non_local=False, use_last_relu=True
                 ):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)

        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_last_relu = use_last_relu
        # GCNet

        self.use_non_local = use_non_local
        if self.use_non_local:
            self.non_local_block = NonLocal(planes * self.expansion,
                                            planes * self.expansion // 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_last_relu:
            out = self.relu(out)

        if self.use_non_local:
            out = self.non_local_block(out)

        return out

"""
class ResNet(nn.Module):
    def __init__(self, last_stride,
                 block, layers, model_name, use_non_local=False,
                 groups=1, width_per_group=64, use_last_relu=True):

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        self._model_name = model_name
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_non_local=use_non_local)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_non_local=use_non_local)
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=last_stride,
                                       use_non_local=use_non_local, use_last_relu=use_last_relu)

    def _make_layer(self, block, planes, blocks, stride=1,
                    use_non_local=False, use_last_relu=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride=stride, downsample=downsample,
                            groups=self.groups, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            use_non_local_flag = use_non_local and i == blocks - 2
            use_last_relu_flag = False if (not use_last_relu and i == blocks -1) else True
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                use_non_local=use_non_local_flag,
                                use_last_relu=use_last_relu_flag
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print('INPUT SIZE: {}'.format(x.size()))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x1 = x#    ------EEA feat map
        #print('X1 SIZE: {}'.format(x1.size())) #torch.Size([64, 256, 64, 32])
        x = self.layer2(x)
        #x2 = x#   ------EEA feat map
        #print('X2 SIZE: {}'.format(x2.size())) #torch.Size([64, 512, 32, 16])
        x = self.layer3(x)
        #x3 = x#   ------EEA feat map
        #print('X3 SIZE: {}'.format(x3.size())) #torch.Size([64, 1024, 16, 8])
        x = self.layer4(x)
        #x4 = x#   ------EEA feat map
        #print('X4 SIZE: {}'.format(x4.size())) #torch.Size([64, 2048, 16, 8])
        
        #--------------------------------------------------------------------------------------------EEA
        """
        if extractFeatures == True:
            if not os.path.exists("output/feat_maps_dyn"):
            	 os.makedirs("output/feat_maps_dyn")
            for a in range(x4.size(0)): #7 her sample icin loop
              total_feat_map = np.zeros((x4.size(2), x4.size(3))) 
              total_feat_map1 = np.zeros((x1.size(2), x1.size(3)))
              total_feat_map2 = np.zeros((x2.size(2), x2.size(3)))
              total_feat_map3 = np.zeros((x3.size(2), x3.size(3))) 
              total_feat_map4 = np.zeros((x4.size(2), x4.size(3)))  
              for featmap_channel_1 in range(x1.size(1)):
                feat_map_layer1 = x1[a][featmap_channel_1].cpu().detach().numpy()
                total_feat_map1 = np.add(total_feat_map1, feat_map_layer1)
              average_feat_map1 = total_feat_map1 / x1.size(1)
              img1 = cv2.convertScaleAbs(average_feat_map1, alpha=(255.0))
              cv2.imwrite('output/feat_maps_dyn/average_features_dynreid_layer1_{}.png'.format(a), img1)

              for featmap_channel_2 in range(x2.size(1)):
                feat_map_layer2 = x2[a][featmap_channel_2].cpu().detach().numpy()
                total_feat_map2 = np.add(total_feat_map2, feat_map_layer2)
              average_feat_map2 = total_feat_map2 / x2.size(1)
              img2 = cv2.convertScaleAbs(average_feat_map2, alpha=(255.0))
              cv2.imwrite('output/feat_maps_dyn/average_features_dynreid_layer2_{}.png'.format(a), img2)

              for featmap_channel_3 in range(x3.size(1)): #her channel icin loop
                  feat_map_layer3 = x3[a][featmap_channel_3].cpu().detach().numpy()
                  total_feat_map3 = np.add(total_feat_map3, feat_map_layer3)
              average_feat_map3 = total_feat_map3 / x3.size(1)
              img3 = cv2.convertScaleAbs(average_feat_map3, alpha=(255.0))
              cv2.imwrite('output/feat_maps_dyn/average_features_dynreid_layer3_{}.png'.format(a), img3)

              for featmap_channel_4 in range(x4.size(1)): #2048 her channel icin loop
                  feat_map_layer4 = x4[a][featmap_channel_4].cpu().detach().numpy()
                  total_feat_map4 = np.add(total_feat_map4, feat_map_layer4)
              average_feat_map4 = total_feat_map4 / x4.size(1)
              img4 = cv2.convertScaleAbs(average_feat_map4, alpha=(255.0))
              cv2.imwrite('output/feat_maps_dyn/average_features_dynreid_layer4_{}.png'.format(a), img4)

            #self.counter = self.counter + 1;
            cv2.waitKey(300) # The waitKey waits until the value written inside of it. Its property is in a millisecond.
            """

        ##---------------------------------------------------------EEA

        return x



    def load_pretrain(self, model_path=''):
        with_model_path = (model_path is not '')
#-------------model pathi vermediysen burada sıkıntı cıkacak pretrainedte
        if not with_model_path:  # resnet pretrain model path yoksa giriyor buraya
            print('Download from', model_urls[self._model_name])
            if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK']:
                print('map weight to cuda: %s' % str(os.environ['LOCAL_RANK']))
                state_dict = model_zoo.load_url(model_urls[self._model_name],
                                                map_location="cuda:" + str(os.environ['LOCAL_RANK']))
            else:
                pretrained_dir = '/content/drive/MyDrive/PersonReID-YouReID/pretrained_models/resnet50_dcd.pth.tar' #--------EEA
                state_dict = torch.load(pretrained_dir)
                #state_dict = model_zoo.load_url(model_urls[self._model_name]) #statik resnet için
                #print(state_dict.keys())
            #state_dict.pop('fc.weight')
            #state_dict.pop('fc.bias')
            self.load_state_dict(state_dict, strict=False)

        else:  #-----------------------------------model pathi verdiysen:
            # ibn pretrain
            print('load from ', model_path)
            state_dict = torch.load(model_path)['state_dict']

            #state_dict.pop('module.fc.weight')
            #state_dict.pop('module.fc.bias')
            #-------------------------------------EEA

            state_dict_wo_cls = dict()

            for key in state_dict.keys(): #classifier layerı attım
                if key.startswith("module.classifier")==False:
                    state_dict_wo_cls[key] = state_dict[key]

            #print(state_dict_wo_cls.keys())

            new_state_dict = {}
            for k in state_dict_wo_cls:
                new_k = '.'.join(k.split('.')[1:])  # remove module in name
                if self.state_dict()[new_k].shape == state_dict_wo_cls[k].shape:
                    new_state_dict[new_k] = state_dict_wo_cls[k]    #weight matrislerinin boyutu tutuyorsa eşliyor
                    
            state_dict = new_state_dict
            #print(state_dict.keys())
            self.load_state_dict(state_dict, strict=False)


#----------------------------------------------------------------EEA

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def _resnet(pretrained, last_stride, block,
            layers, model_name, model_path='', **kwargs):
    """"""
    model = ResNet(last_stride,
                   block, layers, model_name, **kwargs)
    if pretrained:
        #model.load_pretrain(model_path)
        model.load_pretrain('/content/drive/MyDrive/PersonReID-YouReID/pretrained_models/resnet50_dcd.pth.tar')

    return model


def resnet50(pretrained=True, last_stride=1, use_non_local=False, model_path='', **kwargs): #------------EEA
#def resnet50(pretrained=False, last_stride=1, use_non_local=False, model_path='', **kwargs): #orjinali
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(pretrained=pretrained, last_stride=last_stride,
                   block=Bottleneck, layers=[3, 4, 6, 3], use_non_local=use_non_local,
                   model_path=model_path, model_name='resnet50', **kwargs)


def resnet101(pretrained=False, last_stride=1, use_non_local=False, model_path=''):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _resnet(pretrained=pretrained, last_stride=last_stride, use_non_local=use_non_local,
                   block=Bottleneck, layers=[3, 4, 23, 3], model_path=model_path, model_name='resnet101')


def resnet152(pretrained=False, last_stride=1, model_path=''):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _resnet(pretrained=pretrained, last_stride=last_stride,
                   block=Bottleneck, layers=[3, 8, 36, 3], model_path=model_path, model_name='resnet152')


def resnext101_32x8d(pretrained=False, last_stride=1, model_path='', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(pretrained=pretrained, last_stride=last_stride,
                   block=Bottleneck, layers=[3, 4, 23, 3],
                   model_path=model_path, model_name='resnext101_32x8d', **kwargs)
