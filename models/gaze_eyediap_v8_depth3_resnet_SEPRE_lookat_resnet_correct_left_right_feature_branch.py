#from torchvision.models.resnet import BasicBlock, ResNet, model_urls
from torchvision.models.resnet import  model_urls
import torch.utils.model_zoo as model_zoo
from torch import nn
import torch as th
import torch.nn.functional as F





# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64


        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 =  nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.maxpool(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.relu(out)
        return out


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model =   ResNet(BasicBlock, [3,4,6,3])
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


class Decoder(nn.Module):
    def __init__(self, feat_dim=512):
        super(Decoder, self).__init__()
        self.ldecoder = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(True),
        )
        self.rdecoder = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(True),
        )
        '''
        v1
        self.lcoord = nn.Sequential(
            nn.Linear(128 + 128 , 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )
        self.rcoord = nn.Sequential(
            nn.Linear(128 + 128 , 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )

        '''

        '''
        v8 change
        
        self.lcoord = nn.Sequential(
            nn.Linear(128 + 128 , 64),
            nn.ReLU(True),
        )
        self.rcoord = nn.Sequential(
            nn.Linear(128 + 128 , 64),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential( nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
            )
        '''
        
        self.lcoord = nn.Sequential(
            nn.Linear(128  , 64),
            nn.ReLU(True),
        )
        self.rcoord = nn.Sequential(
            nn.Linear(128  , 64),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential( nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
            )

        self.lcoord_seperate_branch = nn.Sequential(
            nn.Linear(128  , 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )

        self.rcoord_seperate_branch = nn.Sequential(
            nn.Linear(128  , 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )

    def forward(self, lfeat, rfeat): #, head_pose):
        l_coord_feat = self.ldecoder(lfeat)
        r_coord_feat = self.rdecoder(rfeat)
        # l_coord = self.lcoord(th.cat([l_coord_feat, head_pose], 1))
        # r_coord = self.rcoord(th.cat([r_coord_feat, head_pose], 1))
        l_coord = self.lcoord(l_coord_feat)
        r_coord = self.rcoord(r_coord_feat)
        
        seperate_left_gaze = self.lcoord_seperate_branch(l_coord_feat) 
        seperate_right_gaze = self.rcoord_seperate_branch(r_coord_feat) 

        coord = self.fc(  th.cat([l_coord, r_coord], 1) )
        
        '''
        v1origianl coord = (l_coord + r_coord) / 2.
        '''
        return seperate_left_gaze, seperate_right_gaze, coord


class DepthL1(nn.Module):
    def __init__(self, th_lower=None, th_upper=None):
        super(DepthL1, self).__init__()
        self.th_lower = th_lower
        self.th_upper = th_upper

    def forward(self, pred, target):
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        if self.th_lower is not None:
            with th.no_grad():
                mask_lower = (target > self.th_lower).float()
        else:
            mask_lower = 1.
        if self.th_upper is not None:
            with th.no_grad():
                mask_upper = (target < self.th_upper).float()
        else:
            mask_upper = 1.

        return th.sum(th.abs(pred - target) * mask_lower * mask_upper) / (th.sum(mask_lower * mask_upper) + 1e-5)


class DepthBCE(nn.Module):
    def __init__(self, th_lower=None, th_upper=None):
        super(DepthBCE, self).__init__()
        self.th_lower = th_lower
        self.th_upper = th_upper

    def forward(self, pred, target):
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        if self.th_lower is not None:
            with th.no_grad():
                mask_lower = (target > self.th_lower).float()
        else:
            mask_lower = 1.
        if self.th_upper is not None:
            with th.no_grad():
                mask_upper = (target < self.th_upper).float()
        else:
            mask_upper = 1.

        weight = mask_upper * mask_lower
        return F.binary_cross_entropy(pred, target, weight=weight if not isinstance(weight, float) else None)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return F.relu(out)


class RefineDepth(nn.Module):
    def __init__(self):
        super(RefineDepth, self).__init__()
        use_bias = False

        self.face_block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.face_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.face_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.face_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.depth_block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.depth_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.depth_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.depth_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            ResnetBlock(512, padding_type='reflect', norm_layer=nn.BatchNorm2d,
                        use_dropout=False, use_bias=use_bias),
            # ResnetBlock(512, padding_type='reflect', norm_layer=nn.BatchNorm2d,
            #             use_dropout=False, use_bias=use_bias),
            # ResnetBlock(512, padding_type='reflect', norm_layer=nn.BatchNorm2d,
            #             use_dropout=False, use_bias=use_bias)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # ResnetBlock(256, padding_type='reflect', norm_layer=nn.BatchNorm2d,
            #             use_dropout=False, use_bias=use_bias),
            # ResnetBlock(256, padding_type='reflect', norm_layer=nn.BatchNorm2d,
            #             use_dropout=False, use_bias=use_bias)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # ResnetBlock(128, padding_type='reflect', norm_layer=nn.BatchNorm2d,
            #             use_dropout=False, use_bias=use_bias)
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # ResnetBlock(64, padding_type='reflect', norm_layer=nn.BatchNorm2d,
            #             use_dropout=False, use_bias=use_bias)
        )

        self.head_pose = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.AvgPool2d(7),
            nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(128),
            # nn.ReLU(True)
        )

        self.gen_block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.gen_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.gen_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        # self.gen_block4 = nn.Sequential(
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(64, 1, kernel_size=7, padding=0),
        #     nn.Sigmoid()
        # )
        self.gen_block4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Sigmoid()
        )

    def forward(self, face, depth):
        # import pudb
        # pudb.set_trace()
        face_f1 = self.face_block1(face)
        face_f2 = self.face_block2(face_f1)
        face_f3 = self.face_block3(face_f2)
        face_f4 = self.face_block4(face_f3)
        depth_f1 = self.depth_block1(depth)
        depth_f2 = self.depth_block2(depth_f1)
        depth_f3 = self.depth_block3(depth_f2)
        depth_f4 = self.depth_block4(depth_f3)
        mixed_f4 = self.down1(th.cat([face_f4, depth_f4], dim=1))
        mixed_f3 = self.down2(th.cat([face_f3, depth_f3], dim=1))
        mixed_f2 = self.down3(th.cat([face_f2, depth_f2], dim=1))
        mixed_f1 = self.down4(th.cat([face_f1, depth_f1], dim=1))
        gen_f3 = self.gen_block1(mixed_f4) + mixed_f3#zff: fpn
        gen_f2 = self.gen_block2(gen_f3) + mixed_f2
        gen_f1 = self.gen_block3(gen_f2) + mixed_f1
        gen_depth = self.gen_block4(gen_f1)
        head_pose = self.head_pose(mixed_f4)
        return head_pose.view(head_pose.size(0), -1), gen_depth
