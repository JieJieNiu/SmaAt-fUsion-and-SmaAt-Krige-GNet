from models.unet_parts import Down, DoubleConv, Up, OutConv
from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS, NodeInput3d
from models.layers import CBAM, Interpolate
from models.regression_lightning import Precip_regression_base, node_regression_base, Kriging_regression_base
from torch import nn, conv3d
import torch

class UNet(Precip_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_Attention(Precip_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio

        self.inc = DoubleConv(self.n_channels, 64)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = Down(64, 128)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = Down(128, 256)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = Down(256, 512)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class UNetDS(Precip_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetDS_Attention(Precip_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class UNetDS_Attention_4CBAMs(Precip_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
        

class SmaAt_UNet(Precip_regression_base):
    def __init__(self, hparams):
        super(SmaAt_UNet, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        dropout_prob = self.hparams.dropout

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        
        x = self.up1(x5Att, x4Att)
        x = self.dropout(x)
        x = self.up2(x, x3Att)
        x = self.dropout(x)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class Node_SmaAt(node_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        dropout_prob = self.hparams.dropout

        self.inc = DoubleConvDS(self.n_channels * 2, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.ip = Interpolate(size=(288,288), mode='bilinear') #interpolate data to image size
        self.outc = OutConv(64, self.n_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x , y):
        y1 = self.ip(y)
        x = torch.cat((x, y1), dim= 1)
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        
        x = self.up1(x5Att, x4Att)
        x = self.dropout(x)
        x = self.up2(x, x3Att)
        x = self.dropout(x)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class Node_SmaAt_root(node_regression_base): #version two with Double depth separated convolution before concatenating
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        dropout_prob = self.hparams.dropout

        self.inc = DoubleConvDS(self.n_channels * 2, 64, kernels_per_layer=kernels_per_layer)
        self.inc2 = DoubleConvDS(self.n_channels, 12, mid_channels = 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.ip = Interpolate(size=(288,288), mode='bilinear') #interpolate data to image size
        self.outc = OutConv(64, self.n_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x , y):
        y1 = self.inc2(y)
        y2 = self.ip(y1)
        x = torch.cat((x, y2), dim= 1)
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        
        x = self.up1(x5Att, x4Att)
        x = self.dropout(x)
        x = self.up2(x, x3Att)
        x = self.dropout(x)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class Node_SmaAt_bridge(node_regression_base): #version with embedded tensor added in the bottleneck
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        dropout_prob = self.hparams.dropout

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.inc2 = NodeInput3d()
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1536, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        
        self.outc = OutConv(64, self.n_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x , y):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        #print(x5Att.shape)
        #bridge part
        y1 = self.inc2(y)
        #print(y1.shape)
        #print(x5Att.shape)
        x = torch.cat((x5Att, y1), dim= 1) #attach the 4x4 node data to the 4x4 embedded tensor along the channels
        #print(x.shape)
        
        x = self.up1(x, x4Att)
        #print(x.shape)
        x = self.dropout(x)
        x = self.up2(x, x3Att)
        x = self.dropout(x)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        #print(logits.shape)
        return logits
 
class Node_GNet(node_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        dropout_prob = self.hparams.dropout
        
        # map down
        self.inc1 = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam11 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down11 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam12 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down12 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam13 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down13 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam14 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down14 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam15 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        
        # mask down
        self.inc2 = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam21 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down21 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam22 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down22 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam23 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down23 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam24 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down24 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam25 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        # up
        self.up1 = UpDS(1024*2, 512*2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512*2, 256*2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256*2, 128*2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128*2, 64*2, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64*2, self.n_classes)
        
        self.dropout = nn.Dropout(p=dropout_prob)
         
    def forward(self, x, y):
        # down map
        x1 = self.inc1(x)
        x1Att = self.cbam11(x1)
        x2 = self.down11(x1)
        x2Att = self.cbam12(x2)
        x3 = self.down12(x2)
        x3Att = self.cbam13(x3)
        x4 = self.down13(x3)
        x4Att = self.cbam14(x4)
        x5 = self.down14(x4)
        x5Att = self.cbam15(x5)
        
        # down nodes
        y1 = self.inc2(y)
        print(y1.shape)
        y1Att = self.cbam21(y1)
        y2 = self.down21(y1)
        y2Att = self.cbam22(y2)
        y3 = self.down22(y2)
        y3Att = self.cbam23(y3)
        y4 = self.down23(y3)
        y4Att = self.cbam24(y4)
        y5 = self.down24(y4)
        y5Att = self.cbam25(y5)
        
        # concatenate
        x5Att = torch.cat((x5Att, y5Att), dim=1)
        x4Att = torch.cat((x4Att, y4Att), dim=1)
        x3Att = torch.cat((x3Att, y3Att), dim=1)
        x2Att = torch.cat((x2Att, y2Att), dim=1)
        x1Att = torch.cat((x1Att, y1Att), dim=1)
        
        # up
        x = self.up1(x5Att, x4Att)
        x = self.dropout(x)
        x = self.up2(x, x3Att)
        x = self.dropout(x)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    
class Krige_GNet(Kriging_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        dropout_prob = self.hparams.dropout
        
        # map down
        self.inc1 = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam11 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down11 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam12 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down12 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam13 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down13 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam14 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down14 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam15 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        
        # mask down
        self.inc2 = DoubleConvDS(8*self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam21 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down21 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam22 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down22 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam23 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down23 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam24 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down24 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam25 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        # up
        self.up1 = UpDS(1024*2, 512*2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512*2, 256*2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256*2, 128*2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128*2, 64*2, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64*2, self.n_classes)
        
        self.dropout = nn.Dropout(p=dropout_prob)
         
    def forward(self, x, y):
        # down map
        x1 = self.inc1(x)
        x1Att = self.cbam11(x1)
        x2 = self.down11(x1)
        x2Att = self.cbam12(x2)
        x3 = self.down12(x2)
        x3Att = self.cbam13(x3)
        x4 = self.down13(x3)
        x4Att = self.cbam14(x4)
        x5 = self.down14(x4)
        x5Att = self.cbam15(x5)
        
        # down nodes
        y = torch.flatten(y, start_dim=1, end_dim=2)
        print(y.shape)
        y1 = self.inc2(y)
        y1Att = self.cbam21(y1)
        y2 = self.down21(y1)
        y2Att = self.cbam22(y2)
        y3 = self.down22(y2)
        y3Att = self.cbam23(y3)
        y4 = self.down23(y3)
        y4Att = self.cbam24(y4)
        y5 = self.down24(y4)
        y5Att = self.cbam25(y5)
        
        # concatenate
        x5Att = torch.cat((x5Att, y5Att), dim=1)
        x4Att = torch.cat((x4Att, y4Att), dim=1)
        x3Att = torch.cat((x3Att, y3Att), dim=1)
        x2Att = torch.cat((x2Att, y2Att), dim=1)
        x1Att = torch.cat((x1Att, y1Att), dim=1)
        
        # up
        x = self.up1(x5Att, x4Att)
        x = self.dropout(x)
        x = self.up2(x, x3Att)
        x = self.dropout(x)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    
class Kriging_SmaAt_root(Kriging_regression_base): #version two with Double depth separated convolution before concatenating
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        dropout_prob = self.hparams.dropout

        self.inc = DoubleConvDS(self.n_channels * 2, 64, kernels_per_layer=kernels_per_layer)
        self.inc2 = DoubleConvDS(self.n_channels, 12, mid_channels = 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.ip = Interpolate(size=(288,288), mode='bilinear') #interpolate data to image size
        self.outc = OutConv(64, self.n_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x , y):
        y1 = self.inc2(y)
        y2 = self.ip(y1)
        x = torch.cat((x, y2), dim= 1)
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        
        x = self.up1(x5Att, x4Att)
        x = self.dropout(x)
        x = self.up2(x, x3Att)
        x = self.dropout(x)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
