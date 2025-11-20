import torch
import torch.nn as nn
from torch.nn import init
import functools
import torchvision.models as models

###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='batch'):
    """Return a normalization layer"""
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights."""

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'init method [{init_type}] not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f'initialize network with {init_type}')
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize network weights"""
    init_weights(net, init_type, init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch',
             use_dropout=False, init_type='normal', init_gain=0.02):
    """Create generator"""
    norm_layer = get_norm_layer(norm)

    if netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError(f'Generator {netG} not recognized')

    return init_net(net, init_type, init_gain)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch',
             init_type='normal', init_gain=0.02):
    """Create discriminator"""
    norm_layer = get_norm_layer(norm)

    if netD == 'basic':  # 70×70 PatchGAN
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError(f'Discriminator {netD} not recognized')

    return init_net(net, init_type, init_gain)


###############################################################################
# Learning Rate Scheduler
###############################################################################

def get_scheduler(optimizer, opt):
    """Return learning rate scheduler"""
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(
                opt.n_epochs_decay + 1
            )
            return lr_l

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )

    elif opt.lr_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5
        )

    elif opt.lr_policy == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epochs, eta_min=0
        )
    else:
        raise NotImplementedError(f'lr_policy [{opt.lr_policy}] is not implemented')

    return scheduler


###############################################################################
# GAN Loss
###############################################################################

class GANLoss(nn.Module):
    """Define GAN objective"""

    def __init__(self, gan_mode='vanilla'):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode [{gan_mode}] not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        return (self.real_label if target_is_real else self.fake_label).expand_as(prediction)

    def forward(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            return -prediction.mean() if target_is_real else prediction.mean()


###############################################################################
# U-Net Generator
###############################################################################

class UnetSkipConnectionBlock(nn.Module):
    """Defines a Unet submodule"""

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost

        use_bias = (norm_layer == nn.InstanceNorm2d)
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else nn.Identity()
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc) if norm_layer is not None else nn.Identity()

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x) if self.outermost else torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """Full U-Net generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        # innermost
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer
        )
        # intermediate 8×8
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, submodule=unet_block,
                norm_layer=norm_layer, use_dropout=use_dropout
            )
        # down 4×
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer
        )
        # outermost
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block,
            outermost=True, norm_layer=norm_layer
        )

    def forward(self, input):
        return self.model(input)


###############################################################################
# PatchGAN Discriminator
###############################################################################

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        use_bias = (norm_layer == nn.InstanceNorm2d
                    or isinstance(norm_layer, functools.partial))

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kw, 2, padw),
                    nn.LeakyReLU(0.2, True)]

        nf_mult, nf_mult_prev = 1, 1

        # downsampling
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kw, 2, padw, bias=use_bias),
                norm_layer(ndf * nf_mult)
                if norm_layer is not None else nn.Identity(),
                nn.LeakyReLU(0.2, True)
            ]

        # final conv layers
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kw, 1, padw, bias=use_bias),
            norm_layer(ndf * nf_mult) if norm_layer else nn.Identity(),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kw, 1, padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


###############################################################################
# VGG19 Perceptual Feature Extractor
###############################################################################

class VGG19FeatureExtractor(nn.Module):
    """Feature extractor for perceptual loss"""

    def __init__(self, layer_index=16):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg[:layer_index])).eval()

        for p in self.features.parameters():
            p.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        # convert from [0,1] to normalized ImageNet
        x = (x - self.mean) / self.std
        return self.features(x)
