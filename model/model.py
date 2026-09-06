import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("C:\\Users\\PycharmProjects\\vmamba")
from VMamba.classification.models.Vmama2 import SRMBlock
# from mamba_ssm import Mamba

class SpectralECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(SpectralECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.alpha_self = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        mask = self.sigmoid(y)
        enhanced_x = x + self.alpha_self * (x * mask)
        return (enhanced_x, mask)


class LiDARGeometricAttention(nn.Module):
    def __init__(self):
        super(LiDARGeometricAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha_self = nn.Parameter(torch.tensor([0.01]))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        mask = self.sigmoid(self.conv(y))
        enhanced_x = x + self.alpha_self * (x * mask)
        return (enhanced_x, mask)


def cal_similarity(
    image: torch.Tensor, text: torch.Tensor, sigma: float = 10.0
) -> torch.Tensor:
    img_sq = (image**2).sum(1, keepdim=True)
    txt_sq = (text**2).sum(1, keepdim=True)
    dist_sq = (img_sq + txt_sq.t() - 2.0 * (image @ text.t())).clamp(min=0.0)
    return torch.exp(-dist_sq / (2.0 * sigma**2))


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(x.dtype)


# Mamba reference implementation
# class Text_Encoder(nn.Module):
#     def __init__(self, embed_dim, context_length, vocab_size, transformer_width):
#         super().__init__()
#         self.context_length = context_length
#         self.transformer_width = transformer_width
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
#         self.mamba = Mamba(d_model=transformer_width, d_state=16, d_conv=4, expand=2)
#         self.ln_final = LayerNorm(transformer_width)
#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         nn.init.normal_(self.positional_embedding, std=0.01)
#         nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)
#
#     def forward(self, text: torch.Tensor) -> torch.Tensor:
#         x = self.token_embedding(text).float() + self.positional_embedding.float()
#         x = self.mamba(x)
#         x = self.ln_final(x)
#         return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
#
#
# class Text_Decoder(nn.Module):
#     def __init__(self, embed_dim=128, context_length=77, vocab_size=49408, transformer_width=64):
#         super().__init__()
#         self.context_length = context_length
#         self.transformer_width = transformer_width
#         self.expand = nn.Linear(embed_dim, context_length * transformer_width)
#         self.mamba = Mamba(d_model=transformer_width, d_state=16, d_conv=4, expand=2)
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         x = self.expand(z).view(z.size(0), self.context_length, self.transformer_width)
#         return self.mamba(x)


class Text_Encoder(nn.Module):
    def __init__(self, embed_dim, context_length, vocab_size, transformer_width):
        super().__init__()
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, transformer_width)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_width,
            nhead=4,
            dim_feedforward=transformer_width * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=transformer_width ** (-0.5))

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text).float() + self.positional_embedding.float()
        x = self.transformer(x)
        x = self.ln_final(x)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection


class Text_Decoder(nn.Module):
    def __init__(
        self, embed_dim=128, context_length=77, vocab_size=49408, transformer_width=64
    ):
        super().__init__()
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.expand = nn.Linear(embed_dim, context_length * transformer_width)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_width,
            nhead=4,
            dim_feedforward=transformer_width * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.expand(z).view(z.size(0), self.context_length, self.transformer_width)
        return self.transformer(x)


class FrequencyDecoupledVSS(nn.Module):
    def __init__(self, dim, d_state=32):
        super().__init__()
        self.dim = dim
        self.vss = VSSBlock(dim=dim, drop_path=0.1, d_state=d_state, mlp_ratio=2.0)
        self.laplacian = nn.Parameter(
            torch.tensor(
                [[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]],
                dtype=torch.float32,
            ).repeat(dim, 1, 1, 1),
            requires_grad=False,
        )
        self.high_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
        )
        self.gate = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        high = F.conv2d(x, self.laplacian, padding=1, groups=self.dim)
        low = x - high
        h_feat = self.high_conv(high)
        l_feat = self.vss(low)
        return h_feat + l_feat * self.gate


class IPFM(nn.Module):
    def __init__(self, C1: int, C2: int, Classes: int):
        super().__init__()
        self.stem1 = nn.Sequential(
            nn.Conv2d(C1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(C2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fuse_w = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.hsi_conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.hsi_eca1 = SpectralECA(64)
        self.lidar_conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.lidar_geo1 = LiDARGeometricAttention()
        self.pool1 = nn.MaxPool2d(2)
        self.hsi_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.hsi_eca2 = SpectralECA(128)
        self.lidar_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.lidar_geo2 = LiDARGeometricAttention()
        self.pool2 = nn.MaxPool2d(2)
        self.fusion_weight = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.alpha_h1 = nn.Parameter(torch.tensor([0.01]))
        self.alpha_l1 = nn.Parameter(torch.tensor([0.01]))
        self.alpha_h2 = nn.Parameter(torch.tensor([0.01]))
        self.alpha_l2 = nn.Parameter(torch.tensor([0.01]))
        self.out_fusion = nn.Linear(128, Classes)
        self.out_fusion_hsi = nn.Linear(128, Classes)
        self.out_fusion_lidar = nn.Linear(128, Classes)
        self.mimf = nn.Linear(32, Classes)
        self.w_cnn = nn.Parameter(torch.ones(1) * 0.6)

    def forward(self, x1, x2):
        B = x1.shape[0]
        x1_stem = self.stem1(x1)
        x2_stem = self.stem2(x2)
        h1 = self.hsi_conv1(x1_stem)
        h1, h_mask1 = self.hsi_eca1(h1)
        l1 = self.lidar_conv1(x2_stem)
        l1, l_mask1 = self.lidar_geo1(l1)
        h1 = h1 + self.alpha_h1 * (h1 * l_mask1)
        l1 = l1 + self.alpha_l1 * (l1 * h_mask1)
        h1 = self.pool1(h1)
        l1 = self.pool1(l1)
        h2 = self.hsi_conv2(h1)
        h2, h_mask2 = self.hsi_eca2(h2)
        l2 = self.lidar_conv2(l1)
        l2, l_mask2 = self.lidar_geo2(l2)
        h2 = h2 + self.alpha_h2 * (h2 * l_mask2)
        l2 = l2 + self.alpha_l2 * (l2 * h_mask2)
        h2 = self.pool2(h2)
        l2 = self.pool2(l2)
        xh_flat = h2.view(B, -1)
        xl_flat = l2.view(B, -1)
        fuse_flat = xh_flat + xl_flat
        Fuse_result = self.out_fusion(fuse_flat)
        HSI_result = self.out_fusion_hsi(xh_flat)
        LiDAR_result = self.out_fusion_lidar(xl_flat)
        return (
            Fuse_result,
            HSI_result,
            LiDAR_result,
            xh_flat,
            xl_flat,
            fuse_flat,
        )


class TAAM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        lambda_re: float,
        lambda_c: float,
    ):
        super().__init__()
        self.lambda_re = lambda_re
        self.lambda_c = lambda_c
        self.img_proj1 = nn.Sequential(
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.img_proj2 = nn.Sequential(
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.img_proj_fused = nn.Sequential(
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.encode_text = Text_Encoder(
            embed_dim, context_length, vocab_size, transformer_width
        )
        self.decode_text1 = Text_Decoder(
            embed_dim, context_length, vocab_size, transformer_width
        )
        self.decode_text2 = Text_Decoder(
            embed_dim, context_length, vocab_size, transformer_width
        )
        self.decode_text_fused = Text_Decoder(
            embed_dim, context_length, vocab_size, transformer_width
        )
        self.decode_text3 = Text_Decoder(
            embed_dim, context_length, vocab_size, transformer_width
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
        self.mse = nn.MSELoss()

    def project_features(self, xh_flat, xl_flat, fuse_flat):
        img_feat1 = self.img_proj1(xh_flat)
        img_feat2 = self.img_proj2(xl_flat)
        img_feat_fused = self.img_proj_fused(fuse_flat)
        return img_feat1, img_feat2, img_feat_fused

    def complementary_mask(self, image_emb, text_emb, mask_ratio=0.8):
        B, D = image_emb.shape
        sim = cal_similarity(image_emb, text_emb)
        k = max(1, math.floor(B * mask_ratio))
        _, topk = torch.topk(torch.diagonal(sim), k=k, largest=True)
        im = torch.zeros(D, dtype=torch.bool, device=image_emb.device)
        im[topk] = True
        tm = ~im
        im = im.unsqueeze(0).expand(B, -1)
        tm = tm.unsqueeze(0).expand(B, -1)
        return (image_emb[:, im[0]], text_emb[:, tm[0]], im, tm)

    @staticmethod
    def _fuse(image_emb, text_emb, image_mask):
        return image_emb * image_mask.float() + text_emb * (1.0 - image_mask.float())

    def _clip_loss(self, fa, fb, scale):
        fa = F.normalize(fa, dim=1)
        fb = F.normalize(fb, dim=1)
        s = scale.exp().clamp(max=100.0)
        lg = s * fa @ fb.t()
        lb = torch.arange(fa.shape[0], device=fa.device)
        return (F.cross_entropy(lg, lb) + F.cross_entropy(lg.t(), lb)) * 0.5

    def complementary_mask_channelwise(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        mask_ratio: float = 0.8,
        sigma: float = 10.0,
    ):
        B, D = image_emb.shape
        image_dim = image_emb.transpose(0, 1)
        text_dim = text_emb.transpose(0, 1)
        image_sq = (image_dim**2).sum(dim=1, keepdim=True)
        text_sq = (text_dim**2).sum(dim=1, keepdim=True).t()
        dist = image_sq + text_sq - 2.0 * (image_dim @ text_dim.t())
        dist = dist.clamp(min=0.0)
        sim = torch.exp(-dist / (2.0 * sigma**2))
        diag_score = torch.diagonal(sim)
        k = max(1, math.floor(D * mask_ratio))
        topk = torch.topk(diag_score, k=k, largest=True).indices
        image_mask = torch.zeros(D, dtype=torch.bool, device=image_emb.device)
        image_mask[topk] = True
        image_mask = image_mask.unsqueeze(0).expand(B, D)
        text_mask = ~image_mask
        fused = image_emb * image_mask.float() + text_emb * text_mask.float()
        return (fused, image_mask, text_mask, sim)

    def _text_losses_one_branch(self, feat, text_feat, token_emb, text_proto, decoder):
        token_emb = F.normalize(token_emb, dim=-1)
        _, _, im, _ = self.complementary_mask(feat, text_feat)
        fused = self._fuse(feat, text_feat, im)
        decoded = F.normalize(decoder(fused), dim=-1)
        re_loss = self.mse(decoded, token_emb)
        clip_loss = self._clip_loss(feat, text_proto, self.logit_scale)
        return (re_loss, clip_loss)

    def compute_losses(
        self,
        img_feat1,
        img_feat2,
        img_feat_fused,
        text,
        text_proto,
        device,
    ):
        re_loss = torch.tensor(0.0, device=device)
        loss_c = torch.tensor(0.0, device=device)
        if self.training and text is not None and (text_proto is not None):
            text_feat = self.encode_text(text)
            token_emb = self.encode_text.token_embedding(text).float()
            re1, c1 = self._text_losses_one_branch(
                img_feat1, text_feat, token_emb, text_proto, self.decode_text1
            )
            re2, c2 = self._text_losses_one_branch(
                img_feat2, text_feat, token_emb, text_proto, self.decode_text2
            )
            re_f, c_f = self._text_losses_one_branch(
                img_feat_fused, text_feat, token_emb, text_proto, self.decode_text_fused
            )
            re_loss = (re1 + re2 + re_f) / 3.0 * self.lambda_re
            loss_c = (c1 + c2 + c_f) / 3.0 * self.lambda_c
        return re_loss, loss_c


class GSGM(nn.Module):
    def __init__(self, C2: int, Classes: int):
        super().__init__()
        global_dim = 32
        self.global_stem1 = nn.Sequential(
            nn.InstanceNorm2d(C2),
            nn.Conv2d(C2, 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(inplace=True),
        )
        self.global_stem2 = nn.Sequential(
            nn.Conv2d(8, 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(inplace=True),
        )
        self.global_stem3 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(inplace=True),
        )
        self.global_stem4 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.vss_global1 = nn.Sequential(
            SRMBlock(dim=8, drop_path=0.1, d_state=32, mlp_ratio=2.0)
        )
        self.vss_global2 = nn.Sequential(
            SRMBlock(dim=16, drop_path=0.1, d_state=32, mlp_ratio=2.0)
        )
        self.vss_global3 = nn.Sequential(
            SRMBlock(dim=32, drop_path=0.1, d_state=32, mlp_ratio=2.0)
        )
        self.vss_global4 = nn.Sequential(
            SRMBlock(dim=64, drop_path=0.1, d_state=32, mlp_ratio=2.0)
        )
        self.alpha_vss1 = nn.Parameter(torch.tensor([0.01]))
        self.alpha_vss2 = nn.Parameter(torch.tensor([0.01]))
        self.alpha_vss3 = nn.Parameter(torch.tensor([0.01]))
        self.ms_proj1 = nn.Conv2d(8, 32, 1)
        self.ms_proj2 = nn.Conv2d(16, 32, 1)
        self.ms_proj3 = nn.Conv2d(32, 32, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.scene_fuse = nn.Linear(32 * 2, 32)
        self.cls = nn.Sequential(
            nn.Linear(global_dim, 16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(16, Classes),
        )

    def forward(self, x3, batch_size):
        x3_map = x3.permute(2, 0, 1).unsqueeze(0)
        x3_feat = self.global_stem1(x3_map)
        x3_feat = self.vss_global1(x3_feat)

        x3_feat = self.global_stem2(x3_feat)
        x3_feat = self.vss_global2(x3_feat)

        x3_feat = self.global_stem3(x3_feat)

        scene = self.global_pool(x3_feat).view(1, -1)
        return F.softmax(self.cls(scene), dim=1).expand(batch_size, -1)


class TAPSCN(nn.Module):
    _MODULE_PREFIXES = ("ipfm.", "taam.", "gsgm.")

    def __init__(
        self,
        Q=None,
        A=None,
        C1: int = 20,
        C2: int = 1,
        Classes: int = 15,
        image_size: int = 11,
        embed_dim: int = 128,
        context_length: int = 77,
        vocab_size: int = 49408,
        transformer_width: int = 64,
        lambda_re: float = 0.01,
        lambda_c: float = 1.0,
        lambda_global: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.Classes = Classes
        self.embed_dim = embed_dim
        self.lambda_re = lambda_re
        self.lambda_c = lambda_c
        self.lambda_global = lambda_global
        self.ipfm = IPFM(C1, C2, Classes)
        self.taam = TAAM(
            embed_dim,
            context_length,
            vocab_size,
            transformer_width,
            lambda_re,
            lambda_c,
        )
        self.gsgm = GSGM(C2, Classes)

    @property
    def encode_text(self):
        return self.taam.encode_text

    def state_dict(self, *args, **kwargs):
        nested_state = super().state_dict(*args, **kwargs)
        legacy_state = nested_state.__class__()
        for key, value in nested_state.items():
            legacy_key = key
            for prefix in self._MODULE_PREFIXES:
                if key.startswith(prefix):
                    legacy_key = key[len(prefix) :]
                    break
            legacy_state[legacy_key] = value
        return legacy_state

    def load_state_dict(self, state_dict, strict=True):
        if any(key.startswith(self._MODULE_PREFIXES) for key in state_dict):
            return super().load_state_dict(state_dict, strict=strict)
        target_keys = set(super().state_dict())
        nested_state = state_dict.__class__()
        for key, value in state_dict.items():
            matches = [
                prefix + key
                for prefix in self._MODULE_PREFIXES
                if prefix + key in target_keys
            ]
            nested_state[matches[0] if len(matches) == 1 else key] = value
        return super().load_state_dict(nested_state, strict=strict)

    def forward(self, x1, x2, x3, text=None, text_proto=None, labels=None):
        B = x1.shape[0]
        (
            Fuse_result,
            HSI_result,
            LiDAR_result,
            xh_flat,
            xl_flat,
            fuse_flat,
        ) = self.ipfm(x1, x2)
        img_feat1, img_feat2, img_feat_fused = self.taam.project_features(
            xh_flat, xl_flat, fuse_flat
        )
        VSS_result = self.gsgm(x3, B)
        re_loss, loss_c = self.taam.compute_losses(
            img_feat1,
            img_feat2,
            img_feat_fused,
            text,
            text_proto,
            x1.device,
        )
        final_out = (
            Fuse_result * VSS_result
            + HSI_result * VSS_result
            + LiDAR_result * VSS_result
        )
        return (final_out, re_loss, loss_c)

