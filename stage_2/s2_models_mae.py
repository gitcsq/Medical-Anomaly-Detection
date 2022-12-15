
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from util.patch_embed_ import PatchEmbed_


class cls_token_model(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=1, num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=1, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_patches=None, k=4):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed_(img_size, patch_size, in_chans, embed_dim)
        self.s1_pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.MLP = nn.Sequential(
            nn.Linear(1024, 1024*k),
            nn.Linear(1024*k, 1024)
        )
        self.MLP_img = nn.Sequential(
            nn.Linear(1024, 1024*k),
            nn.Linear(1024*k, in_chans*256)
        )
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(num_patches + 1, 1, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        # self.initialize_weights()

    def initialize_weights(self):

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, label_poses):

        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # b, 1, 1024

        cls_tokens = cls_tokens + label_poses
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # 64, 33, 1024
        x = x[:, :1, :]  # 64, 1, 1024
        x = self.MLP(x)  # 64, 1, 1024 -> k*1024 -> 1024

        return x

    def forward_loss(self, target, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        loss = (pred - target) ** 2
        loss = loss.mean()  # [N, L], mean loss per patch
        return loss

    # source为latent的训练主函数
    def forward(self, patches, labels, pos_embeds):
        pred = self.forward_encoder(patches, pos_embeds)
        loss = self.forward_loss(labels, pred)
        return loss, pred

    # source为latent的测试主函数
    def forward_test(self, patches, labels, pos_embeds):
        a, b, c, d = patches.shape
        patches = patches.reshape((a*b, c, d))
        labels = labels.reshape((a*b, 1, d))
        pos_embeds = pos_embeds.reshape((a*b, 1, d))
        pred = self.forward_encoder(patches, pos_embeds)
        loss = self.forward_loss(labels, pred)
        return loss

    # source为img的测试主函数
    def forward_test_img(self, data, label, data_pos, label_pos):
        # b, bs, patch_num, channel_num, patch_size, patch_size_ = x.shape
        #    64, 32,        3,           16,         16
        # print(data.shape, label.shape, data_pos.shape, label_pos.shape)
        a, b, c, d, e, f = data.shape
        data = data.reshape((a * b, c, d, e, f))  # b, 32, 3, 16, 16
        label = label.reshape((a * b, 1, d, e, f))  # b, 1, 3, 16, 16
        data_pos = data_pos.reshape((a * b, c, 1024))  # 36, 32, 1024
        label_pos = label_pos.reshape((a * b, 1, 1024))  # 36, 1, 1024
        # print('forward_test_img')
        # print(data.shape, label.shape, data_pos.shape, label_pos.shape)
        pred = self.forward_encoder_img(data, data_pos, label_pos)
        label = label.reshape((label.shape[0], label.shape[1], -1))

        loss = self.forward_loss(label, pred)
        return loss

    # source为img的训练主函数
    def forward_img(self, data, label, data_pos, label_pos):
        pred = self.forward_encoder_img(data, data_pos, label_pos)
        label = label.reshape((label.shape[0], label.shape[1], -1))
        # print(label.shape, pred.shape)
        loss = self.forward_loss(label, pred)
        return loss, pred

    def forward_encoder_img(self, x, x_pos, label_pos):
        # bs, patch_num, channel_num, patch_size, patch_size_ = x.shape
        # 64, 32,        3,           16,         16
        x = x.transpose(1, 2)  # b, 3, 32, 16, 16
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2]*x.shape[3], -1))  # 64, 3, 512, 16
        x = self.patch_embed(x)  # 64, 3, 512, 16      64, 32, 1024
        x = x + x_pos
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # b, 1, 1024
        cls_tokens = cls_tokens + label_pos
        x = torch.cat((cls_tokens, x), dim=1)  # 64, 33, 1024
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # 64, 33, 1024
        x = x[:, :1, :]  # 64, 1, 1024
        x = self.MLP_img(x)  # 64, 1, 1024 -> k*1024 -> 1024 -> 256    64, 1, 256
        return x


class fc_model(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=1, num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=1, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_patches=48, k=4):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed_(img_size, patch_size, in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pred_fc = nn.Linear((1024 * num_patches), 1024)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.norm_pix_loss = norm_pix_loss

        # self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, target, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        loss = (pred - target) ** 2
        loss = loss.mean()  # [N, L], mean loss per patch
        return loss

    def forward(self, patches, labels, pos_embeds):
        pred = self.forward_encoder(patches, pos_embeds)
        loss = self.forward_loss(labels, pred)
        return loss, pred

    # source为latent的测试主函数
    def forward_test(self, patches, labels, pos_embeds):
        a, b, c, d = patches.shape
        patches = patches.reshape((a * b, c, d))
        labels = labels.reshape((a * b, 1, d))
        pos_embeds = pos_embeds.reshape((a * b, 1, d))
        pred = self.forward_encoder(patches, pos_embeds)
        loss = self.forward_loss(labels, pred)
        return loss

    # source为img的测试主函数
    def forward_test_img(self, data, label, data_pos, label_pos):
        # b, bs, patch_num, channel_num, patch_size, patch_size_ = x.shape
        #    64, 32,        3,           16,         16
        # print(data.shape, label.shape, data_pos.shape, label_pos.shape)
        a, b, c, d, e, f = data.shape
        data = data.reshape((a * b, c, d, e, f))  # b, 32, 3, 16, 16
        label = label.reshape((a * b, 1, d, e, f))  # b, 1, 3, 16, 16
        data_pos = data_pos.reshape((a * b, c, 1024))  # 36, 32, 1024
        label_pos = label_pos.reshape((a * b, 1, 1024))  # 36, 1, 1024
        # print('forward_test_img')
        # print(data.shape, label.shape, data_pos.shape, label_pos.shape)
        pred = self.forward_encoder_img(data, data_pos, label_pos)
        label = label.reshape((label.shape[0], label.shape[1], -1))

        loss = self.forward_loss(label, pred)
        return loss

    # source为img的训练主函数
    def forward_img(self, data, label, data_pos, label_pos):
        pred = self.forward_encoder_img(data, data_pos, label_pos)
        label = label.reshape((label.shape[0], label.shape[1], -1))
        # print(label.shape, pred.shape)
        loss = self.forward_loss(label, pred)
        return loss, pred

    def forward_encoder_img(self, x, x_pos, label_pos):
        # bs, patch_num, channel_num, patch_size, patch_size_ = x.shape
        # 64, 32,        3,           16,         16
        x = x.transpose(1, 2)  # b, 3, 32, 16, 16
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2]*x.shape[3], -1))  # 64, 3, 512, 16
        x = self.patch_embed(x)  # 64, 3, 512, 16      64, 32, 1024
        x = x + x_pos
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # b, 1, 1024
        cls_tokens = cls_tokens + label_pos
        x = torch.cat((cls_tokens, x), dim=1)  # 64, 33, 1024

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # 64, 33, 1024
        x = x[:, 1:, :]  # 64, 32, 1024
        x = x.reshape((x.shape[0], -1))  # 64, 32*1024
        x = self.pred_fc(x)  # 64, 1024
        x = x.unsqueeze(1)  # 64, 1, 1024

        return x

    def forward_encoder(self, x, pos_embeds):

        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # b, 1, 1024
        cls_tokens = cls_tokens + pos_embeds
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # 64, 33, 1024
        x = x[:, 1:, :]  # 64, 32, 1024
        x = x.reshape((x.shape[0], -1))  # 64, 32*1024
        x = self.pred_fc(x)  # 64, 1024
        x = x.unsqueeze(1)  # 64, 1, 1024

        return x


class mlp_model(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=1, num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=1, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_patches=48, k=4):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pred_fc = nn.Linear((1024 * num_patches), 1024)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.norm_pix_loss = norm_pix_loss

        # self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, pos_embeds):

        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # b, 1, 1024
        cls_tokens = cls_tokens + pos_embeds
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # 64, 33, 1024
        x = x[:, 1:, :]  # 64, 32, 1024
        x = x.reshape((x.shape[0], -1))  # 64, 32*1024
        x = self.pred_fc(x)  # 64, 1024
        x = x.unsqueeze(1)  # 64, 1, 1024

        return x

    def forward_loss(self, target, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        loss = (pred - target) ** 2
        loss = loss.mean()  # [N, L], mean loss per patch
        return loss

    def forward(self, patches, labels, pos_embeds):
        pred = self.forward_encoder(patches, pos_embeds)
        loss = self.forward_loss(labels, pred)
        return loss, pred

    def forward_test(self, patches, labels, pos_embeds):
        a, b, c, d = patches.shape
        patches = patches.reshape((a * b, c, d))
        labels = labels.reshape((a * b, 1, d))
        pos_embeds = pos_embeds.reshape((a * b, 1, d))
        pred = self.forward_encoder(patches, pos_embeds)
        loss = self.forward_loss(labels, pred)
        return loss


if __name__=="__main__":
    device = torch.device('cuda')
    model = cls_token_model(num_patches=32)
    x = torch.rand((2, 32, 1024))
    x = x.to(device)
    y = x[:,:1,:]
    y = y.to(device)
    model = model.to(device)
    y_pred = model(x, y)
    pass