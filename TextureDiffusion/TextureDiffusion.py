import os

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .TextureDiffusion_utils import AttentionBase

from torchvision.utils import save_image


class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(0, total_steps - start_step))

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)


        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out_u_target = self.attn_batch(qu[:num_heads], ku[-num_heads:], vu[-num_heads:], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_target = self.attn_batch(qu[:num_heads], kc[-num_heads:], vc[-num_heads:], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs) 
        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)

        return out

class MutualSelfAttentionControlMaskAuto(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, thres=0.1, ref_token_idx=[1], cur_token_idx=[1], mask_save_dir=None, model_type="SD"):
        """
        TextureDiffusion with mask auto generation from cross-attention map
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        # print("Using MutualSelfAttentionControlMaskAuto")
        self.thres = thres
        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx

        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.mask = None

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)

    def after_step(self):
        self.self_attns = []
        self.cross_attns = []

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, pre_sim, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q.shape[0] // num_heads   # 1 = 8 / 8
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")

        attn = sim.softmax(-1) #[8,N^2,N^2]

        if len(attn) == 2 * len(v): 
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out, sim #这里要返回没有softmax的版本

    def aggregate_cross_attn_map(self, idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))

        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source, sim_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, None, **kwargs)
        out_c_source, sim_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, None, **kwargs)

        if len(self.cross_attns) == 0:
            out_u_target = self.attn_batch(qu[:num_heads], ku[-num_heads:], vu[-num_heads:], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qu[:num_heads], kc[-num_heads:], vc[-num_heads:], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs) 
        else:
            mask = self.aggregate_cross_attn_map(idx=self.ref_token_idx)  # (2, H, W)
            mask_source = mask[-2]  # (H, W)
           
            res = int(np.sqrt(q.shape[1]))
            spatial_mask = F.interpolate(mask_source.unsqueeze(0).unsqueeze(0), (res, res)).reshape(-1, 1) 

            thres = self.thres
            spatial_mask[spatial_mask >= thres] = 1
            spatial_mask[spatial_mask < thres] = 0
            
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(spatial_mask.shape[0]))
                mask_image = spatial_mask.reshape(H, W).unsqueeze(0)
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_s_{self.cur_step}_{self.cur_att_layer}.png"))

            self.mask = spatial_mask # save the mask

            out_u_target, _ = self.attn_batch(qu[:num_heads], ku[-num_heads:], vu[-num_heads:], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, None, **kwargs)
            out_c_target, _ = self.attn_batch(qu[:num_heads], kc[-num_heads:], vc[-num_heads:], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, None, **kwargs) 

            # blend the self-attention results
            out_u_target = out_u_target * spatial_mask + out_u_source * (1 - spatial_mask)
            out_c_target = out_c_target * spatial_mask + out_c_source * (1 - spatial_mask)


        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out
