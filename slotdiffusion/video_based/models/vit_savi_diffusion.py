import copy
import enum
import math

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Resize
import timm

from nerv.training import BaseModel
from nerv.models import deconv_out_shape, conv_norm_act, deconv_norm_act

from .savi import SlotAttention
from .savi_diffusion import ConsistentSAViDiffusion
from .utils import assert_shape, SoftPositionEmbed, torch_cat
from .predictor import ResidualMLPPredictor, TransformerPredictor, \
    RNNPredictorWrapper


def resize_patches_to_image(patches, size=None, scale_factor=None, resize_mode="bilinear"):
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to.
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns:
        Tensor of shape (..., C, S, S) where S is the image size.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = math.sqrt(n_patches)
    patch_size = int(math.sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = F.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])

class _VitFeatureType(enum.Enum):
    BLOCK = 1
    KEY = 2
    VALUE = 3
    QUERY = 4
    CLS = 5


class _VitFeatureHook:
    """Auxilliary class used to extract features from timm ViT models."""

    def __init__(self, feature_type: _VitFeatureType, block: int, drop_cls_token: bool = True):
        """Initialize VitFeatureHook.

        Args:
            feature_type: Type of feature to extract.
            block: Number of block to extract features from. Note that this is not zero-indexed.
            drop_cls_token: Drop the cls token from the features. This assumes the cls token to
                be the first token of the sequence.
        """
        assert isinstance(feature_type, _VitFeatureType)
        self.feature_type = feature_type
        self.block = block
        self.drop_cls_token = drop_cls_token
        self.name = f"{feature_type.name.lower()}{block}"
        self.remove_handle = None  # Can be used to remove this hook from the model again

        self._features = None

    @staticmethod
    def create_hook_from_feature_level(feature_level):
        feature_level = str(feature_level)
        prefixes = ("key", "query", "value", "block", "cls")
        for prefix in prefixes:
            if feature_level.startswith(prefix):
                _, _, block = feature_level.partition(prefix)
                feature_type = _VitFeatureType[prefix.upper()]
                block = int(block)
                break
        else:
            feature_type = _VitFeatureType.BLOCK
            try:
                block = int(feature_level)
            except ValueError:
                raise ValueError(f"Can not interpret feature_level '{feature_level}'.")

        return _VitFeatureHook(feature_type, block)

    def register_with(self, model):
        supported_models = (
            timm.models.vision_transformer.VisionTransformer,
            timm.models.beit.Beit,
            timm.models.vision_transformer_sam.VisionTransformerSAM,
        )
        model_names = ["vit", "beit", "samvit"]

        if not isinstance(model, supported_models):
            raise ValueError(
                f"This hook only supports classes {', '.join(str(cl) for cl in supported_models)}."
            )

        if self.block > len(model.blocks):
            raise ValueError(
                f"Trying to extract features of block {self.block}, but model only has "
                f"{len(model.blocks)} blocks"
            )

        block = model.blocks[self.block - 1]
        if self.feature_type == _VitFeatureType.BLOCK:
            self.remove_handle = block.register_forward_hook(self)
        else:
            if isinstance(block, timm.models.vision_transformer.ParallelBlock):
                raise ValueError(
                    f"ViT with `ParallelBlock` not supported for {self.feature_type} extraction."
                )
            elif isinstance(model, timm.models.beit.Beit):
                raise ValueError(f"BEIT not supported for {self.feature_type} extraction.")
            self.remove_handle = block.attn.qkv.register_forward_hook(self)

        model_name_map = dict(zip(supported_models, model_names))
        self.model_name = model_name_map.get(type(model), None)

        return self

    def pop(self) -> torch.Tensor:
        """Remove and return extracted feature from this hook.

        We only allow access to the features this way to not have any lingering references to them.
        """
        assert self._features is not None, "Feature extractor was not called yet!"
        features = self._features
        self._features = None
        return features

    def __call__(self, module, inp, outp):
        if self.feature_type == _VitFeatureType.BLOCK:
            features = outp
            print(features.shape)
            if self.drop_cls_token:
                # First token is CLS token.
                if self.model_name == "samvit":
                    # reshape outp (B,H,W,C) -> (B,H*W,C)
                    features = outp.flatten(1,2)
                else:
                    features = features[:, 1:]
        elif self.feature_type in {
            _VitFeatureType.KEY,
            _VitFeatureType.QUERY,
            _VitFeatureType.VALUE,
        }:
            # This part is adapted from the timm implementation. Unfortunately, there is no more
            # elegant way to access keys, values, or queries.
            B, N, C = inp[0].shape
            qkv = outp.reshape(B, N, 3, C)  # outp has shape B, N, 3 * H * (C // H)
            q, k, v = qkv.unbind(2)

            if self.feature_type == _VitFeatureType.QUERY:
                features = q
            elif self.feature_type == _VitFeatureType.KEY:
                features = k
            else:
                features = v
            if self.drop_cls_token:
                # First token is CLS token.
                features = features[:, 1:]
        elif self.feature_type == _VitFeatureType.CLS:
            # We ignore self.drop_cls_token in this case as it doesn't make any sense.
            features = outp[:, 0]  # Only get class token.
        else:
            raise ValueError("Invalid VitFeatureType provided.")

        self._features = features


class ConsistentViTSAViDiffusion(ConsistentSAViDiffusion):
    """SA model with stochastic kernel and additional prior_slots head.
    Encoder is replaced by a SAM ViT.
    If loss_dict['kld_method'] = 'none', it becomes a standard SAVi model.
    """

    def __init__(
            self,
            resolution,
            clip_len,
            slot_dict=dict(
                num_slots=7,
                slot_size=128,
                slot_mlp_size=256,
                num_iterations=2,
            ),
            enc_dict=dict(
                enc_channels=(3, 64, 64, 64, 64),
                enc_ks=5,
                enc_out_channels=128,
                enc_norm='',
            ),
            dec_dict=dict(
                resolution=(128, 128),
                vae_dict=dict(),
                unet_dict=dict(),
                use_ema=True,
                diffusion_dict=dict(
                    timesteps=1000,
                    beta_schedule="linear",
                    linear_start=1e-4,
                    linear_end=2e-2,
                    cosine_s=8e-3,
                    log_every_t=100,  # log every t steps in denoising sampling
                    logvar_init=0.,
                ),
                conditioning_key='crossattn',  # 'concat'
                cond_stage_key='slots',
            ),
            pred_dict=dict(
                pred_type='transformer',
                pred_rnn=True,
                pred_norm_first=True,
                pred_num_layers=2,
                pred_num_heads=4,
                pred_ffn_dim=512,
                pred_sg_every=None,
            ),
            loss_dict=dict(use_denoise_loss=True, ),
            eps=1e-6,
    ):
        super().__init__(
            resolution=resolution,
            clip_len=clip_len,
            slot_dict=slot_dict,
            enc_dict=enc_dict,
            dec_dict=dec_dict,
            pred_dict=pred_dict,
            loss_dict=loss_dict,
            eps=eps,
        )

        # a hack for only extracting slots
        self.testing = False

    def _build_encoder(self):
        self.vit_model_name = self.enc_dict['vit_model_name']
        self.vit_use_pretrained = self.enc_dict['vit_use_pretrained']
        self.vit_freeze = self.enc_dict['vit_freeze']
        self.vit_feature_level = self.enc_dict['vit_feature_level']
        self.vit_num_patches = self.enc_dict['vit_num_patches']

        def feature_level_to_list(feature_level):
            if feature_level is None:
                return []
            elif isinstance(feature_level, (int, str)):
                return [feature_level]
            else:
                return list(feature_level)

        self.feature_levels = feature_level_to_list(self.vit_feature_level)

        if "samvit" in self.vit_model_name:
            model = timm.create_model(self.vit_model_name, pretrained=self.vit_use_pretrained)
        else:              
            model = timm.create_model(self.vit_model_name, pretrained=self.vit_use_pretrained, dynamic_img_size=True)
        # Delete unused parameters from classification head
        if hasattr(model, "head"):
            del model.head
        if hasattr(model, "fc_norm"):
            del model.fc_norm

        if len(self.feature_levels) > 0:
            self._feature_hooks = [
                _VitFeatureHook.create_hook_from_feature_level(level).register_with(model) for level in self.feature_levels
            ]
            feature_dim = model.num_features * len(self.feature_levels)

            # Remove modules not needed in computation of features
            max_block = max(hook.block for hook in self._feature_hooks)
            new_blocks = model.blocks[:max_block]  # Creates a copy
            del model.blocks
            model.blocks = new_blocks
            model.norm = nn.Identity()

        self.vit = model
        self._feature_dim = feature_dim

        if self.vit_freeze:
            self.vit.requires_grad_(False)
            # BatchNorm layers update their statistics in train mode. This is probably not desired
            # when the model is supposed to be frozen.
            contains_bn = any(
                isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                for m in self.vit.modules()
            )
            self.run_in_eval_mode = contains_bn
        else:
            self.run_in_eval_mode = False

        self._init_pos_embed(self.enc_dict["vit_out_dim"], self.enc_dict["enc_out_channels"])

        self.visual_channels = self.enc_dict['vit_out_dim']
        self.visual_resolution = (8, 8)

    def _init_pos_embed(self, encoder_output_dim, token_dim):
        layers = []
        layers.append(nn.LayerNorm(encoder_output_dim))
        layers.append(nn.Linear(encoder_output_dim, encoder_output_dim))
        nn.init.zeros_(layers[-1].bias)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(encoder_output_dim, token_dim))
        nn.init.zeros_(layers[-1].bias)
        self.encoder_pos_embedding = nn.Sequential(*layers)

    def _build_loss(self):
        """Loss calculation settings."""
        self.use_denoise_loss = self.loss_dict['use_denoise_loss']
        self.use_consistency_loss = self.loss_dict['use_consistency_loss']
        assert self.use_denoise_loss

    def _transformer_compute_positions(self, features):
        """Compute positions for Transformer features."""
        n_tokens = features.shape[1]
        image_size = math.sqrt(n_tokens)
        image_size_int = int(image_size)
        assert (
            image_size_int == image_size
        ), "Position computation for Transformers requires square image"

        spatial_dims = (image_size_int, image_size_int)
        positions = torch.cartesian_prod(
            *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
        )
        return positions
    
    def vit_encode(self, x):
        if self.run_in_eval_mode and self.training:
            self.eval()

        if self.vit_freeze:
            # Speed things up a bit by not requiring grad computation.
            with torch.no_grad():
                features = self.vit.forward_features(x)
        else:
            features = self.vit.forward_features(x)

        if self._feature_hooks is not None:
            hook_features = [hook.pop() for hook in self._feature_hooks]

        if len(self.feature_levels) == 0:
            # Remove class token when not using hooks.
            features = features[:, 1:]
            positions = self._transformer_compute_positions(features)
        else:
            features = hook_features[: len(self.feature_levels)]
            positions = self._transformer_compute_positions(features[0])
            features = torch.cat(features, dim=-1)

        return features
    
    def _get_encoder_out(self, img):
        features = self.vit_encode(img)
        encoder_out = self.encoder_pos_embedding(features)
        # `encoder_out` has shape: [B, H*W, enc_out_channels]
        return encoder_out