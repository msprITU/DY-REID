"""
Elif Ecem Akbaba tarafindan
vatana millete hayirli olmasi amaciyla
oradan buradan kod araklayarak tasarlanmistir.
"""
import torch
import numpy as np
from torch import nn, Tensor
from typing import Dict, Iterable, Callable

def ChannelAverage(feat_matrix):
    #print(feat_matrix.shape)
    av_feat_map_batch = np.zeros((feat_matrix.shape[0], 1, feat_matrix.shape[2], feat_matrix.shape[3]))
    for a in range(feat_matrix.shape[0]): #16 batchteki her sample icin loop
        total_feat_map = np.zeros((feat_matrix.shape[2], feat_matrix.shape[3]))
        for featmap_channel in range(feat_matrix.shape[1]): #her kanali donuyor
            feat_map_layer = feat_matrix[a, featmap_channel, :, :] #[a][featmap_channel]
            #print(feat_map_layer.shape)
            #feat_map_layer = feat_matrix[a][featmap_channel].cpu().detach().numpy() #bunlar tensore uygulanır, feat map burda nparray
            total_feat_map = np.add(total_feat_map, feat_map_layer)
        average_feat_map = total_feat_map / feat_matrix.shape[1] #1 samplein av feat mapi
        av_feat_map_batch[a, 0, :, :] = average_feat_map
    #print(av_feat_map_batch.shape)
    return av_feat_map_batch


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features

class VerboseExecution(nn.Module):
    #her layerin output shapei ve ismi icin
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.layer_names = []

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            self.layer_names.append(layer.__name__)
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        print(self.layer_names)
        return self.model(x)
