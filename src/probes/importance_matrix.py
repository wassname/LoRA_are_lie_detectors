
import safetensors.torch
from einops import rearrange
import torch

def get_importance_matrix(saved_adaptop_file, layers=['fc1', 'Wqkv']):
    state_dict = safetensors.torch.load_file(saved_adaptop_file)
    keys = sorted(state_dict.keys())
    if layers is None:
        layers = list(set([k.split('.')[-2] for k in state_dict.keys()]))

    activations = {}

    for k in keys:
        suffix = k.split('.')[-2]
        if suffix not in activations:
            activations[suffix] = []
        activations[suffix].append(state_dict[k])

    for k in activations.keys():
        activations[k] = rearrange(activations[k], 'l h b -> (b l) h').detach().cpu().float()
        print(k, activations[k].shape)


    importance_matrix = torch.concat([activations[i] for i in layers], dim=1)
    
    # # normalize?
    # FIXME: I need to think of the best way to convert from adapter to importance matrix
    # for example we want extreme values so it makes a difference
    # we want it centered around 1
    # we might want it to add up to 1?
    # adapter weights <1 indicate less importance? Or less changes. Or neither.
    importance_matrix = (importance_matrix-1)
    # importance_matrix = importance_matrix ** 3 # square to make it positive
    importance_matrix = importance_matrix / importance_matrix.std()
    importance_matrix = importance_matrix + 1

    # square to make it positive
    return importance_matrix.abs().clamp(0, None) ** 3


# f = "/media/wassname/SGIronWolf/projects5/elk/sgd_probes_are_lie_detectors/notebooks/lightning_logs/version_276/checkpoint_last/adapter_model.safetensors"
# importance_matrix = get_importance_matrix(f)[SKIP::STRIDE, ::DECIMATE]
# print(importance_matrix.shape)
# plt.hist(importance_matrix.flatten(), bins=55);
