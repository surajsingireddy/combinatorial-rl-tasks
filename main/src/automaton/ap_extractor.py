import torch
from torch_utils.dictlist import DictList

class DungeonQuestAPExtractor(APExtractor):
    def __init__(self, device: torch.device):
        self.device = device
    
    def num_transitions(self):
        return 16
    
    def extract_aps_batch(self, observations: DictList):
        return observations.zone_obs.sum(1)[:,1:5] @ 2**torch.arange(4, dtype=torch.float32, device=self.device)