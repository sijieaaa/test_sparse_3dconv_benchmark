import numpy as np
import torch
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
import spconv


from spconv.core import ConvAlgo
import open3d

import spconv.pytorch as spconv
import MinkowskiEngine as ME
from tqdm import tqdm

import torch.nn as nn

from torchsparse.utils.quantize import sparse_quantize


from torchsparse import nn as spnn
from torchsparse import SparseTensor



batch_size = 128
num_points = 4096
quantization_size = 0.05
num_iters = 200
kernel_size = 5




def quantization(pc, quantization_size):
    quantized_pc = ME.utils.sparse_quantize(pc, quantization_size=quantization_size, return_index=False)
    return quantized_pc




class MinkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ME.MinkowskiConvolution(1, 64, kernel_size=kernel_size,dimension=3), # [1,64,5,3]
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(64, 64, kernel_size=kernel_size,dimension=3), # [1,64,5,3]
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(64, 64, kernel_size=kernel_size,dimension=3), # [1,64,5,3]
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(True),
        )
    def forward(self, x):
        x = self.model(x)
        return x





class SPNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            # spconv.SparseConv3d(1, 64, kernel_size=kernel_size, algo=ConvAlgo.MaskImplicitGemm),
            spconv.SubMConv3d(1, 64, kernel_size=3),
            spconv.SparseBatchNorm(64),
            spconv.SparseReLU(),
            # spconv.SparseConv3d(64, 64, kernel_size=kernel_size, algo=ConvAlgo.MaskImplicitGemm),
            spconv.SubMConv3d(64, 64, kernel_size=3),
            spconv.SparseBatchNorm(64),
            spconv.SparseReLU(),
            # spconv.SparseConv3d(64, 64, kernel_size=kernel_size, algo=ConvAlgo.MaskImplicitGemm),
            spconv.SubMConv3d(64, 64, kernel_size=3),
            spconv.SparseBatchNorm(64),
            spconv.SparseReLU(),
        )
    def forward(self, x):
        x = self.model(x)
        return x





class HanNet(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            spnn.Conv3d(1, 64, kernel_size),
            spnn.BatchNorm(64),
            spnn.ReLU(),
            spnn.Conv3d(64, 64, kernel_size),
            spnn.BatchNorm(64),
            spnn.ReLU(),
            spnn.Conv3d(64, 64, kernel_size),
            spnn.BatchNorm(64),
            spnn.ReLU(),
        )
    def forward(self, x):
        x = self.model(x)
        return x





class RandomDataset:
    def __init__(self, coords) -> None:
        self.coords = coords
    def __getitem__(self, _: int):
        coords = np.random.rand(num_points,3)
        feats = np.ones([num_points,1])
        coords -= np.min(coords, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords,
                                          voxel_size=quantization_size,
                                          return_index=True)
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)
        input = SparseTensor(coords=coords, feats=feats)
        return {'input': input}
    def __len__(self):
        return batch_size
    







pcs_raw = torch.rand([1,num_points, 3]).cuda()
pcs = pcs_raw.repeat(batch_size,1,1)
pcs = [quantization(pc,quantization_size=quantization_size) for pc in pcs]


c = ME.utils.batched_coordinates(pcs).cuda()
x_max = c[:,1].max()
y_max = c[:,2].max()
z_max = c[:,3].max()
x_min = c[:,1].min()
print(x_max,y_max,z_max)
f = torch.ones((c.shape[0], 1), dtype=torch.float32).cuda()
x_mink = ME.SparseTensor(features=f, coordinates=c)
minknet = MinkNet().cuda()



features = f.cuda() 
indices = c.cuda() 
spatial_shape =  [x_max,y_max,z_max] 
batch_size = batch_size 
x_sp = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
spnet = SPNet().cuda()


hannet = HanNet().cuda()







print(len(x_mink))
for i in tqdm(range(num_iters)):
    output = minknet(x_mink)



print(len(x_sp.indices))
for i in tqdm(range(num_iters)):
    output = spnet(x_sp)



from torchsparse.utils.collate import sparse_collate_fn
dataset = RandomDataset(coords=pcs_raw[0])
dataflow = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=sparse_collate_fn,
)
for k, feed_dict in tqdm(enumerate(dataflow)):
    inputs = feed_dict['input'].cuda()
    outputs = hannet(inputs)



print(len(inputs.coords))
for i in tqdm(range(num_iters)):
    output = hannet(inputs)
