import torch
import argparse
import torchgeo.datasets
import torchvision.models as models


def main():
    args = parser.parse_args()
    path_weights = "pretrainedWeights/moco_geo+tp.pth.tar"
    checkpoint = torch.load(path_weights, map_location=torch.device('cpu'))
    model_weights = checkpoint['state_dict']
    print(model_weights)

    dataset = torchgeo.datasets.BigEarthNet(root='BigEarthNet', download=True, bands="s2")


if __name__ == '__main__':
    main()
