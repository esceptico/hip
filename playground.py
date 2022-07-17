import torch

from hip import HiP, BlockConfig, HiPConfig


if __name__ == '__main__':
    inputs = torch.randn(2, 128, 32)
    mask = torch.randint(0, 2, (2, 128))

    config = HiPConfig(
        input_dim=32,
        blocks=[
            BlockConfig(16, 2, 4, 128, 128),
            BlockConfig(4, 2, 4, 256, 256),
            BlockConfig(1, 18, 4, 256, 512),
            BlockConfig(1, 2, 4, 64, 1024),
            BlockConfig(1, 1, 4, 256, 512),
            BlockConfig(4, 1, 4, 256, 256),
            BlockConfig(16, 1, 4, 128, 128),
        ]
    )

    hip = HiP(config)

    print(f'{sum(p.numel() for p in hip.parameters()):_}')
    outputs = hip(inputs)

    print(outputs.shape)
