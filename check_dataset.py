from utils.dataset import IAMDataset, decode_text

ds = IAMDataset('data/iam', split='train', img_height=64)
print(f'Train: {len(ds)} samples')
img, label, length = ds[0]
print(f'Sample image shape: {img.shape}')
print(f'Sample text: "{decode_text(label)}"')

ds_val = IAMDataset('data/iam', split='val', img_height=64)
print(f'Val:   {len(ds_val)} samples')

ds_test = IAMDataset('data/iam', split='test', img_height=64)
print(f'Test:  {len(ds_test)} samples')
print('Dataset OK!')
