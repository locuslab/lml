import os

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .utils import LabelNoise, split_dataset


def get_loaders(args):
    if args.dataset == 'cifar100':
        loaders = loaders_cifar
    elif args.dataset == 'imagenet':
        loaders = loaders_imagenet
    else:
        raise ValueError("dataset {} is not available".format(args.dataset))

    return loaders(dataset_name=args.dataset, batch_size=args.batch_size,
                   test_batch_size=args.test_batch_size,
                   cuda=args.cuda, topk=args.topk, train_size=args.train_size,
                   val_size=args.val_size, noise=args.noise_labels,
                   augment=args.augment, multiple_crops=args.multiple_crops,
                   data_root=args.data_root, use_dali=args.use_dali)


def loaders_cifar(dataset_name, batch_size, cuda,
                  train_size, augment=True, val_size=5000,
                  test_batch_size=1000, topk=None, noise=False,
                  multiple_crops=False, data_root=None, use_dali=False):

    assert not use_dali
    assert dataset_name == 'cifar100'
    assert not multiple_crops, "no multiple crops for CIFAR-100"

    data_root = data_root if data_root is not None else os.environ['VISION_DATA']
    root = '{}/{}'.format(data_root, dataset_name)

    # Data loading code
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    normalize = transforms.Normalize(mean=[x / 255.0 for x in mean],
                                     std=[x / 255.0 for x in std])

    if use_dali:
        assert augment

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    # define two datasets in order to have different transforms
    # on training and validation (no augmentation on validation)
    dataset_train = datasets.CIFAR100(root=root, train=True,
                                      transform=transform_train, download=True)
    dataset_val = datasets.CIFAR100(root=root, train=True,
                                    transform=transform_test, download=True)
    dataset_test = datasets.CIFAR100(root=root, train=False,
                                     transform=transform_test, download=True)

    # label noise
    if noise:
        dataset_train = LabelNoise(dataset_train, k=5, n_labels=100, p=noise)

    return create_loaders(dataset_name, dataset_train, dataset_val,
                          dataset_test, train_size, val_size, batch_size,
                          test_batch_size, cuda, num_workers=4, noise=noise)


def loaders_imagenet(dataset_name, batch_size, cuda,
                     train_size, augment=True, val_size=50000,
                     test_batch_size=256, topk=None, noise=False,
                     multiple_crops=False, data_root=None, use_dali=False):

    assert dataset_name == 'imagenet'
    data_root = data_root if data_root is not None else \
                os.environ['VISION_DATA_SSD']
    root = '{}/ILSVRC2012-smooth-topk-splits/images'.format(data_root)

    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    testdir = os.path.join(root, 'test')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)

    if multiple_crops:
        print('Using multiple crops')
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            lambda x: [normalize(transforms.functional.to_tensor(img)) for img in x]])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform_train = transform_test

    dataset_train = datasets.ImageFolder(traindir, transform_train)
    dataset_val = datasets.ImageFolder(valdir, transform_test)
    dataset_test = datasets.ImageFolder(testdir, transform_test)

    return create_loaders(dataset_name, dataset_train, dataset_val,
                          dataset_test, train_size, val_size, batch_size,
                          test_batch_size, cuda, noise=noise, num_workers=32,
                          use_dali=use_dali, traindir=traindir, valdir=valdir,
                          multiple_crops=multiple_crops)


def create_loaders(dataset_name, dataset_train, dataset_val, dataset_test,
                   train_size, val_size, batch_size, test_batch_size, cuda,
                   num_workers, topk=None, noise=False, use_dali=False,
                   traindir=None, valdir=None, multiple_crops=False):

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {}

    dataset_train, dataset_val = split_dataset(dataset_train, dataset_val, train_size, val_size)

    print('Dataset sizes: \t train: {} \t val: {} \t test: {}'
          .format(len(dataset_train), len(dataset_val), len(dataset_test)))

    if dataset_name == 'imagenet' and use_dali:
        assert traindir is not None
        n_threads = 8
        new_traindir = create_symlink_dataset(traindir, dataset_train)
        pipe = DALITrainPipe(
            batch_size=batch_size, num_threads=n_threads,
            device_id=0, data_dir=new_traindir, crop=224,
            dali_cpu=False)
        pipe.build()
        train_loader = DALIClassificationIterator(
            pipe, size=int(pipe.epoch_size("Reader"))
        )
    else:
        train_loader = data.DataLoader(
            dataset_train, batch_size=batch_size,
            shuffle=True, **kwargs)

    if not multiple_crops and dataset_name == 'imagenet' and use_dali:
        assert valdir is not None
        n_threads = 4
        pipe = DALIValPipe(
            batch_size=test_batch_size, num_threads=n_threads,
            device_id=0, data_dir=valdir, crop=224, size=256,
            dali_cpu=False)
        pipe.build()
        val_loader = DALIClassificationIterator(
            pipe, size=int(pipe.epoch_size("Reader"))
        )
    else:
        val_loader = data.DataLoader(
            dataset_val, batch_size=test_batch_size,
            shuffle=False, **kwargs)

    test_loader = data.DataLoader(
        dataset_test, batch_size=test_batch_size,
        shuffle=False, **kwargs)

    train_loader.tag = 'train'
    val_loader.tag = 'val'
    test_loader.tag = 'test'

    return train_loader, val_loader, test_loader


def create_symlink_dataset(traindir, dataset_train):
    if len(dataset_train.imgs) == 1231167:
        return traindir

    root = os.path.split(os.path.abspath(traindir))[0]
    new_traindir = os.path.join(root, 'train-{}'.format(len(dataset_train.imgs)))
    if os.path.exists(new_traindir):
        print('{} already exists, not creating'.format(new_traindir))
        return new_traindir
    else:
        print('Creating {}'.format(new_traindir))

    for (path, _) in dataset_train.imgs:
        new_path = new_traindir + '/' + '/'.join(path.split('/')[-2:])
        new_dir = os.path.dirname(new_path)
        os.makedirs(new_dir, exist_ok=True)
        os.symlink(path, new_path)

    return new_traindir


try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types

    class DALITrainPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, data_dir,
                     crop, dali_cpu=False):
            super(DALITrainPipe, self).__init__(
                batch_size, num_threads, device_id, seed=12 + device_id
            )
            self.input = ops.FileReader(
                file_root=data_dir, shard_id=device_id,
                num_shards=1, random_shuffle=True
            )
            #let user decide which pipeline works him bets for RN version he runs
            if dali_cpu:
                dali_device = "cpu"
                self.decode = ops.HostDecoder(
                    device=dali_device, output_type=types.RGB)
            else:
                dali_device = "gpu"
                # This padding sets the size of the internal nvJPEG buffers to be
                # able to handle all images from full-sized ImageNet
                # without additional reallocations
                self.decode = ops.nvJPEGDecoder(
                    device="mixed", output_type=types.RGB,
                    device_memory_padding=211025920, host_memory_padding=140544512
                )

            self.rrc = ops.RandomResizedCrop(
                device=dali_device, size =(crop, crop))
            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(crop, crop),
                image_type=types.RGB,
                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                std=[0.229 * 255,0.224 * 255,0.225 * 255])
            self.coin = ops.CoinFlip(probability=0.5)
            print('DALI "{0}" variant'.format(dali_device))

        def define_graph(self):
            rng = self.coin()
            self.jpegs, self.labels = self.input(name="Reader")
            images = self.decode(self.jpegs)
            images = self.rrc(images)
            output = self.cmnp(images.gpu(), mirror=rng)
            return [output, self.labels]

    class DALIValPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, data_dir,
                     crop, size, dali_cpu=False):
            super(DALIValPipe, self).__init__(
                batch_size, num_threads, device_id, seed=12 + device_id
            )
            self.input = ops.FileReader(
                file_root=data_dir, shard_id=device_id,
                num_shards=1, random_shuffle=True
            )
            #let user decide which pipeline works him bets for RN version he runs
            if dali_cpu:
                dali_device = "cpu"
                self.decode = ops.HostDecoder(
                    device=dali_device, output_type=types.RGB)
            else:
                dali_device = "gpu"
                # This padding sets the size of the internal nvJPEG buffers to be
                # able to handle all images from full-sized ImageNet
                # without additional reallocations
                self.decode = ops.nvJPEGDecoder(
                    device="mixed", output_type=types.RGB,
                    device_memory_padding=211025920, host_memory_padding=140544512
                )

            self.res = ops.Resize(
                device=dali_device, resize_shorter=size)
            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(crop, crop),
                image_type=types.RGB,
                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                std=[0.229 * 255,0.224 * 255,0.225 * 255])
            print('DALI "{0}" variant'.format(dali_device))

        def define_graph(self):
            self.jpegs, self.labels = self.input(name="Reader")
            images = self.decode(self.jpegs)
            images = self.res(images)
            output = self.cmnp(images.gpu())
            return [output, self.labels]


except ImportError:
    pass
    # raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
