import gzip
import os
import torch
from io import BytesIO
from diskcache.core import MODE_BINARY, BytesType
from diskcache import FanoutCache, Disk
from torchmetrics.functional.classification import binary_accuracy, binary_recall, binary_precision, binary_f1_score


cache_root = '/data_hdd2/users/ZhouHeng/Projects/HW/ParallelComputerVision/Luna/'


class GzipDisk(Disk):
    def store(self, value, read, key=None):
        if type(value) is BytesType:
            if read:
                value = value.read()
                read = False
            str_io = BytesIO()
            gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)
            for offset in range(0, len(value), 2**30):
                gz_file.write(value[offset:offset+2**30])
            gz_file.close()
            value = str_io.getvalue()
        return super(GzipDisk, self).store(value, read)

    def fetch(self, mode, filename, value, read):
        value = super(GzipDisk, self).fetch(mode, filename, value, read)
        if mode == MODE_BINARY:
            str_io = BytesIO(value)
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)
            read_csio = BytesIO()
            while True:
                uncompressed_data = gz_file.read(2**30)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break
            value = read_csio.getvalue()
        return value


def get_cache(scope_str):
    return FanoutCache(
        os.path.join(cache_root + scope_str),
        disk=GzipDisk,
        shards=64,
        timeout=1,
        size_limit=3e11
    )


def dice_loss(prediction, label, epsilon=1):
    dice_label = label.sum(dim=[1, 2, 3])
    dice_prediction = prediction.sum(dim=[1, 2, 3])
    dice_correct = (prediction * label).sum(dim=[1, 2, 3])

    dice_ratio = (2 * dice_correct + epsilon) / (dice_prediction + dice_label + epsilon)

    return 1 - dice_ratio


@torch.no_grad()
def get_statistics(prediction_prob, label, threshold):
    prediction_bool = (prediction_prob[:, 0:1] > threshold).to(torch.float32)
    tp = (prediction_bool * label).sum(dim=[1, 2, 3])
    fn = ((1 - prediction_bool) * label).sum(dim=[1, 2, 3])
    fp = (prediction_bool * (~label)).sum(dim=[1, 2, 3])
    return tp, fn, fp


def get_accuracy(y_hat, y):
    acc = binary_accuracy(y_hat, y)
    return acc.cpu().numpy()


def get_recall(y_hat, y):
    rec = binary_recall(y_hat, y)
    return rec.cpu().numpy()


def get_precision(y_hat, y):
    pre = binary_precision(y_hat, y)
    return pre.cpu().numpy()


def get_f1(y_hat, y):
    f1 = binary_f1_score(y_hat, y)
    return f1.cpu().numpy()