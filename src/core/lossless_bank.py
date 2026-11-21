"""Lossless archives for signed INT4 banks containing only nonnegative codes."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_LZMA, ZipFile

import numpy as np


ENCODINGS = {
    'deflate-int4': (False, ZIP_DEFLATED),
    'lzma-int4': (False, ZIP_LZMA),
    'deflate-planes3': (True, ZIP_DEFLATED),
    'lzma-planes3': (True, ZIP_LZMA),
}


def encode_planes(bank):
    if bank.ndim != 2 or bank.dtype != np.uint8 or bank.shape[1] % 4:
        raise ValueError('Expected INT4 rows with a multiple of eight symbols')
    low, high = bank & 15, bank >> 4
    if min(low.min(), high.min()) < 7 or max(low.max(), high.max()) > 14:
        raise ValueError('This bank needs the full signed INT4 range')
    codes = np.stack((low, high), axis=2).reshape(len(bank), -1) - 7
    bits = np.stack([np.packbits((codes >> bit) & 1, axis=1, bitorder='little') for bit in range(3)])
    return np.ascontiguousarray(bits.transpose(0, 2, 1))


def recover_planes(planes):
    if planes.ndim != 3 or planes.dtype != np.uint8 or planes.shape[0] != 3:
        raise ValueError('Invalid INT4 bit planes')
    bits = np.unpackbits(planes.transpose(0, 2, 1), axis=2, bitorder='little')
    codes = 7 + (bits[0] | (bits[1] << 1) | (bits[2] << 2))
    return codes[:, ::2] | (codes[:, 1::2] << 4)


def transcode(payload, codec):
    use_planes, compression = ENCODINGS[codec]
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        fields = dict(archive)
    if 'bank_planes' in fields:
        fields['bank'] = recover_planes(fields.pop('bank_planes'))
    if use_planes and not bool(fields.get('unsigned', False)):
        bank = fields['bank']
        try:
            planes = encode_planes(bank)
        except ValueError:
            pass  # Other bank formats still get ordinary lossless ZIP compression.
        else:
            fields.pop('bank')
            fields['bank_planes'] = planes
    stream = BytesIO()
    with ZipFile(stream, mode='w', compression=compression, compresslevel=9) as archive:
        for key, values in fields.items():
            with archive.open(key + '.npy', mode='w') as member:
                np.lib.format.write_array(member, values, allow_pickle=False)
    return stream.getvalue()
