"""Fetch the published split from SVD without downloading the 6 GB ZIP.

SVD: Pützer and Barry, Saarbruecken Voice Database, CC BY 4.0,
https://doi.org/10.5281/zenodo.16874898 . ZIP members are CRC checked;
each downloaded NSP and converted WAV is also recorded with SHA256.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import struct
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
SVD_RECORD = 'https://zenodo.org/api/records/16874898'
CHECKPOINT_URL = ('https://huggingface.co/speechbrain/tts-diffwave-ljspeech/'
                  'resolve/3a0cbf1/diffwave.ckpt')
CHECKPOINT_SHA = 'b434ac56f45486ec899e4ef50fb1bab09443bbe4ccc844acae35486e89541fc2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(url, start=None, end=None):
    headers = {'User-Agent': 'TorchDIVA-reproduction/1.0', 'Accept-Encoding': 'identity'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end - 1}'
    for attempt in range(6):
        try:
            with urlopen(Request(url, headers=headers), timeout=40) as response:
                if start is not None:
                    expected = f'bytes {start}-{end-1}/'
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                        raise ValueError('Server did not honor the requested byte range')
                    data = response.read(end-start+1)
                    if len(data) != end-start:
                        raise OSError(f'Incomplete range response: {len(data)} != {end-start}')
                    return data
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError, IncompleteRead) as error:
            if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 8))


class HTTPRangeFile(io.RawIOBase):
    """Seekable ZIP-directory reader. Never silently downloads the full ZIP."""
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.size) + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.position
        end = min(self.position + count, self.size)
        if end <= self.position:
            return b''
        data = get(self.url, self.position, end)
        self.position = end
        return data


def member_for_name(name):
    parts = Path(name).stem.split('_')
    session, vowel, pitch = parts[0], parts[-2], parts[-1]
    if vowel not in ('a', 'i', 'u') or pitch not in ('h', 'l', 'n'):
        raise ValueError(f'Unexpected published split filename: {name}')
    return f'{session}/vowels/{session}-{vowel}_{pitch}.nsp'


def fetch_member(url, size, info):
    # A complete local entry can be fetched in one small request. The extra
    # 4 KiB covers the local filename/extra fields; bounds are checked below.
    end = min(size, info.header_offset + info.compress_size + 4096)
    entry = get(url, info.header_offset, end)
    header = struct.unpack('<4s5H3L2H', entry[:30])
    if header[0] != b'PK\x03\x04' or header[2] & 1:
        raise ValueError('Invalid or encrypted ZIP member')
    name_size, extra_size = header[-2:]
    stored_name = entry[30:30+name_size].decode('utf8')
    if stored_name != info.filename:
        raise ValueError('ZIP filename mismatch')
    offset = 30 + name_size + extra_size
    compressed = entry[offset:offset+info.compress_size]
    if len(compressed) != info.compress_size:
        raise ValueError('Local ZIP header exceeds reserved range')
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError(f'Unsupported ZIP compression {info.compress_type}')
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError(f'CRC/size mismatch: {info.filename}')
    return data


def read_nsp(data):
    """Read mono Kay CSL/NSP DS16 chunks without changing the PCM samples.

    Format: https://mmsp.ece.mcgill.ca/Documents/AudioFormats/CSL/CSL.html
    The SVD HEDR count is in samples; SDA_ contains little-endian int16 PCM.
    """
    import numpy as np
    if data[:8] != b'FORMDS16':
        raise ValueError('Expected a Kay DS16 sound file')
    position, sample_rate, expected, pcm = 12, None, None, None
    while position + 8 <= len(data):
        tag = data[position:position+4]
        count = struct.unpack_from('<I', data, position+4)[0]
        payload = data[position+8:position+8+count]
        if len(payload) != count:
            raise ValueError('Truncated NSP chunk')
        if tag == b'HEDR':
            sample_rate, expected = struct.unpack_from('<II', payload, 20)
        elif tag in (b'SDA_', b'SD_A'):
            pcm = np.frombuffer(payload, dtype='<i2').copy()
        position += 8 + count + count % 2
    if pcm is None or sample_rate is None or len(pcm) != expected:
        raise ValueError('Missing or inconsistent mono NSP data/header')
    return pcm, sample_rate


def prepare(directory, workers=4, limit=None):
    import numpy as np
    import soundfile as sf

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    asset_dir = directory / 'assets'
    raw_dir, wav_dir = directory / 'raw_nsp', directory / 'human'
    for path in (asset_dir, raw_dir, wav_dir):
        path.mkdir(exist_ok=True)
    checkpoint = asset_dir / 'speechbrain-diffwave.ckpt'
    if not checkpoint.exists() or digest(checkpoint) != CHECKPOINT_SHA:
        checkpoint.write_bytes(get(CHECKPOINT_URL))
    if digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Pretrained checkpoint checksum mismatch')

    record_path = asset_dir / 'svd-record.json'
    if not record_path.exists():
        record_path.write_bytes(get(SVD_RECORD))
    record = json.loads(record_path.read_text())
    archive = next(item for item in record['files'] if item['key'] == 'healthy.zip')
    url, size = archive['links']['self'], archive['size']
    split = []
    for label, filename in [('train', 'dw_train_files.txt'), ('test', 'dw_test_files.txt')]:
        for name in (ROOT / 'test/figure_data' / filename).read_text().splitlines():
            if name.strip():
                split.append((label, name.strip()))
    if limit is not None:
        split = split[:limit]
    with zipfile.ZipFile(HTTPRangeFile(url, size)) as archive_file:
        infos = {name: archive_file.getinfo(member_for_name(name)) for _, name in split}
    print(f'Located {len(split)} published SVD files in the healthy archive.', flush=True)

    def one(label, name):
        info = infos[name]
        raw_path = raw_dir / (Path(name).stem + '.nsp')
        if raw_path.exists():
            data = raw_path.read_bytes()
        else:
            data = fetch_member(url, size, info)
            raw_path.write_bytes(data)
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise ValueError(f'Cached NSP checksum mismatch: {name}')
        wav = wav_dir / name
        pcm, sample_rate = read_nsp(data)
        if not wav.exists():
            sf.write(wav, pcm, sample_rate, subtype='PCM_16')
        converted, converted_rate = sf.read(wav, dtype='int16')
        if converted_rate != sample_rate or not np.array_equal(converted, pcm):
            raise ValueError(f'WAV does not preserve original NSP PCM: {name}')
        sound_info = sf.info(wav)
        return dict(name=name, split=label, session_id=name.split('_')[0],
                    speaker_id=name.split('_')[4], vowel=Path(name).stem.split('_')[-2],
                    pitch_condition=Path(name).stem.split('_')[-1],
                    archive_member=info.filename, archive_crc32=f'{info.CRC:08x}',
                    nsp_sha256=hashlib.sha256(data).hexdigest(), wav_sha256=digest(wav),
                    sample_rate=sound_info.samplerate, frames=sound_info.frames,
                    duration=sound_info.duration, human_path=f'human/{name}')

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, label, name) for label, name in split]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 25 == 0 or len(rows) == len(split):
                print(f'Fetched and converted {len(rows)}/{len(split)} files.', flush=True)
    rows.sort(key=lambda item: (item['split'], item['name']))
    manifest = dict(dataset='Saarbruecken Voice Database',
                    dataset_doi='10.5281/zenodo.16874898', dataset_license='CC-BY-4.0',
                    archive_url=url, checkpoint_url=CHECKPOINT_URL,
                    checkpoint_sha256=CHECKPOINT_SHA,
                    checkpoint_note='SpeechBrain LJSpeech weights; not the unavailable LMNT checkpoint',
                    samples=rows)
    manifest_path = directory / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    prepare(args.data_dir, args.workers, args.limit)
