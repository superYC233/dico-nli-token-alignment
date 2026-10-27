"""准备 NLTK 英文 WordNet；训练批次中不会联网或下载。"""
import hashlib
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree

import nltk
from nltk.corpus import wordnet as wn

from experiments.deberta_kim_gate import config as cfg


INDEX_URL = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml'
CORPUS_URL = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip'


def download_wordnet():
    # 显式指定项目内目录，不写到系统 Python 或用户公共 nltk_data 目录。
    folder = cfg.WORDNET_DATA_DIR / 'corpora'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'wordnet.zip'
    if target.exists():
        return target
    print('下载 NLTK 官方英文 WordNet（约 11 MB，仅首次需要）……', flush=True)
    try:
        with urlopen(INDEX_URL, timeout=40) as response:
            index = ElementTree.fromstring(response.read())
        package = next(p.attrib for p in index.iter('package') if p.attrib.get('id') == 'wordnet')
        algorithm = 'sha256' if package.get('sha256_checksum') else 'md5'
        expected = package.get('sha256_checksum') or package['checksum']
        partial = folder / 'wordnet.zip.part'
        with urlopen(CORPUS_URL, timeout=60) as response, partial.open('wb') as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        actual = hashlib.new(algorithm, partial.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('WordNet 下载文件校验失败，不使用该文件。')
        partial.replace(target)
    except Exception as error:
        raise RuntimeError(
            'WordNet 下载未完成。请联网后单独运行 prepare_wordnet.py；'
            f'也可将官方 wordnet.zip 放到 {folder}。原错误：{error}'
        ) from error
    return target


@lru_cache(maxsize=1)
def ensure_wordnet(allow_download=False):
    folder = str(cfg.WORDNET_DATA_DIR)
    if folder not in nltk.data.path:
        nltk.data.path.insert(0, folder)
    try:
        wn.ensure_loaded()
    except LookupError:
        if not allow_download:
            raise RuntimeError('未找到 WordNet，请先运行 prepare_wordnet.py。') from None
        download_wordnet()
        nltk.data.clear_cache()
        wn.ensure_loaded()
    version = wn.get_version()
    if version != cfg.WORDNET_EXPECTED_VERSION:
        raise ValueError(f'WordNet 版本是 {version}，本版要求 {cfg.WORDNET_EXPECTED_VERSION}。')
    return wn


def resource_info():
    reader = ensure_wordnet()
    result = {'wordnet_version': reader.get_version(), 'nltk_version': nltk.__version__}
    archive = cfg.WORDNET_DATA_DIR / 'corpora/wordnet.zip'
    if archive.exists():
        result['wordnet_archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    result['reader_root'] = str(reader.root)
    return result
