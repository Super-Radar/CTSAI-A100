"""CTSAI-A100 ADC 文本文件的鲁棒解析器。

仓内存在两种布局：

1. 无头布局（`示例adc数据和结果/adc_近波_*.txt`、`adc_远波_*.txt` 及
   MATLAB 平台 data/ 下的 `adc_test_*_f1_*.txt`）：
   整个文件为逗号分隔的 uint32 序列，无任何头字段。
2. 三字段头布局（MATLAB 平台 data/ 下的 `adc_test_*_Pf0_Rx*.txt`）：
   文件以 `<rx>,<rng_nfft>,<nchirp>,` 开头，其后为 uint32 载荷。

解析器对两种布局统一处理，并显式记录所有异常 token
（非数值 / 超出 uint32 范围），而不是静默跳过。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UINT32_MAX = 4_294_967_295
#: 头部 rx 序号的最大合法值（保守上界，仅用于格式识别）
_HEADER_RX_MAX = 15
#: 头部 rng_nfft 合法区间（具体数值由规则层与配置交叉核对）
_HEADER_NFFT_MIN, _HEADER_NFFT_MAX = 64, 16384
_HEADER_CHIRP_MIN, _HEADER_CHIRP_MAX = 8, 8192

_INT_RE = re.compile(r"^[+-]?\d+$")


@dataclass
class BadToken:
    """一个无法作为 uint32 采信的 token。"""

    index: int  # 在原始 token 流中的位置（0 起，含头字段）
    token: str
    reason: str  # "non-numeric" | "out-of-range(<0)" | "out-of-range(>2^32-1)"

    def as_dict(self) -> dict:
        return {"index": self.index, "token": self.token, "reason": self.reason}


@dataclass
class LoadedFile:
    """单个 ADC 文本文件的解析结果。"""

    path: Path
    sha256: str
    size_bytes: int
    token_count: int  # 原始 token 总数（含头、含坏 token）
    header: tuple[int, int, int] | None  # (rx, rng_nfft, nchirp)
    values: np.ndarray  # uint32 载荷（去头后、仅合法 token）
    bad_tokens: list[BadToken] = field(default_factory=list)

    @property
    def payload_count(self) -> int:
        return int(self.values.size)

    @property
    def format_name(self) -> str:
        return "header3" if self.header is not None else "headerless"


def _split_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s,;]+", text) if t]


def _parse_tokens_fast(tokens: list[str]) -> np.ndarray | None:
    """全部 token 合法且在 uint32 范围内时走 C 级快速路径，否则返回 None。"""
    if not tokens:
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.array(tokens, dtype=np.str_).astype(np.int64)
    except (ValueError, OverflowError, TypeError):
        return None
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > UINT32_MAX):
        return None
    return arr


def _parse_tokens_robust(tokens: list[str]) -> tuple[np.ndarray, list[BadToken]]:
    """逐 token 解析，记录每个非法 token 的位置与原因。"""
    values: list[int] = []
    bad: list[BadToken] = []
    for i, tok in enumerate(tokens):
        if not _INT_RE.match(tok):
            bad.append(BadToken(i, tok[:32], "non-numeric"))
            continue
        v = int(tok)
        if v < 0:
            bad.append(BadToken(i, tok[:32], "out-of-range(<0)"))
        elif v > UINT32_MAX:
            bad.append(BadToken(i, tok[:32], "out-of-range(>2^32-1)"))
        else:
            values.append(v)
    return np.asarray(values, dtype=np.int64), bad


def _looks_like_header(vals: np.ndarray) -> bool:
    """判断前三个合法值是否构成 `<rx>,<rng_nfft>,<nchirp>` 头。

    判据：rx 为很小的通道序号，且 rng_nfft / nchirp 落在工程合理区间。
    无头文件的首个 ADC 字在仓内实测中均为大 uint32（>10^6），不会误判。
    """
    if vals.size < 3:
        return False
    rx, nfft, nchirp = (int(v) for v in vals[:3])
    return (
        0 <= rx <= _HEADER_RX_MAX
        and _HEADER_NFFT_MIN <= nfft <= _HEADER_NFFT_MAX
        and _HEADER_CHIRP_MIN <= nchirp <= _HEADER_CHIRP_MAX
    )


def load_adc_file(path: str | Path) -> LoadedFile:
    """读取一个 ADC 文本文件，返回显式记录了全部异常的解析结果。"""
    path = Path(path)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    tokens = _split_tokens(text)

    arr = _parse_tokens_fast(tokens)
    if arr is not None:
        bad: list[BadToken] = []
    else:
        arr, bad = _parse_tokens_robust(tokens)

    header: tuple[int, int, int] | None = None
    payload = arr
    if _looks_like_header(arr):
        header = (int(arr[0]), int(arr[1]), int(arr[2]))
        payload = arr[3:]

    return LoadedFile(
        path=path.resolve(),
        sha256=sha,
        size_bytes=len(raw),
        token_count=len(tokens),
        header=header,
        values=payload.astype(np.uint32, copy=False),
        bad_tokens=bad,
    )


def unpack_int16(values: np.ndarray, rng_nfft: int, nchirp: int) -> np.ndarray:
    """uint32 字 -> 有符号 int16 样本，逐字高位在前（官方 adc_mem_2_real_fixed.m 语义）。

    MATLAB 参考：`reshape(col, [rng_nfft/2, nchirp])`（列主序）再转置，等价于
    numpy 按行主序直接 `reshape(nchirp, rng_nfft/2)`：第 c 个 chirp 对应载荷中
    第 c*wpc..(c+1)*wpc 个连续字。输入须已对齐到期望长度
    (rng_nfft//2)*nchirp；返回形状 (nchirp, rng_nfft)。
    """
    expected = (rng_nfft // 2) * nchirp
    if values.size != expected:
        raise ValueError(f"unpack_int16 需要 {expected} 字，实际 {values.size}")
    words = values.astype(np.uint32, copy=False).reshape(nchirp, rng_nfft // 2)
    hi = (words >> np.uint32(16)).astype(np.uint16).view(np.int16)
    lo = (words & np.uint32(0xFFFF)).astype(np.uint16).view(np.int16)
    samples = np.empty((nchirp, rng_nfft), dtype=np.int16)
    samples[:, 0::2] = hi
    samples[:, 1::2] = lo
    return samples
