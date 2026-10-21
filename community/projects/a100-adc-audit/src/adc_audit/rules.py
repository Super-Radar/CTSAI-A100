"""审计规则引擎：格式识别 / 长度 / 数值合法性 / 信号质量 / 批次一致性。

设计原则：与现有校验器（a100-matlab-adc-pipeline-validation）的"静默修补"
不同，本工具把每一次修补（丢弃多余值、补零）都变成一条显式报告，
写明丢的是哪个位置、值是多少、依据什么规则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import SensorConfig
from .loader import LoadedFile, load_adc_file, unpack_int16

# ---------------------------------------------------------------------------
# 状态与规则定义
# ---------------------------------------------------------------------------

_STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def worst_status(statuses) -> str:
    """取最差状态。"""
    return max(statuses, key=lambda s: _STATUS_ORDER.get(s, 0))


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    title: str
    description: str
    level: str  # 触发时的最高级别（错误 / 警告 / 提示）


RULES: dict[str, RuleSpec] = {
    spec.rule_id: spec
    for spec in (
        RuleSpec(
            "FMT-01",
            "格式识别与三字段头交叉核对",
            "识别无头/三字段头布局；若带 `<rx>,<rng_nfft>,<nchirp>` 头，"
            "头字段须与所用配置的 rng_nfft/nchirp 一致；文件名 f0/f1 标记须与配置 profile 一致。",
            "错误（头字段或 profile 不一致）/ 警告（无法从文件名判断 profile）",
        ),
        RuleSpec(
            "LEN-01",
            "载荷长度审计（显式修补报告）",
            "实际合法载荷值数 vs 配置期望 (rng_nfft/2*nchirp)；多余/缺失值的数量、"
            "位置（首/尾）与数值全部写明；为继续分析所做的截断/补零修复显式输出。"
            "尾部多余零值=警告；多余非零值或缺失值=错误。",
            "错误（多余非零值、任何缺失）/ 警告（尾部多余零值）",
        ),
        RuleSpec(
            "VAL-01",
            "数值合法性",
            "所有 token 必须是可解析的 uint32（0..2^32-1）；非数值 token 与越界值"
            "（负数 / >2^32-1）逐个列出位置与原值。",
            "错误（存在任何非法 token）",
        ),
        RuleSpec(
            "SIG-01",
            "信号质量统计（解包后 int16 样本）",
            "按官方 adc_mem_2_real_fixed.m 语义解包后统计：饱和计数（±32767/-32768）、"
            "最长常值游程、全零 chirp 计数、DC 偏置与 RMS。阈值：饱和率 ≥1e-3 警告、"
            "≥1e-2 错误；任何全零 chirp 警告（全文件为零错误）；最长游程 ≥ rng_nfft（整 chirp 常值）警告。",
            "错误（饱和率 ≥1%、全零文件）/ 警告（饱和率 ≥0.1%、全零 chirp、整 chirp 常值）",
        ),
        RuleSpec(
            "BAT-01",
            "跨批次 sha256 去重",
            "对全部审计文件按 sha256 分组；同一内容出现在多个批次（目录/波形组）"
            "时报『重复批次』，列出全部所在位置（不同目录重复共享=数据管理提示）。",
            "警告（跨批次重复）",
        ),
        RuleSpec(
            "BAT-02",
            "同批通道一致性",
            "按（目录, profile）分批：文件名 Rx 序号应构成 0..N-1 的完整集合；"
            "三字段头中的 rx 序号须与文件名一致；同批不同 Rx 文件内容不得完全相同。",
            "错误（Rx 序号重复/头与文件名不符/同批内容相同）/ 警告（缺通道）",
        ),
    )
}

#: 饱和率阈值（占解包样本比例）
SAT_WARN_RATIO = 1e-3
SAT_FAIL_RATIO = 1e-2

NUM_RX_DEFAULT = 4


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """一条规则结论（针对单文件或整批）。"""

    rule_id: str
    status: str  # pass / warn / fail
    message: str
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class QualityStats:
    """解包后 int16 样本的质量统计（仅在真实载荷上计算，不含修复补零）。"""

    n_real_samples: int = 0
    sat_pos: int = 0  # == +32767
    sat_neg: int = 0  # == -32768
    sat_ratio: float = 0.0
    longest_run: int = 0
    longest_run_value: int = 0
    longest_run_chirp: int = -1
    longest_run_start: int = -1
    zero_chirps: int = 0
    n_real_chirps: int = 0
    partial_chirp: bool = False
    dc: float = 0.0
    rms: float = 0.0
    max_abs: int = 0

    def as_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class FileAudit:
    """单文件审计结果。"""

    loaded: LoadedFile
    repo_rel_path: str
    short_label: str
    group_label: str
    filename_profile: int | None
    filename_rx: int | None
    config: SensorConfig | None
    findings: list[Finding] = field(default_factory=list)
    quality: QualityStats | None = None
    repair: str = ""  # 为继续分析所做的显式修复说明

    @property
    def status(self) -> str:
        return worst_status([f.status for f in self.findings]) if self.findings else "pass"

    def finding(self, rule_id: str) -> Finding | None:
        for f in self.findings:
            if f.rule_id == rule_id:
                return f
        return None

    def as_dict(self) -> dict:
        return {
            "path": self.repo_rel_path,
            "label": self.short_label,
            "group": self.group_label,
            "sha256": self.loaded.sha256,
            "size_bytes": self.loaded.size_bytes,
            "format": self.loaded.format_name,
            "header": list(self.loaded.header) if self.loaded.header else None,
            "token_count": self.loaded.token_count,
            "payload_count": self.loaded.payload_count,
            "expected_words": self.config.expected_words if self.config else None,
            "filename_profile": self.filename_profile,
            "filename_rx": self.filename_rx,
            "config_profile": self.config.profile if self.config else None,
            "repair": self.repair,
            "status": self.status,
            "findings": [f.as_dict() for f in self.findings],
            "quality": self.quality.as_dict() if self.quality else None,
        }


@dataclass
class AuditReport:
    files: list[FileAudit] = field(default_factory=list)
    batch_findings: list[Finding] = field(default_factory=list)
    data_dirs: list[str] = field(default_factory=list)
    configs: list[SensorConfig] = field(default_factory=list)
    num_rx: int = NUM_RX_DEFAULT
    repo_root: Path | None = None  # 报告输出用于把绝对路径显示为仓库相对路径

    @property
    def n_fail(self) -> int:
        return sum(1 for f in self.files if f.status == "fail") + sum(
            1 for f in self.batch_findings if f.status == "fail"
        )

    @property
    def n_warn(self) -> int:
        return sum(1 for f in self.files if f.status == "warn") + sum(
            1 for f in self.batch_findings if f.status == "warn"
        )

    def rule_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {
            rid: {"pass": 0, "warn": 0, "fail": 0} for rid in RULES
        }
        for fa in self.files:
            for f in fa.findings:
                counts[f.rule_id][f.status] += 1
        for f in self.batch_findings:
            counts[f.rule_id][f.status] += 1
        return counts

    def exit_code(self, strict: bool = False) -> int:
        if self.n_fail > 0:
            return 1
        if strict and self.n_warn > 0:
            return 1
        return 0


# ---------------------------------------------------------------------------
# 文件名推断
# ---------------------------------------------------------------------------

_PROFILE_PATTERNS = (
    re.compile(r"[_.]?Pf([01])(?=[_.])", re.IGNORECASE),
    re.compile(r"[_.]f([01])_\d+", re.IGNORECASE),
    re.compile(r"[_.]f([01])(?=[_.])", re.IGNORECASE),
)
_RX_PATTERNS = (
    re.compile(r"_Rx(\d+)", re.IGNORECASE),
    re.compile(r"[_.]f[01]_(\d+)_", re.IGNORECASE),
    re.compile(r"[_.][Nn][Rr](\d+)_"),
)


def detect_profile(path: Path) -> int | None:
    """从文件名推断波形 profile（0=远波 f0，1=近波 f1）。"""
    name = path.name
    for pat in _PROFILE_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(1))
    if "近波" in name:
        return 1
    if "远波" in name:
        return 0
    return None


def detect_rx(path: Path) -> int | None:
    """从文件名推断接收通道序号。"""
    name = path.name
    for pat in _RX_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# 单文件规则
# ---------------------------------------------------------------------------


def audit_file(
    loaded: LoadedFile,
    configs: dict[int, SensorConfig],
    repo_root: Path | None,
) -> FileAudit:
    """对一个已解析文件执行 FMT-01 / LEN-01 / VAL-01 / SIG-01。"""
    path = loaded.path
    filename_profile = detect_profile(path)
    filename_rx = detect_rx(path)

    # 配置匹配
    config: SensorConfig | None = None
    fmt_notes: list[str] = []
    fmt_status = "pass"
    if filename_profile is not None and filename_profile in configs:
        config = configs[filename_profile]
    elif len(configs) == 1:
        config = next(iter(configs.values()))
        if filename_profile is None:
            fmt_status = "warn"
            fmt_notes.append("文件名未含 f0/f1/Pf0/Pf1/近波/远波 标记，使用唯一提供的配置")
        elif config.profile != filename_profile:
            # 文件名携带明确的 f0/f1 标记但与唯一配置不符：判为失败，
            # 且不再用错误的配置继续审计长度/信号（避免错误数据被判为可用）。
            fmt_status = "fail"
            fmt_notes.append(
                f"文件名 profile=f{filename_profile} 与所给配置 init{config.profile} 明确不匹配，"
                f"判为失败且不使用该配置审计该文件"
            )
            config = None
    else:
        fmt_status = "fail"
        fmt_notes.append(
            f"无法匹配配置：文件名 profile={filename_profile}，可用 profile={sorted(configs)}"
        )

    rel = _display_path(path, repo_root)
    fa = FileAudit(
        loaded=loaded,
        repo_rel_path=rel,
        short_label=_short_label(path, filename_profile, filename_rx),
        # 批次标识用父目录的完整显示路径（仓内为仓库相对路径），
        # 不同采集路径即使末级目录同名也不会被合并为同一批次
        group_label=(
            f"{_display_path(path.parent, repo_root)}"
            f"/f{filename_profile if filename_profile is not None else '?'}"
        ),
        filename_profile=filename_profile,
        filename_rx=filename_rx,
        config=config,
    )

    # ---- FMT-01 格式识别与头交叉核对 ----
    if config is not None:
        if loaded.header is not None:
            h_rx, h_nfft, h_nchirp = loaded.header
            if h_nfft == config.rng_nfft and h_nchirp == config.nchirp:
                fmt_notes.append(
                    f"三字段头 rx={h_rx}, rng_nfft={h_nfft}, nchirp={h_nchirp} 与配置 init{config.profile} 一致"
                )
            else:
                fmt_status = "fail"
                fmt_notes.append(
                    f"三字段头 rng_nfft={h_nfft}, nchirp={h_nchirp} 与配置 init{config.profile} "
                    f"(rng_nfft={config.rng_nfft}, nchirp={config.nchirp}) 不一致"
                )
        else:
            fmt_notes.append(
                f"无头布局；长度依据配置 init{config.profile} "
                f"(rng_nfft={config.rng_nfft}, nchirp={config.nchirp})"
            )
    fa.findings.append(
        Finding("FMT-01", fmt_status, "；".join(fmt_notes), {"format": loaded.format_name})
    )

    # ---- VAL-01 数值合法性 ----
    if loaded.bad_tokens:
        shown = ", ".join(f"#{b.index}='{b.token}'({b.reason})" for b in loaded.bad_tokens[:10])
        more = f" …等共 {len(loaded.bad_tokens)} 个" if len(loaded.bad_tokens) > 10 else ""
        fa.findings.append(
            Finding(
                "VAL-01",
                "fail",
                f"发现 {len(loaded.bad_tokens)} 个非法 token: {shown}{more}",
                {"bad_tokens": [b.as_dict() for b in loaded.bad_tokens[:50]],
                 "total_bad": len(loaded.bad_tokens)},
            )
        )
    else:
        fa.findings.append(
            Finding("VAL-01", "pass", f"全部 {loaded.token_count} 个 token 均为合法 uint32")
        )

    # ---- LEN-01 长度审计（显式修补） ----
    if config is None:
        fa.findings.append(
            Finding("LEN-01", "fail", "无可用配置，无法审计长度", {})
        )
        return fa

    expected = config.expected_words
    actual = loaded.payload_count
    if actual == expected:
        fa.findings.append(
            Finding(
                "LEN-01",
                "pass",
                f"载荷 {actual} 字 = 期望 {expected} (rng_nfft/2*nchirp = "
                f"{config.rng_nfft}//2*{config.nchirp})，无需修补",
            )
        )
        aligned = loaded.values
    elif actual > expected:
        extra = actual - expected
        tail = loaded.values[expected:]
        tail_zero = bool(np.all(tail == 0))
        head = loaded.values[:extra]
        head_zero = bool(np.all(head == 0))
        preview = ", ".join(str(int(v)) for v in tail[:8])
        if tail_zero:
            status = "warn"
            action = (
                f"尾部多余 {extra} 个零值（载荷索引 {expected}..{actual - 1}，值全为 0）→ "
                f"截断修复：丢弃这些尾零后继续分析"
            )
            if head_zero and not np.all(loaded.values[:expected] == 0):
                action += "；注意首部同数量值亦为 0，但首部零属有效样本，不丢弃"
        else:
            status = "fail"
            action = (
                f"尾部多余 {extra} 个非零值（载荷索引 {expected}..{actual - 1}，前 8 个值: {preview}）→ "
                f"为继续分析按官方截断语义丢弃，但非零数据被丢弃判为失败"
            )
        fa.repair = action
        fa.findings.append(
            Finding(
                "LEN-01",
                status,
                f"载荷 {actual} 字 > 期望 {expected}，多余 {extra} 字。{action}",
                {
                    "expected": expected,
                    "actual": actual,
                    "extra": extra,
                    "position": "tail",
                    "extra_values_preview": [int(v) for v in tail[:16]],
                    "extra_all_zero": tail_zero,
                    "repair": "truncate-tail",
                },
            )
        )
        aligned = loaded.values[:expected]
    else:
        missing = expected - actual
        action = (
            f"缺失 {missing} 字（实际 {actual} < 期望 {expected}；缺失位置未知，无参考批次）→ "
            f"补零修复：末尾补 {missing} 个零后继续分析（与官方 load_adc_data.m 的 pad 语义一致）；"
            f"缺失即数据丢失，判为失败"
        )
        fa.repair = action
        fa.findings.append(
            Finding(
                "LEN-01",
                "fail",
                action,
                {
                    "expected": expected,
                    "actual": actual,
                    "missing": missing,
                    "position": "unknown",
                    "repair": "zero-pad-tail",
                },
            )
        )
        aligned = np.concatenate([loaded.values, np.zeros(missing, dtype=np.uint32)])

    # ---- SIG-01 信号质量统计（仅真实载荷，不含修复补零） ----
    fa.quality = compute_quality(aligned, min(loaded.payload_count, expected), config)
    q = fa.quality
    sub: list[tuple[str, str, str]] = []  # (name, status, text)
    if q.sat_ratio >= SAT_FAIL_RATIO:
        sub.append(("sat", "fail", f"饱和率 {q.sat_ratio:.4%} (≥{SAT_FAIL_RATIO:.0%})"))
    elif q.sat_ratio >= SAT_WARN_RATIO:
        sub.append(("sat", "warn", f"饱和率 {q.sat_ratio:.4%} (≥{SAT_WARN_RATIO:.1%})"))
    else:
        sub.append(("sat", "pass", f"饱和率 {q.sat_ratio:.6%}"))
    if q.n_real_chirps > 0 and q.zero_chirps == q.n_real_chirps:
        sub.append(("zero", "fail", f"全部 {q.zero_chirps} 个真实 chirp 均为全零"))
    elif q.zero_chirps > 0:
        sub.append(("zero", "warn", f"{q.zero_chirps}/{q.n_real_chirps} 个真实 chirp 全零"))
    else:
        sub.append(("zero", "pass", "无全零 chirp"))
    if q.longest_run >= config.rng_nfft:
        sub.append(
            (
                "run",
                "warn",
                f"最长常值游程 {q.longest_run} ≥ rng_nfft {config.rng_nfft}"
                f"（chirp {q.longest_run_chirp} 样本 {q.longest_run_start} 起，值 {q.longest_run_value}）",
            )
        )
    else:
        sub.append(("run", "pass", f"最长常值游程 {q.longest_run}"))
    sig_status = worst_status([s for _, s, _ in sub])
    msg = "；".join(t for _, _, t in sub)
    msg += f"；DC={q.dc:.2f}, RMS={q.rms:.2f}, max|x|={q.max_abs}"
    fa.findings.append(
        Finding("SIG-01", sig_status, msg, {"checks": {k: s for k, s, _ in sub}})
    )
    return fa


def compute_quality(
    aligned_words: np.ndarray, n_real_words: int, config: SensorConfig
) -> QualityStats:
    """对已对齐（截断/补零后）的字流统计质量，仅前 n_real_words 个真实字参与。"""
    samples = unpack_int16(aligned_words, config.rng_nfft, config.nchirp)
    flat = samples.reshape(-1)
    n_real_samples = min(2 * n_real_words, flat.size)
    real = flat[:n_real_samples].astype(np.int64)

    q = QualityStats(n_real_samples=int(n_real_samples))
    q.sat_pos = int(np.count_nonzero(real == 32767))
    q.sat_neg = int(np.count_nonzero(real == -32768))
    q.sat_ratio = (q.sat_pos + q.sat_neg) / n_real_samples if n_real_samples else 0.0
    q.dc = float(real.mean()) if n_real_samples else 0.0
    q.rms = float(np.sqrt((real.astype(np.float64) ** 2).mean())) if n_real_samples else 0.0
    q.max_abs = int(np.abs(real).max()) if n_real_samples else 0

    words_per_chirp = config.rng_nfft // 2
    n_real_chirps = n_real_words // words_per_chirp
    q.n_real_chirps = int(n_real_chirps)
    q.partial_chirp = n_real_words % words_per_chirp != 0
    q.zero_chirps = int(np.count_nonzero(np.all(samples[:n_real_chirps] == 0, axis=1)))

    best_len, best_chirp, best_start, best_val = 0, -1, -1, 0
    for c in range(n_real_chirps):
        row = samples[c]
        change = np.flatnonzero(row[1:] != row[:-1]) + 1
        bounds = np.concatenate(([0], change, [row.size]))
        lengths = np.diff(bounds)
        k = int(lengths.argmax())
        if int(lengths[k]) > best_len:
            best_len, best_chirp, best_start, best_val = (
                int(lengths[k]),
                c,
                int(bounds[k]),
                int(row[bounds[k]]),
            )
    q.longest_run, q.longest_run_chirp, q.longest_run_start, q.longest_run_value = (
        best_len,
        best_chirp,
        best_start,
        best_val,
    )
    return q


# ---------------------------------------------------------------------------
# 批次级规则
# ---------------------------------------------------------------------------


def audit_batches(files: list[FileAudit], num_rx: int) -> list[Finding]:
    """BAT-01 跨批次去重 + BAT-02 同批通道一致性。"""
    findings: list[Finding] = []

    # BAT-01: sha256 跨批次分组
    by_sha: dict[str, list[FileAudit]] = {}
    for fa in files:
        by_sha.setdefault(fa.loaded.sha256, []).append(fa)
    for sha, members in sorted(by_sha.items()):
        groups = sorted({m.group_label for m in members})
        if len(members) > 1 and len(groups) > 1:
            listing = "; ".join(f"{m.short_label} @ {m.group_label}" for m in members)
            findings.append(
                Finding(
                    "BAT-01",
                    "warn",
                    f"重复批次：sha256={sha[:12]}… 的同一内容出现在 {len(groups)} 个批次 "
                    f"({', '.join(groups)})：{listing}",
                    {"sha256": sha, "groups": groups, "files": [m.repo_rel_path for m in members]},
                )
            )
        elif len(members) > 1:
            # 同一批次内的重复交给 BAT-02（通道内容一致性）
            pass

    # BAT-02: 按 (目录, profile) 分批
    by_group: dict[str, list[FileAudit]] = {}
    for fa in files:
        by_group.setdefault(fa.group_label, []).append(fa)
    for glabel, members in sorted(by_group.items()):
        problems: list[str] = []
        warns: list[str] = []

        # Rx 序号完整性
        rx_of: list[tuple[int | None, FileAudit]] = [(f.filename_rx, f) for f in members]
        known = [(rx, f) for rx, f in rx_of if rx is not None]
        seen: dict[int, FileAudit] = {}
        for rx, f in known:
            if rx in seen:
                problems.append(
                    f"Rx 序号重复: {seen[rx].loaded.path.name} 与 {f.loaded.path.name} 均为 Rx{rx}"
                )
            else:
                seen[rx] = f
        for f in members:
            if f.filename_rx is None:
                warns.append(f"无法从文件名解析 Rx 序号: {f.loaded.path.name}")
        missing = sorted(set(range(num_rx)) - set(seen))
        if missing and not problems:
            warns.append(f"批次缺通道: 缺 Rx{', Rx'.join(str(m) for m in missing)}")

        # 头 rx vs 文件名 rx
        for f in members:
            if f.loaded.header is not None and f.filename_rx is not None:
                h_rx = f.loaded.header[0]
                if h_rx != f.filename_rx:
                    problems.append(
                        f"{f.loaded.path.name}: 头部 rx={h_rx} 与文件名 Rx{f.filename_rx} 不一致"
                    )

        # 同批不同 Rx 文件内容不得完全相同
        sha_map: dict[str, FileAudit] = {}
        for f in members:
            first = sha_map.get(f.loaded.sha256)
            if first is not None and first.filename_rx != f.filename_rx:
                problems.append(
                    f"同批不同 Rx 内容完全相同: {first.loaded.path.name} 与 {f.loaded.path.name} "
                    f"(sha256={f.loaded.sha256[:12]}…)"
                )
            else:
                sha_map.setdefault(f.loaded.sha256, f)

        status = "fail" if problems else ("warn" if warns else "pass")
        msg_parts = [f"批次 {glabel}: {len(members)} 个文件"]
        if problems:
            msg_parts.append("问题: " + "；".join(problems))
        if warns:
            msg_parts.append("提示: " + "；".join(warns))
        if not problems and not warns:
            msg_parts.append(f"Rx 序号完整 (0..{num_rx - 1}) 且内容互不相同")
        findings.append(
            Finding(
                "BAT-02",
                status,
                "。".join(msg_parts),
                {
                    "group": glabel,
                    "members": [f.repo_rel_path for f in members],
                    "rx_indices": [f.filename_rx for f in members],
                    "problems": problems,
                    "warnings": warns,
                },
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def find_repo_root(start: Path) -> Path | None:
    """向上查找 git 仓库根，用于把数据路径显示为仓库相对路径。"""
    cur = Path(start).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".git").exists():
            return cand
    return None


def iter_adc_files(data_dir: str | Path) -> list[Path]:
    """列出目录下的 ADC 文本文件（文件名以 adc 开头，排序保证确定性）。"""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    return sorted(
        p for p in data_dir.glob("*.txt") if p.name.lower().startswith("adc")
    )


def audit(
    data_dirs: list[str | Path],
    configs: list[SensorConfig],
    num_rx: int = NUM_RX_DEFAULT,
    repo_root: Path | None = None,
) -> AuditReport:
    """审计多个数据目录下的全部 ADC 文件。"""
    if repo_root is None:
        repo_root = find_repo_root(Path(data_dirs[0])) if data_dirs else None
    by_profile = {c.profile: c for c in configs}
    report = AuditReport(
        data_dirs=[str(Path(d).resolve()) for d in data_dirs],
        configs=configs,
        num_rx=num_rx,
        repo_root=repo_root,
    )
    for d in data_dirs:
        paths = iter_adc_files(d)
        if not paths:
            # 目录存在但没有任何 ADC 文件属于输入错误，必须显式报错，
            # 不能"审计 0 个文件"后返回成功
            raise FileNotFoundError(f"数据目录中未找到任何 ADC 文件 (adc*.txt): {d}")
        for p in paths:
            loaded = load_adc_file(p)
            report.files.append(audit_file(loaded, by_profile, repo_root))
    report.batch_findings = audit_batches(report.files, num_rx)
    return report


# ---------------------------------------------------------------------------
# 显示辅助
# ---------------------------------------------------------------------------

_DIR_LABELS = {
    "示例adc数据和结果": "A",
    "data": "B",
}


def _display_path(path: Path, repo_root: Path | None) -> str:
    try:
        if repo_root is not None:
            return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        pass
    return str(path)


def _short_label(path: Path, profile: int | None, rx: int | None) -> str:
    dir_key = _DIR_LABELS.get(path.parent.name, path.parent.name[:8])
    wave = {0: "far", 1: "near"}.get(profile, "?")
    rx_txt = f"Rx{rx}" if rx is not None else "Rx?"
    return f"{dir_key}·{wave}·{rx_txt}"
