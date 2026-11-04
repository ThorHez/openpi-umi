"""Prepare one task-local, weighted raw-sample statistic before model startup.

Planning is metadata-only and imports no tensor/numeric backends. Execution
projects numeric Parquet columns, performs two finite passes, then publishes a
new immutable artifact. Existing dataset statistics are never merged or edited.
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from openpi.training.shared_norm import load_shared_norm_document, source_descriptors, validate_shared_metadata


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_files():
    root = Path(__file__).resolve().parents[1]
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in (
        "training/training_normalization.py", "training/numeric_stats.py",
        "training/shared_norm.py", "training/composed_processing.py", "shared/weighted_stats.py",
    )}


def _weights(factory):
    weights = list(factory.weights) if factory.weights is not None else [1.] * len(factory.datasets)
    alternate = getattr(factory, "norm_weights", None)
    if alternate is not None and list(alternate) != weights:
        raise ValueError("SHARED_NORM_INVALID: shared statistics must use the actual training row weights")
    return weights


def _context(config, sources, numeric_configs):
    return {"schema_version": 1, "sources": sources,
        "action_horizon": config.model.action_horizon, "model_action_dim": config.model.action_dim,
        "effective_fps": [item.effective_fps for item in numeric_configs],
        "implementation": _source_files()}


def _protect_output(output, config):
    output = Path(output).resolve()
    for factory in config.data.datasets:
        dataset = Path(factory.repo_id).resolve(strict=True)
        if output.is_relative_to(dataset):
            raise ValueError("SHARED_NORM_INVALID: automatic statistics must be outside input datasets")
        assets = getattr(factory, "assets", None)
        if assets is not None:
            base = getattr(assets, "assets_dir", None) or config.assets_dirs
            identity = getattr(assets, "asset_id", None) or factory.repo_id
            if "://" not in str(base):
                original = (Path(base) / identity).resolve()
                if output.is_relative_to(original):
                    raise ValueError("SHARED_NORM_INVALID: automatic statistics must be outside input norm assets")
    return output


def planned_training_normalization(config):
    """Freeze automatic normalization inputs without reading Parquet payloads.

    Registered source statistics remain historical references; automatic mode
    computes the actual training statistic from every raw row. Merely finding
    an existing output never changes this frozen plan.
    """
    factory = config.data
    children = getattr(factory, "datasets", None)
    if not isinstance(children, (list, tuple)) or len(children) <= 1:
        return {"mode": "per_dataset"}
    path = getattr(factory, "shared_norm_stats_path", None)
    if path is not None and getattr(factory, "use_merged_norm_stats", False):
        path = Path(path).expanduser().resolve()
        if path.name != "norm_stats.json":
            path = path / "norm_stats.json"
        return {"mode": "shared", "path": str(path)}
    if not (getattr(factory, "auto_shared_norm", False) or getattr(factory, "use_merged_norm_stats", False)):
        return {"mode": "per_dataset"}
    from openpi.training.numeric_stats import build_numeric_statistics_config, numeric_io_evidence

    weights = _weights(factory)
    sources = source_descriptors(children, weights)
    numeric = [build_numeric_statistics_config(child, config.model,
        effective_fps=getattr(child, "_norm_effective_fps", None), validate_model=False) for child in children]
    context = _context(config, sources, numeric)
    fingerprint = _digest(context)
    checkpoint = Path(config.checkpoint_dir).resolve()
    output = _protect_output(checkpoint.parent / (checkpoint.name + "-normalization") / fingerprint, config)
    return {"mode": "auto_shared", "output_path": str(output / "norm_stats.json"),
        "fingerprint": fingerprint, "method": "weighted_raw_samples", "weight_semantics": "per_sample_repeat",
        "coverage": "full", "quantile_method": "weighted_histogram", "quantile_bins": 5000,
        "action_horizon": context["action_horizon"], "model_action_dim": context["model_action_dim"],
        "sources": sources, "effective_fps": context["effective_fps"],
        "implementation": context["implementation"], "numeric_io": [numeric_io_evidence(item) for item in numeric]}


class _BatchedNumericSources:
    """Map one ordered task to a vectorized source batch, without row collation."""

    def __init__(self, sources):
        self.sources = list(sources)
        self.active_source = None

    def __getitem__(self, request):
        source_index,start,stop = request
        if self.active_source is not None and self.active_source != source_index:
            # One worker must not retain a separate block cache for every
            # source after it moves to the next dataset or the second pass.
            self.sources[self.active_source].clear_cache()
        self.active_source = source_index
        return self.sources[source_index].get_batch(range(start,stop))

    def close(self):
        for source in self.sources:
            source.clear_cache()
        self.active_source = None


class _SourceBatchSampler:
    def __init__(self, sources, batch_size):
        self.sources,self.batch_size,self.source_index = sources,batch_size,0

    def __iter__(self):
        index = self.source_index
        rows = len(self.sources[index])
        for start in range(0,rows,self.batch_size):
            yield index,start,min(start+self.batch_size,rows)

    def __len__(self):
        rows = len(self.sources[self.source_index])
        return (rows+self.batch_size-1)//self.batch_size


def _numeric_batch_identity(batch):
    # DataLoader's default no-auto-batching collate converts NumPy to Tensor;
    # keep the exact already-batched numeric arrays and their stored dtypes.
    return batch


def _numeric_worker_init(_worker_index):
    import torch
    import pyarrow
    torch.set_num_threads(1)
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)


@contextmanager
def _worker_environment():
    names = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
    before = {name:os.environ.get(name) for name in names}
    try:
        os.environ.update({name:'1' for name in names})
        yield
    finally:
        for name,value in before.items():
            if value is None:
                os.environ.pop(name,None)
            else:
                os.environ[name]=value


class NumericStatisticsLoader:
    """One bounded ordered worker pool reused across all sources and both passes.

    Source constructors validate only numeric schema and temporal indexes in
    the parent. No image column or video decoder is used. Payload projection
    and action-chunk gathering happen in vectorized batches inside workers.
    With zero workers the identical batch implementation runs inline.
    """

    def __init__(self,numeric_configs,batch_size,num_workers=0):
        from openpi.training.numeric_stats import NumericParquetDataset
        if type(batch_size) is not int or batch_size<=0 or type(num_workers) is not int or num_workers<0:
            raise ValueError('SHARED_NORM_INVALID: statistics batch size must be positive and num_workers nonnegative')
        self.batch_size,self.num_workers = batch_size,num_workers
        self.sources = [NumericParquetDataset(item) for item in numeric_configs]
        self.dataset = _BatchedNumericSources(self.sources)
        self.sampler = _SourceBatchSampler(self.sources,batch_size)
        self.loader,self.closed,self.in_source = None,False,False
        if num_workers:
            from torch.utils.data import DataLoader
            self.loader = DataLoader(self.dataset,batch_size=None,sampler=self.sampler,
                num_workers=num_workers,multiprocessing_context='spawn',persistent_workers=True,
                prefetch_factor=2,collate_fn=_numeric_batch_identity,worker_init_fn=_numeric_worker_init)

    def __enter__(self):
        return self

    def __exit__(self,*_exception):
        self.close()

    def source_batches(self,source_index):
        if self.closed or self.in_source:
            raise RuntimeError('Statistics sources must be consumed sequentially with an open loader')
        if type(source_index) is not int or not 0<=source_index<len(self.sources):
            raise IndexError(source_index)
        self.in_source=True
        complete=False
        try:
            self.sampler.source_index=source_index
            if self.loader is None:
                iterator=(self.dataset[request] for request in self.sampler)
            else:
                # Spawn imports NumPy/Torch before worker_init_fn. Set thread
                # limits while spawning, then restore the training parent.
                with _worker_environment():
                    iterator=iter(self.loader)
            for batch in iterator:
                yield batch
            complete=True
        finally:
            self.in_source=False
            if not complete:
                self.close()

    def close(self):
        if self.closed:
            return
        self.closed=True
        try:
            if self.loader is not None:
                iterator=getattr(self.loader,'_iterator',None)
                shutdown=getattr(iterator,'_shutdown_workers',None)
                if callable(shutdown):
                    shutdown()
                self.loader._iterator=None
        finally:
            self.dataset.close()


class _StatisticsProgress:
    def __init__(self,sources,num_workers,interval=5.,started=None):
        self.sources,self.num_workers,self.interval=sources,num_workers,interval
        self.last_print,self.started=0.,time.monotonic() if started is None else started

    def __call__(self,event):
        now=time.monotonic()
        if event['event'] not in {'source_start','source_done'} and now-self.last_print<self.interval:
            return
        self.last_print=now
        elapsed=max(now-self.started,1e-9)
        completed=(event['pass']-1)*(event['pass_rows_total'] or 0)+event['pass_rows_done']
        path=str(self.sources[event['source_index']].repo_id)
        name=Path(path).name or path
        phase='均值、方差与范围' if event['pass']==1 else 'q01/q99 分位数'
        if event['event']=='source_start':
            _norm_log(f"第 {event['pass']}/2 轮（{phase}），数据集 {event['source_index']+1}/{event['sources_total']}：{name}")
            _norm_log('数据集完整路径：'+path)
        total=event['rows_total']
        percent=100*event['rows_done']/total if total else 0.
        pass_total=event['pass_rows_total']
        pass_percent=100*event['pass_rows_done']/pass_total if pass_total else 0.
        state='，该数据集完成' if event['event']=='source_done' else ''
        _norm_log(f"第 {event['pass']}/2 轮（{phase}）｜数据集 {event['source_index']+1}/{event['sources_total']} {name}"
            f"｜{event['rows_done']:,}/{total:,} 行（{percent:.2f}%）"
            f"｜本轮 {event['pass_rows_done']:,}/{pass_total:,} 行（{pass_percent:.2f}%）"
            f"｜速度 {completed/elapsed:,.1f} 行/秒｜总耗时 {elapsed:.1f} 秒｜workers={self.num_workers}{state}")


def _norm_log(message):
    print('[norm_stats] '+message,flush=True)


def _ready_log(config,path,started,*,reused):
    method=_method(config)
    label={'quantile':'q01/q99（quantile）','minmax':'Min-Max（minmax）','zscore':'均值/标准差（zscore）'}.get(method,'沿用配方配置')
    _norm_log(('已复用匹配的共享统计文件：' if reused else '已保存共享统计文件：')+str(path))
    _norm_log('SHA-256：'+hashlib.sha256(path.read_bytes()).hexdigest())
    _norm_log(f"归一化方式：{label}；统计准备完成，总耗时 {time.monotonic()-started:.1f} 秒；workers={getattr(config,'num_workers',0)}。")


def _publish(path, document, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".normalization-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _method(config):
    method = getattr(config, "normalization_method", None)
    if method is None:
        values = {(getattr(child, "composition", None) or {}).get("normalization") for child in config.data.datasets}
        if len(values) == 1:
            method = values.pop()
    return method if method in {"quantile", "minmax", "zscore"} else None


def _write_receipt(config, path, document, *, fingerprint=None):
    if fingerprint is None:
        fingerprint = document.get("training_normalization", {}).get("fingerprint")
    receipt = {"schema_version": 1, "mode": "shared", "norm_path": str(path),
        "norm_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "checkpoint_asset_id": "shared",
        "normalization_method": _method(config), "shared_normalization": document["shared_normalization"],
        "statistics_widths": {key: len(values["mean"]) for key, values in document["norm_stats"].items()},
        "fingerprint": fingerprint}
    checkpoint = Path(config.checkpoint_dir).resolve()
    receipt_path = checkpoint.parent / (checkpoint.name + "-normalization.json")
    if getattr(config, "resume", False) and receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        if any(previous.get(key) != receipt[key] for key in ("norm_sha256", "normalization_method")):
            raise ValueError("SHARED_NORM_INVALID: resuming cannot change the model normalization statistic or formula")
    _publish(receipt_path, receipt, replace=True)


def prepare_training_normalization(config):
    """Return a config sharing one verified statistic, before GPU/model startup."""
    children=getattr(config.data,'datasets',None)
    enabled=(isinstance(children,(list,tuple)) and len(children)>1
        and (getattr(config.data,'auto_shared_norm',False) or getattr(config.data,'use_merged_norm_stats',False)))
    started=time.monotonic()
    if enabled:
        _norm_log('正在校验数据源、处理配置、时间索引、混采权重及共享统计来源……')
    try:
        return _prepare_training_normalization(config,started)
    except Exception:
        if enabled:
            _norm_log(f'统计准备失败，训练尚未启动；未完成的计算结果不会用于训练。已耗时 {time.monotonic()-started:.1f} 秒。')
        raise


def _prepare_training_normalization(config,started):
    plan = planned_training_normalization(config)
    if plan["mode"] == "per_dataset":
        return config
    factory = config.data
    weights = _weights(factory)
    if plan["mode"] == "shared":
        path = Path(plan["path"])
        document = load_shared_norm_document(path, factory.datasets, weights)
        if getattr(factory, "auto_shared_norm", False) and "training_normalization" in document:
            automatic = planned_training_normalization(dataclasses.replace(config,
                data=dataclasses.replace(factory, shared_norm_stats_path=None)))
            if document["training_normalization"] != {"schema_version": 1, "fingerprint": automatic["fingerprint"]}:
                raise ValueError("SHARED_NORM_INVALID: automatic statistics do not match the current horizon, FPS or execution context")
        _write_receipt(config, path, document)
        _ready_log(config,path,started,reused=True)
        return config

    from openpi.shared.weighted_stats import compute_weighted_statistics
    from openpi.training.numeric_stats import build_numeric_statistics_config

    path = Path(plan["output_path"])
    if getattr(config, "resume", False):
        checkpoint = Path(config.checkpoint_dir).resolve()
        receipt_path = checkpoint.parent / (checkpoint.name + "-normalization.json")
        if receipt_path.is_file():
            previous = json.loads(receipt_path.read_text())
            if (previous.get("fingerprint") != plan["fingerprint"]
                    or previous.get("normalization_method") != _method(config)):
                raise ValueError("SHARED_NORM_INVALID: resuming cannot change data, weights, processing or normalization formula")
    reused=path.is_file()
    if reused:
        document = load_shared_norm_document(path, factory.datasets, weights)
        if document.get("training_normalization") != {"schema_version": 1, "fingerprint": plan["fingerprint"]}:
            raise ValueError("SHARED_NORM_INVALID: existing automatic statistics have a conflicting execution context")
        _norm_log('数据、处理方式与权重匹配，复用已完整计算的统计，不重复扫描原始样本。')
    else:
        if path.exists() or path.is_symlink():
            raise ValueError("SHARED_NORM_INVALID: statistics destination is not a regular file")
        batch_size = config.batch_size
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("SHARED_NORM_INVALID: statistics batch size must be a positive integer")
        numeric = [build_numeric_statistics_config(child, config.model,
            effective_fps=getattr(child, "_norm_effective_fps", None), validate_model=False) for child in factory.datasets]
        workers=getattr(config,'num_workers',0)
        progress=_StatisticsProgress(numeric,workers,started=started)
        _norm_log(f'开始完整两遍统计：{len(numeric)} 个数据集，共 {sum(item.total_frames for item in numeric):,} 行，workers={workers}；只读取数值字段，不解码图像。')
        with NumericStatisticsLoader(numeric,batch_size,workers) as loader:
            values, coverage = compute_weighted_statistics(
                [lambda index=index: loader.source_batches(index) for index in range(len(numeric))],weights,
                source_rows=[item.total_frames for item in numeric],progress=progress)
        if planned_training_normalization(config) != plan:
            raise ValueError("SHARED_NORM_INVALID: data or numeric processing changed during automatic statistics")
        document = {"norm_stats": {key: {**{name: array.tolist() for name, array in stats.items()},
            "quantile_method": "q01_q99"} for key, stats in values.items()}, "shared_normalization": {
                "schema_version": 1, "method": "weighted_raw_samples", "weight_semantics": "per_sample_repeat",
                "coverage": "full", "sources": [{**source, **count} for source, count in
                    zip(plan["sources"], coverage, strict=True)],
                "quantile_method": "weighted_histogram", "quantile_bins": 5000},
            "training_normalization": {"schema_version": 1, "fingerprint": plan["fingerprint"]}}
        validate_shared_metadata(document, factory.datasets, weights)
        try:
            _publish(path, document)
        except FileExistsError:
            existing = load_shared_norm_document(path, factory.datasets, weights)
            if existing != document:
                raise ValueError("SHARED_NORM_INVALID: another statistics writer published conflicting values") from None
    _write_receipt(config, path, document, fingerprint=plan["fingerprint"])
    _ready_log(config,path,started,reused=reused)
    return dataclasses.replace(config, data=dataclasses.replace(factory,
        use_merged_norm_stats=True, auto_shared_norm=True, shared_norm_stats_path=str(path)))
