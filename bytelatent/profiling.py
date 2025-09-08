# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
import logging
import os
from pathlib import Path

import torch.distributed
import wandb
from pydantic import BaseModel
from torch.profiler.profiler import profile

from bytelatent.distributed import get_is_master


class ProfilerArgs(BaseModel):
    run: bool = False
    trace_folder: str = "profiling"
    mem_warmup: int = 100
    mem_steps: int = 2
    profile_warmup: int = 102
    profile_steps: int = 2


logger = logging.getLogger()


def perfetto_to_html(json_file, html_file):
    import gzip
    import string

    import viztracer

    root = os.path.dirname(viztracer.__file__)
    sub = {}
    json_file = gzip.open(json_file) if ".gz" in str(json_file) else open(json_file)
    with open(
        os.path.join(root, "html/trace_viewer_embedder.html"), encoding="utf-8"
    ) as f:
        tmpl = f.read()
    with open(os.path.join(root, "html/trace_viewer_full.html"), encoding="utf-8") as f:
        sub["trace_viewer_full"] = f.read()
    with json_file as j:
        content = j.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        sub["json_data"] = content.replace("</script>", "<\\/script>")  # type: ignore
    with open(html_file, "w+", encoding="utf-8") as output_file:
        output_file.write(string.Template(tmpl).substitute(sub))


class NoopProfiler:
    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = output_dir

    def step(self) -> None:
        return None


def _log_profile_artifacts(trace_dir: str):
    try:
        if get_is_master() and wandb.run is not None:
            # Best-effort logging if torch profiler emitted traces
            candidates = list(
                Path(trace_dir).glob("profile_CPU_CUDA*/*.pt.trace.json*")
            )
            if candidates:
                html_path = str(candidates[0]).replace(".json", ".html")
                perfetto_to_html(candidates[0], html_path)
                wandb.log({"profile_trace": wandb.Html(html_path)})
    except Exception:
        pass


@contextlib.contextmanager
def maybe_run_profiler(dump_dir, module, config: ProfilerArgs):
    # get user defined profiler settings

    if config.run:
        trace_dir = os.path.join(dump_dir, config.trace_folder)

        logger.info("Profiling active.  Traces will be saved at %s", trace_dir)

        if get_is_master() and not os.path.exists(trace_dir):
            os.makedirs(trace_dir)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Fallback minimal profiler: return a noop but still log path so callers can step()
        profiler = NoopProfiler(output_dir=trace_dir)
        try:
            yield profiler
        finally:
            _log_profile_artifacts(trace_dir)

    else:
        yield None
