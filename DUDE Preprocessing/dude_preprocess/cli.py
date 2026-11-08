"""Command line interface: ``python -m dude_preprocess <command> ...``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import urllib.request
from pathlib import Path

from . import __version__
from .build import BuildConfig, run_build
from .validate import print_report, validate

ANNOTATIONS_URL = (
    "https://zenodo.org/records/7763635/files/2023-03-23_DUDE_gt_test_PUBLIC.json?download=1"
)


def _add_build_parser(sub) -> None:
    p = sub.add_parser("build", help="preprocess DUDE into the normalized dataset")
    p.add_argument("--data-dir", default=None, help="folder with PDF/ and OCR/ (or an unpacked DUDE root)")
    p.add_argument("--out-dir", default=None, help="output folder")
    p.add_argument("--annotations", default=None, help="annotation JSON (defaults to one found in --data-dir)")
    p.add_argument("--splits", nargs="+", default=None)
    p.add_argument(
        "--stage",
        choices=["all", "process", "emit"],
        default=None,
        help="'process' = PDFs/OCR only, 'emit' = rebuild jsonl/stats from processed docs",
    )
    p.add_argument(
        "--documents",
        choices=["render", "probe", "none"],
        default=None,
        help="render=page images, probe=page sizes only, none=ignore PDFs",
    )
    p.add_argument("--dpi", type=int, default=None, help="page render resolution (DUDE OCR is 200 dpi)")
    p.add_argument("--image-format", choices=["jpeg", "png"], default=None)
    p.add_argument("--max-side", type=int, default=None, help="downscale pages longer than N px (0 = keep)")
    p.add_argument("--jpeg-quality", type=int, default=None)
    p.add_argument("--ocr-engine", default=None, help="OCR sub-folder, e.g. Amazon, Azure, Tesseract")
    p.add_argument("--ocr-format", choices=["due", "original"], default=None)
    p.add_argument(
        "--ocr-adapter",
        choices=["auto", "due", "azure", "textract", "tesseract", "generic", "none"],
        default=None,
        help="parser for the OCR dump; 'auto' detects it",
    )
    p.add_argument("--lowercase-ocr", action="store_true", default=None, help="lower-case OCR tokens (DUDE baseline style)")
    p.add_argument("--inline-ocr", action="store_true", default=None, help="also embed OCR tokens inside each question record")
    p.add_argument("--workers", type=int, default=None, help="parallel processes (0 = auto)")
    p.add_argument("--limit-docs", type=int, default=None, help="process at most N new documents per split (debug)")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", default=None, help="redo documents that are already processed")
    p.add_argument("--config", default=None, help="optional YAML file with the same keys as the flags")


def _add_validate_parser(p) -> None:
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--check-files", action="store_true", help="verify page images / OCR files exist on disk")
    p.add_argument(
        "--ocr-sample",
        type=int,
        default=0,
        help="check OCR boxes for at most N documents per split (0 = all)",
    )
    p.add_argument("--json", action="store_true", help="print the report as JSON")


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        raise SystemExit("PyYAML is required for --config: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a mapping at top level")
    return {k.replace("-", "_"): v for k, v in data.items()}


def cmd_build(args: argparse.Namespace) -> int:
    provided = {
        k: v for k, v in vars(args).items() if k not in {"command", "config"} and v is not None
    }
    values = _load_yaml(args.config) if args.config else {}
    values.update(provided)

    defaults = {f.name: f.default for f in dataclasses.fields(BuildConfig)}
    for key, default in defaults.items():
        if default is dataclasses.MISSING:
            continue
        values.setdefault(key, default)

    stage = values.pop("stage", None) or "all"
    values["splits"] = tuple(values.get("splits") or ("train", "val", "test"))
    values["max_side"] = values.get("max_side") or None
    values["workers"] = values.get("workers") or min(8, os.cpu_count() or 1)

    if not values.get("data_dir"):
        raise SystemExit("error: --data-dir is required (flag or 'data-dir' in the config file)")
    if not values.get("out_dir"):
        raise SystemExit("error: --out-dir is required (flag or 'out-dir' in the config file)")

    allowed = set(BuildConfig.__dataclass_fields__)
    values = {k: v for k, v in values.items() if k in allowed}
    cfg = BuildConfig(**values)  # type: ignore[arg-type]

    stats = run_build(cfg, stage=stage)
    if stats:
        print(json.dumps(stats.get("totals", {}), indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate(
        args.out_dir,
        check_files=args.check_files,
        ocr_sample=args.ocr_sample,
        splits=tuple(args.splits),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)
    return 0 if report.ok() else 1


def cmd_download(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to re-download)")
        return 0
    print(f"downloading {ANNOTATIONS_URL}\n     -> {out}")
    urllib.request.urlretrieve(ANNOTATIONS_URL, out)
    size = out.stat().st_size / 1e6
    print(f"done ({size:.1f} MB)")
    if size < 10:
        print("warning: file looks truncated; re-run with --force", file=sys.stderr)
        return 1
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.out_dir) / "questions" / f"{args.split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))
            print("-" * 40)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dude_preprocess",
        description="Preprocess the DUDE document-understanding dataset "
        "(documents, questions, answers, OCR, bounding boxes, page info, answer types, splits).",
    )
    parser.add_argument("--version", action="version", version=f"dude_preprocess {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_build_parser(sub)
    _add_validate_parser(sub.add_parser("validate", help="check the produced dataset"))

    d = sub.add_parser("download-annotations", help="fetch the public DUDE annotations (14 MB)")
    d.add_argument("--out", required=True, help="destination .json path")
    d.add_argument("--force", action="store_true")

    i = sub.add_parser("inspect", help="pretty-print a few processed question records")
    i.add_argument("--out-dir", required=True)
    i.add_argument("--split", default="train", choices=["train", "val", "test"])
    i.add_argument("-n", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "build": cmd_build,
        "validate": cmd_validate,
        "download-annotations": cmd_download,
        "inspect": cmd_inspect,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
