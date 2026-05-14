"""Convenience entrypoint for the full MLFoot pipeline."""

from __future__ import annotations

from src.pipelines.full_pipeline import main as pipeline_main


if __name__ == "__main__":
    pipeline_main()
