"""Command-line interface: ``python -m rag_eval run ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runner import load_dataset, run_evaluation, summary_table, write_reports
from .systems import DocQARagSystem, HttpRagSystem, StubSystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_eval",
        description="Evaluate a RAG system against a golden QA dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the golden dataset through a system.")
    run.add_argument("--dataset", required=True, help="Path to the golden dataset JSON.")
    run.add_argument(
        "--system",
        required=True,
        choices=["doc-qa-rag", "stub", "http"],
        help="Which system adapter to evaluate.",
    )
    run.add_argument("--endpoint", default=None, help="HTTP endpoint for --system http.")
    run.add_argument(
        "--doc-qa-rag-path",
        default=None,
        help="Path to the document-qa-rag checkout (default: env RAG_EVAL_DOC_QA_RAG_PATH or the local copy).",
    )
    run.add_argument("--out", default="results", help="Output directory for reports (default: results).")
    run.add_argument("--k", type=int, default=5, help="Retrieval depth for scoring (default: 5).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        dataset = load_dataset(args.dataset)
        if args.system == "doc-qa-rag":
            system = DocQARagSystem(doc_qa_rag_path=args.doc_qa_rag_path)
        elif args.system == "http":
            if not args.endpoint:
                print("error: --system http needs --endpoint URL", file=sys.stderr)
                return 2
            system = HttpRagSystem(args.endpoint)
        else:
            system = StubSystem()
        report = run_evaluation(system, dataset, k=args.k)
        print(summary_table(report))
        json_path, md_path = write_reports(report, args.out)
        print(f"\nJSON report: {json_path}")
        print(f"Markdown report: {md_path}")
        return 0
    return 2


def console_main() -> None:
    """Entry point for the ``rag-eval`` console script.

    ``main()`` returns an int exit code; the console script must turn it
    into a real process exit, so this wrapper raises SystemExit with it
    instead of silently returning 0.
    """
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
