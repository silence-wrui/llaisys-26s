import argparse
import subprocess
import sys
from pathlib import Path


OPS = [
    "add",
    "argmax",
    "embedding",
    "linear",
    "rms_norm",
    "rope",
    "self_attention",
    "swiglu",
]


def run_operator_test(
    operator_name: str,
    device: str,
    profile: bool,
) -> None:
    test_dir = Path(__file__).resolve().parent

    test_file = (
        test_dir
        / "ops"
        / f"{operator_name}.py"
    )

    command = [
        sys.executable,
        str(test_file),
        "--device",
        device,
    ]

    if profile:
        command.append("--profile")

    print(
        f"\n{'=' * 60}\n"
        f"Running operator test: {operator_name}\n"
        f"{'=' * 60}",
        flush=True,
    )

    subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all LLAISYS operator tests"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "nvidia"],
        help="Device used by operator tests",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run performance benchmarks",
    )

    parser.add_argument(
        "--ops",
        nargs="+",
        choices=OPS,
        default=OPS,
        help="Only run selected operators",
    )

    args = parser.parse_args()

    for operator_name in args.ops:
        run_operator_test(
            operator_name,
            args.device,
            args.profile,
        )

    print(
        "\n\033[92m"
        "All operator tests passed!"
        "\033[0m"
    )


if __name__ == "__main__":
    main()