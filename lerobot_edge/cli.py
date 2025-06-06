"""Command line interface for the edge agent."""
import argparse

from . import connect_arm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LeRobot edge agent")
    parser.add_argument("run", action="store_true")
    parser.add_argument("--model-path", required=False)
    parser.add_argument("--token", required=False)
    args = parser.parse_args()

    if args.run:
        print("Starting edge agent")
        if args.model_path:
            print(f"Using model at {args.model_path}")
        if args.token:
            print("Token provided")
        # Placeholder for hardware connection guidance
        connect_arm(0, 0)


if __name__ == "__main__":
    main()
