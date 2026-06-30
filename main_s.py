import argparse

from src.transmission import sender


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RadYOLO 송신측")
    parser.add_argument("mode", type=int, choices=[0, 1],
                        help="0: 웹캠만   1: 레이더 + 웹캠")
    args = parser.parse_args()

    sender.run(args.mode)
