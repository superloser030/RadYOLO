"""RadYOLO 송신측 진입점 — 노트북(fordev / jayou_n)에서 실행.

웹캠(+레이더) 데이터를 데스크톱으로 실시간 송신.
설정: config/network.toml, config/sender.toml

사용법:
    python main_s.py 0      # 웹캠만 송신
    python main_s.py 1      # 레이더 + 웹캠 송신
"""
import argparse

from src.transmission import sender


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RadYOLO 송신측")
    parser.add_argument("mode", type=int, choices=[0, 1],
                        help="0: 웹캠만   1: 레이더 + 웹캠")
    args = parser.parse_args()

    sender.run(args.mode)
