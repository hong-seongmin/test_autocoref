# coref_automl/bus.py
from __future__ import annotations
import os
import io
import json
import time
import errno
from typing import Iterator, Optional


def _default_bus_path() -> str:
    """
    빈 문자열/미설정/공백만 있는 경우를 모두 안전하게 처리.
    기본값은 프로젝트 루트의 data 폴더
    """
    raw = os.getenv("COREF_BUS", "").strip()
    if raw:
        return raw

    # 프로젝트 루트 찾기 (coref_automl 폴더의 상위)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # data 폴더 생성 및 경로 반환
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "coref_automl_bus.ndjson")


def _ensure_file(path: str) -> None:
    """
    경로의 상위 디렉터리를 생성하고, 파일이 없으면 빈 파일을 만든다.
    """
    dirpath = os.path.dirname(path)
    if not dirpath:
        # 'bus.ndjson' 같은 상대 파일명만 온 경우 현재 디렉터리 사용
        dirpath = "."
    os.makedirs(dirpath, exist_ok=True)
    # 파일 생성(존재하면 no-op)
    with open(path, "a", encoding="utf-8") as _:
        pass


BUS_PATH: str = _default_bus_path()


class BusWriter:
    """
    여러 프로세스에서 동시에 append하는 간단한 NDJSON 버스.
    각 레코드는 한 줄짜리 JSON.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = (path or BUS_PATH).strip()
        if not self.path:
            # 최후 방어: 여기도 기본 경로로
            self.path = _default_bus_path()
        _ensure_file(self.path)

    def log(self, **kv) -> None:
        kv.setdefault("ts", time.time())
        # write-once 방식(프로세스 간 락 고민 없이 안전)
        line = json.dumps(kv, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


class BusTail:
    """
    NDJSON 파일을 tail 하여 새 레코드를 dict로 yield.
    - 파일이 아직 없어도 즉시 생성 후 tail 대기
    - 파일 truncate/rotate에 대해 파일 크기 감소를 감지하면 다시 처음부터 읽음
    """

    def __init__(self, path: Optional[str] = None, poll_sec: float = 0.2):
        self.path = (path or BUS_PATH).strip() or _default_bus_path()
        self.poll_sec = poll_sec
        _ensure_file(self.path)
        self._pos = 0  # 현재 읽기 위치

    def __iter__(self) -> Iterator[dict]:
        while True:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    # 파일 크기 체크(rotate/truncate 대응)
                    f.seek(0, io.SEEK_END)
                    end = f.tell()
                    if self._pos > end:
                        # 파일이 작아졌다면(아마도 truncate/rotate) 처음부터
                        self._pos = 0
                    f.seek(self._pos, io.SEEK_SET)

                    line = f.readline()
                    if not line:
                        # 새 줄이 아직 없음 → 잠시 대기
                        time.sleep(self.poll_sec)
                        continue
                    self._pos = f.tell()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # 깨진 줄은 스킵
                        continue
            except FileNotFoundError:
                # 누군가 파일을 지웠다면 다시 생성
                _ensure_file(self.path)
                time.sleep(self.poll_sec)
            except OSError as e:
                # 일시적 I/O 에러는 대기 후 재시도
                if e.errno in (errno.EAGAIN, errno.EINTR):
                    time.sleep(self.poll_sec)
                    continue
                raise


# 학습/튜닝/스윕 코드가 import 즉시 사용할 수 있게 기본 writer 제공
BUS = BusWriter()

