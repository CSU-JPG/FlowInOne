#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from threading import Thread
from queue import Queue
import itertools
import webdataset as wds
from PIL import Image
import io

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return None
    return Path(v).expanduser()


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    return int(v)


def parse_args(argv: list[str] | None = None):
    """路径等配置优先来自命令行，未提供时使用环境变量（见各参数 help）。"""
    p = argparse.ArgumentParser(
        description="将含 input/output 的编辑数据对打成 WebDataset tar 分片。"
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="源数据根目录（含若干 input/ 与 output/ 子目录）。可用环境变量 UNIFIED_TO_TARS_ROOT。",
    )
    p.add_argument(
        "--tar-dir",
        type=Path,
        default=None,
        help="输出 tar 目录。可用环境变量 UNIFIED_TO_TARS_TAR_DIR。",
    )
    p.add_argument(
        "--samples-per-shard",
        type=int,
        default=None,
        help="每个 tar 内样本数。默认 600；可用环境变量 UNIFIED_TO_TARS_SAMPLES_PER_SHARD。",
    )
    p.add_argument(
        "--data-type",
        choices=["t2i"],
        default=None,
        help="写入样本中的类型字段；不传则不写。可用环境变量 UNIFIED_TO_TARS_DATA_TYPE。",
    )
    p.add_argument(
        "--key-prefix",
        default=None,
        help="__key__ 前缀；传空字符串表示无前缀。默认 t2i；可用环境变量 UNIFIED_TO_TARS_KEY_PREFIX。",
    )
    p.add_argument(
        "--read-workers",
        type=int,
        default=None,
        help="读图进程数。默认 min(CPU, 40)；可用环境变量 UNIFIED_TO_TARS_READ_WORKERS。",
    )
    args = p.parse_args(argv)

    root = args.root or _env_path("UNIFIED_TO_TARS_ROOT")
    tar_dir = args.tar_dir or _env_path("UNIFIED_TO_TARS_TAR_DIR")
    if root is None:
        p.error("必须指定 --root 或设置环境变量 UNIFIED_TO_TARS_ROOT")
    if tar_dir is None:
        p.error("必须指定 --tar-dir 或设置环境变量 UNIFIED_TO_TARS_TAR_DIR")

    if args.samples_per_shard is None:
        args.samples_per_shard = _env_int("UNIFIED_TO_TARS_SAMPLES_PER_SHARD", 600)
    if args.samples_per_shard < 1:
        p.error("--samples-per-shard 须为正整数")

    if args.data_type is None:
        dt = (os.environ.get("UNIFIED_TO_TARS_DATA_TYPE") or "").strip()
        if dt:
            if dt not in ("t2i", "c2i"):
                p.error("UNIFIED_TO_TARS_DATA_TYPE 须为 t2i 或 c2i")
            args.data_type = dt

    if args.key_prefix is None:
        if "UNIFIED_TO_TARS_KEY_PREFIX" in os.environ:
            args.key_prefix = os.environ["UNIFIED_TO_TARS_KEY_PREFIX"]
        else:
            args.key_prefix = "t2i"

    if args.read_workers is None:
        args.read_workers = _env_int(
            "UNIFIED_TO_TARS_READ_WORKERS", min(os.cpu_count() or 4, 40)
        )
    if args.read_workers < 1:
        p.error("--read-workers 须为正整数")

    args.root = root.resolve()
    args.tar_dir = tar_dir.resolve()
    return args

def check_directory_for_pairs(path: Path):
    if path.name in ["input", "output", ".cache", "omniedit_object_swap", "omniedit_removal", "omniedit_swap", "omniedit_attribute_modification"]:
        return None
    in_d = path / "input"
    out_d = path / "output"
    if in_d.is_dir() and out_d.is_dir():
        return in_d, out_d, path
    return None

def iter_pair_dirs(root_dir: Path):
    skip_dirs = ["input", "output", ".cache", "omniedit_object_swap", "omniedit_removal", "omniedit_swap", "omniedit_attribute_modification"]
    for cur, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        maybe = check_directory_for_pairs(Path(cur))
        if maybe:
            yield maybe  # (input_dir, output_dir, base_path)

def iter_images(input_dir: Path):
    for ext in IMAGE_EXTENSIONS:
        yield from input_dir.rglob(f"*{ext}")

def resize_image_if_needed(image_bytes: bytes, max_size: int = 1024, quality: int = 95) -> tuple[bytes, bool]:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        max_dim = max(width, height)
        need_resize = max_dim > max_size

        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                rgb_img.paste(img, mask=img.split()[-1])
            else:
                rgb_img.paste(img)
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        if need_resize:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue(), need_resize
    except Exception as e:
        print(f"warning: image processing failed: {e}")
        return image_bytes, False

def load_pair(in_path: str, out_path: str):
    try:
        with open(in_path, "rb") as fi:
            inb = fi.read()
        with open(out_path, "rb") as fo:
            outb = fo.read()

        inb, in_resized = resize_image_if_needed(inb, max_size=1024, quality=95)
        outb, out_resized = resize_image_if_needed(outb, max_size=1024, quality=95)

        return os.path.basename(in_path), inb, outb, in_resized, out_resized
    except Exception:
        return None

def writer_thread_fn(result_q: Queue, stop_token, samples_per_shard: int, out_dir: Path, data_type: str = None):
    shard_id = 0
    n = 0
    sink = None

    def open_sink(i):
        p = out_dir / f"pairs-{i:06d}.tar"
        print(f"[writer] open: {p}")
        return wds.TarWriter(str(p))

    while True:
        item = result_q.get()
        if item is stop_token:
            break

        if sink is None:
            sink = open_sink(shard_id)

        key, in_bytes, out_bytes = item

        if n > 0 and n % samples_per_shard == 0:
            sink.close()
            shard_id += 1
            sink = open_sink(shard_id)

        sample = {"__key__": key, "in.png": in_bytes, "out.png": out_bytes}
        if data_type is not None and data_type in ["t2i", "c2i"]:
            sample["type"] = data_type
        sink.write(sample)
        n += 1
        del in_bytes, out_bytes, sample

    if sink is not None:
        sink.close()
        print(f"[writer] closed. total samples={n}")

def main(argv: list[str] | None = None):
    args = parse_args(argv)
    root = args.root
    tar_dir = args.tar_dir
    samples_per_shard = args.samples_per_shard
    data_type = args.data_type
    key_prefix = args.key_prefix
    read_workers = args.read_workers
    max_inflight = read_workers * 50
    result_queue_max = read_workers * 30

    tar_dir.mkdir(parents=True, exist_ok=True)

    print("task type: editing task (input/output pairs)")
    if data_type:
        print(f"data type: {data_type}")
    print("scanning directories and streaming processing...")
    t0 = time.time()

    STOP = object()
    result_q = Queue(maxsize=result_queue_max)
    writer = Thread(
        target=writer_thread_fn,
        args=(result_q, STOP, samples_per_shard, tar_dir, data_type),
        daemon=True,
    )
    writer.start()

    seen_names = set()
    use_path_in_key = False
    counter = itertools.count()

    produced = 0
    dispatched = 0
    resized_count = 0

    with ProcessPoolExecutor(max_workers=read_workers) as pool:
        inflight = set()  # (future, rel_path, filename)

        def submit(in_p: Path, out_p: Path, rel_path: str):
            fut = pool.submit(load_pair, str(in_p), str(out_p))
            inflight.add((fut, rel_path, in_p.name))

        def drain_done(block=False):
            nonlocal dispatched, use_path_in_key, resized_count
            while True:
                done_any = False
                for f, rel_path, fname in list(inflight):
                    if f.done():
                        done_any = True
                        inflight.remove((f, rel_path, fname))
                        tup = f.result()
                        if tup is None:
                            continue
                        _fname, inb, outb, in_resized, out_resized = tup

                        if in_resized:
                            resized_count += 1
                        if out_resized:
                            resized_count += 1

                        if not use_path_in_key:
                            if _fname in seen_names:
                                use_path_in_key = True
                            else:
                                seen_names.add(_fname)

                        idx = next(counter)
                        if use_path_in_key:
                            base_key = f"{idx:08d}_{rel_path.replace('/', '_').rsplit('.',1)[0]}"
                        else:
                            base_key = f"{idx:08d}"

                        if key_prefix:
                            key = f"{key_prefix}_{base_key}"
                        else:
                            key = base_key

                        result_q.put((key, inb, outb))
                        dispatched += 1
                if done_any:
                    if not block:
                        break
                else:
                    if block:
                        time.sleep(0.01)
                    else:
                        break

        dir_count = 0
        for input_dir, output_dir, base_path in iter_pair_dirs(root):
            imgs = list(iter_images(input_dir))
            if not imgs:
                continue

            for in_path in imgs:
                rel_path = in_path.relative_to(input_dir)
                stem_with_subdir = rel_path.with_suffix('')

                out_path = None
                for ext in IMAGE_EXTENSIONS:
                    candidate = output_dir / f"{stem_with_subdir}{ext}"
                    if candidate.exists():
                        out_path = candidate
                        break

                if out_path is None:
                    continue

                rel_path = str(in_path.relative_to(root))
                submit(in_path, out_path, rel_path)
                produced += 1

                if len(inflight) >= max_inflight:
                    wait({f for f,_,_ in inflight}, timeout=0.05, return_when=FIRST_COMPLETED)
                    drain_done(block=False)

            drain_done(block=False)

            dir_count += 1
            if dir_count % 10 == 0:
                print(f"directory={dir_count}, in flight={len(inflight)}, produced={produced}, dispatched={dispatched}, resized={resized_count}")

        while inflight:
            wait({f for f,_,_ in inflight}, timeout=0.05, return_when=FIRST_COMPLETED)
            drain_done(block=False)

    result_q.put(STOP)
    writer.join()

    t1 = time.time()
    print(f"completed: produced={produced}, dispatched={dispatched}, resized={resized_count}, time={t1 - t0:.2f}s")

if __name__ == "__main__":
    main()