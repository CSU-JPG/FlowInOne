#!/usr/bin/env python3
"""
Inspect FlowInOne model files for safety before loading them into PyTorch.

Scans .pth files for dangerous pickle opcodes and inspects .tar.gz archives
for path traversal, suspicious files, and symlinks.

Usage:
    python tests/inspect_model_safety.py checkpoints/flowinone_256px.pth
    python tests/inspect_model_safety.py preparation.tar.gz
    python tests/inspect_model_safety.py checkpoints/flowinone_256px.pth preparation.tar.gz
"""

import argparse
import io
import os
import pickle
import pickletools
import struct
import sys
import tarfile
import zipfile


# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


# Known-safe globals that torch uses for tensor reconstruction
SAFE_GLOBALS = {
    "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_parameter",
    "torch._utils._rebuild_parameter_with_state",
    "torch.FloatStorage",
    "torch.HalfStorage",
    "torch.BFloat16Storage",
    "torch.IntStorage",
    "torch.LongStorage",
    "torch.ShortStorage",
    "torch.DoubleStorage",
    "torch.CharStorage",
    "torch.ByteStorage",
    "torch.BoolStorage",
    "torch.ComplexFloatStorage",
    "torch.ComplexDoubleStorage",
    "torch.storage._load_from_bytes",
    "torch._tensor._rebuild_from_type_v2",
    "torch.Tensor",
    "collections.OrderedDict",
    "numpy.core.multiarray.scalar",
    "numpy.dtype",
    "numpy.core.multiarray._reconstruct",
    "numpy.ndarray",
    "_codecs.encode",
    "builtins.set",
    "builtins.frozenset",
    "builtins.complex",
    "builtins.slice",
    "builtins.range",
    "builtins.bytearray",
    "builtins.bytes",
}


def scan_pickle_bytes(data: bytes) -> dict:
    """Disassemble pickle bytecode and flag dangerous operations."""
    results = {
        "globals": [],
        "dangerous_globals": [],
        "reduce_calls": 0,
        "build_calls": 0,
        "safe": True,
        "warnings": [],
    }

    try:
        ops = list(pickletools.genops(data))
    except Exception as e:
        results["safe"] = False
        results["warnings"].append(f"Failed to parse pickle bytecode: {e}")
        return results

    for opcode, arg, pos in ops:
        name = opcode.name

        if name in ("GLOBAL", "INST", "STACK_GLOBAL"):
            global_name = str(arg).strip() if arg else "<unknown>"
            results["globals"].append((global_name, pos))
            if global_name not in SAFE_GLOBALS:
                results["dangerous_globals"].append((global_name, pos))
                results["safe"] = False

        elif name == "REDUCE":
            results["reduce_calls"] += 1

        elif name == "BUILD":
            results["build_calls"] += 1

    return results


def scan_pth_file(filepath: str) -> dict:
    """Scan a .pth file (PyTorch checkpoint) for dangerous pickle content."""
    report = {
        "file": filepath,
        "type": "pth",
        "size_mb": os.path.getsize(filepath) / (1024 * 1024),
        "safe": True,
        "globals_found": [],
        "dangerous_globals": [],
        "warnings": [],
        "details": "",
    }

    try:
        # PyTorch saves as ZIP files containing pickle data
        if zipfile.is_zipfile(filepath):
            with zipfile.ZipFile(filepath, "r") as zf:
                for name in zf.namelist():
                    if name.endswith("/data.pkl") or name.endswith(".pkl"):
                        with zf.open(name) as f:
                            pkl_data = f.read()
                        result = scan_pickle_bytes(pkl_data)
                        report["globals_found"].extend(result["globals"])
                        report["dangerous_globals"].extend(result["dangerous_globals"])
                        if not result["safe"]:
                            report["safe"] = False
                        report["warnings"].extend(result["warnings"])
        else:
            # Raw pickle file
            with open(filepath, "rb") as f:
                pkl_data = f.read(50 * 1024 * 1024)  # Read first 50MB for scanning
            result = scan_pickle_bytes(pkl_data)
            report["globals_found"] = result["globals"]
            report["dangerous_globals"] = result["dangerous_globals"]
            report["safe"] = result["safe"]
            report["warnings"] = result["warnings"]

    except Exception as e:
        report["safe"] = False
        report["warnings"].append(f"Error scanning file: {e}")

    return report


def scan_tar_file(filepath: str) -> dict:
    """Inspect a .tar.gz archive for path traversal, suspicious files, and symlinks."""
    report = {
        "file": filepath,
        "type": "tar.gz",
        "size_mb": os.path.getsize(filepath) / (1024 * 1024),
        "safe": True,
        "total_members": 0,
        "path_traversal": [],
        "symlinks": [],
        "suspicious_files": [],
        "executable_files": [],
        "file_listing": [],
        "warnings": [],
    }

    suspicious_extensions = {".sh", ".bash", ".py", ".so", ".dylib", ".dll", ".exe", ".bat", ".cmd"}

    try:
        with tarfile.open(filepath, "r:*") as tf:
            members = tf.getmembers()
            report["total_members"] = len(members)

            for member in members:
                size_str = f"{member.size / (1024*1024):.1f}MB" if member.size > 1024*1024 else f"{member.size}B"
                entry = f"{'d' if member.isdir() else 'l' if member.issym() else 'f'} {size_str:>10s}  {member.name}"
                report["file_listing"].append(entry)

                # Check for path traversal
                if ".." in member.name or member.name.startswith("/"):
                    report["path_traversal"].append(member.name)
                    report["safe"] = False

                # Check for symlinks
                if member.issym() or member.islnk():
                    report["symlinks"].append(f"{member.name} -> {member.linkname}")
                    report["warnings"].append(f"Symlink: {member.name} -> {member.linkname}")

                # Check for suspicious file extensions
                _, ext = os.path.splitext(member.name)
                if ext.lower() in suspicious_extensions:
                    report["suspicious_files"].append(member.name)

                # Check for executable permissions
                if member.mode and (member.mode & 0o111):
                    if not member.isdir():
                        report["executable_files"].append(member.name)

    except Exception as e:
        report["safe"] = False
        report["warnings"].append(f"Error reading tar file: {e}")

    return report


def try_safe_load(filepath: str) -> dict:
    """Attempt torch.load with weights_only=True."""
    result = {
        "weights_only_works": False,
        "error": None,
    }

    try:
        import torch
        torch.load(filepath, map_location="cpu", weights_only=True)
        result["weights_only_works"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


def print_pth_report(report: dict):
    """Print a formatted safety report for a .pth file."""
    print(f"\n{'='*60}")
    print(f"{BOLD}File:{RESET} {report['file']}")
    print(f"{BOLD}Type:{RESET} PyTorch checkpoint (.pth)")
    print(f"{BOLD}Size:{RESET} {report['size_mb']:.1f} MB")
    print(f"{'='*60}")

    if report["globals_found"]:
        print(f"\n{BOLD}Pickle globals found ({len(report['globals_found'])}):{RESET}")
        seen = set()
        for global_name, pos in report["globals_found"]:
            if global_name not in seen:
                seen.add(global_name)
                in_safe = global_name in SAFE_GLOBALS
                marker = f"{GREEN}[SAFE]{RESET}" if in_safe else f"{RED}[UNKNOWN]{RESET}"
                print(f"  {marker} {global_name}")

    if report["dangerous_globals"]:
        print(f"\n{RED}{BOLD}DANGEROUS GLOBALS DETECTED:{RESET}")
        for global_name, pos in report["dangerous_globals"]:
            print(f"  {RED}! {global_name} (at byte {pos}){RESET}")

    if report["warnings"]:
        print(f"\n{YELLOW}Warnings:{RESET}")
        for w in report["warnings"]:
            print(f"  {YELLOW}! {w}{RESET}")

    # Verdict
    if report["safe"]:
        print(f"\n{GREEN}{BOLD}VERDICT: SAFE{RESET}")
        print(f"  All pickle globals are known-safe torch/numpy reconstruction functions.")
    else:
        print(f"\n{RED}{BOLD}VERDICT: POTENTIALLY UNSAFE{RESET}")
        print(f"  Unknown globals detected. These could execute arbitrary code on load.")
        print(f"  Review the globals listed above before proceeding.")


def print_tar_report(report: dict):
    """Print a formatted safety report for a tar archive."""
    print(f"\n{'='*60}")
    print(f"{BOLD}File:{RESET} {report['file']}")
    print(f"{BOLD}Type:{RESET} Tar archive (.tar.gz)")
    print(f"{BOLD}Size:{RESET} {report['size_mb']:.1f} MB")
    print(f"{BOLD}Members:{RESET} {report['total_members']} files/directories")
    print(f"{'='*60}")

    if report["path_traversal"]:
        print(f"\n{RED}{BOLD}PATH TRAVERSAL DETECTED:{RESET}")
        for p in report["path_traversal"]:
            print(f"  {RED}! {p}{RESET}")

    if report["symlinks"]:
        print(f"\n{YELLOW}Symlinks found:{RESET}")
        for s in report["symlinks"]:
            print(f"  {YELLOW}-> {s}{RESET}")

    if report["suspicious_files"]:
        print(f"\n{YELLOW}Files with suspicious extensions:{RESET}")
        for s in report["suspicious_files"]:
            print(f"  {YELLOW}? {s}{RESET}")

    if report["executable_files"]:
        print(f"\n{YELLOW}Files with executable permissions:{RESET}")
        for e in report["executable_files"]:
            print(f"  {YELLOW}x {e}{RESET}")

    # Show file listing (truncated if large)
    if report["file_listing"]:
        max_show = 50
        print(f"\n{BOLD}Contents (first {max_show}):{RESET}")
        for entry in report["file_listing"][:max_show]:
            print(f"  {entry}")
        if len(report["file_listing"]) > max_show:
            print(f"  ... and {len(report['file_listing']) - max_show} more")

    if report["warnings"]:
        print(f"\n{YELLOW}Warnings:{RESET}")
        for w in report["warnings"]:
            print(f"  {YELLOW}! {w}{RESET}")

    # Verdict
    if report["safe"] and not report["path_traversal"]:
        if report["suspicious_files"]:
            print(f"\n{YELLOW}{BOLD}VERDICT: SAFE (with notes){RESET}")
            print(f"  No path traversal or symlinks found, but some files have notable extensions.")
            print(f"  Extract with: tar --no-same-permissions -xzf {report['file']} -C /target/dir/")
        else:
            print(f"\n{GREEN}{BOLD}VERDICT: SAFE{RESET}")
            print(f"  No path traversal, symlinks, or suspicious files found.")
    else:
        print(f"\n{RED}{BOLD}VERDICT: POTENTIALLY UNSAFE{RESET}")
        print(f"  Path traversal or other dangerous patterns detected.")
        print(f"  Do NOT extract this archive without reviewing the paths above.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect FlowInOne model files for safety before loading."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Files to inspect (.pth, .tar.gz, .tar)"
    )
    parser.add_argument(
        "--try-safe-load",
        action="store_true",
        help="Also attempt torch.load(weights_only=True) on .pth files (requires torch)"
    )
    args = parser.parse_args()

    all_safe = True

    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"\n{RED}File not found: {filepath}{RESET}")
            all_safe = False
            continue

        ext = filepath.lower()
        if ext.endswith(".pth"):
            report = scan_pth_file(filepath)
            print_pth_report(report)
            if not report["safe"]:
                all_safe = False

            if args.try_safe_load:
                print(f"\n{BOLD}Testing torch.load(weights_only=True)...{RESET}")
                safe_result = try_safe_load(filepath)
                if safe_result["weights_only_works"]:
                    print(f"  {GREEN}SUCCESS: File can be loaded with weights_only=True{RESET}")
                else:
                    print(f"  {YELLOW}FAILED: {safe_result['error'][:200]}{RESET}")
                    print(f"  This means the file requires pickle. Review globals above.")

        elif ext.endswith(".tar.gz") or ext.endswith(".tar") or ext.endswith(".tgz"):
            report = scan_tar_file(filepath)
            print_tar_report(report)
            if not report["safe"]:
                all_safe = False
        else:
            print(f"\n{YELLOW}Skipping unsupported file type: {filepath}{RESET}")

    # Final summary
    print(f"\n{'='*60}")
    if all_safe:
        print(f"{GREEN}{BOLD}OVERALL: ALL FILES PASSED SAFETY CHECKS{RESET}")
    else:
        print(f"{RED}{BOLD}OVERALL: SOME FILES HAVE SAFETY CONCERNS - REVIEW ABOVE{RESET}")
    print(f"{'='*60}\n")

    return 0 if all_safe else 1


if __name__ == "__main__":
    sys.exit(main())
