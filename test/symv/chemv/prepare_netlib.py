# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build the task's pinned Netlib BLAS/CBLAS 3.10.0 in an isolated directory.

No package installation or system environment modification is performed.
Use --extract-only to prepare the verified source without building.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ARCHIVE_SHA256 = "328c1bea493a32cac5257d84157dc686cc3ab0b004e2bea22044e0a59f6f8a19"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_member(stream, member, destination):
    relative = Path(member.name)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "lapack-3.10.0":
        raise ValueError("Unsafe archive member: " + member.name)
    target = destination / relative
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        return
    if not member.isfile():
        raise ValueError("Unexpected link/special archive entry: " + member.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with stream.extractfile(member) as source, target.open("wb") as output:
        shutil.copyfileobj(source, output)


def extract(archive, destination):
    # Extract only ordinary files/directories below one known archive root.
    with tarfile.open(archive) as stream:
        for member in stream:
            extract_member(stream, member, destination)


def run(command, log):
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=3600)


def build_reference(args, work, source, manifest, manifest_path):
    build = work / "build"
    configure = [args.cmake, "-S", str(source), "-B", str(build), "-DCBLAS=ON",
                 "-DBUILD_SHARED_LIBS=ON", "-DBUILD_TESTING=OFF", "-DLAPACKE=OFF",
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_FLAGS=-O2 -fno-fast-math -ffp-contract=off",
                 "-DCMAKE_Fortran_FLAGS=-O2 -fno-fast-math -ffp-contract=off"]
    run(configure, work / "configure.log")
    run([args.cmake, "--build", str(build), "--target", "cblas", "--parallel", str(args.jobs)],
        work / "build.log")
    libraries = [(build / "lib" / name).resolve(strict=True) for name in ("libcblas.so", "libblas.so")]
    header_paths = (source / "CBLAS/include/cblas.h", build / "include/cblas_mangling.h")
    headers = {str(path): sha(path) for path in header_paths}
    manifest.update(built=True, libraries={str(path): sha(path) for path in libraries},
                    headers=headers, configure_command=configure,
                    cmake_cache_sha256=sha(build / "CMakeCache.txt"))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ['set(CHEMV_REFERENCE_VERSION "3.10.0")',
             'set(CHEMV_REFERENCE_SOURCE_SHA256 "' + ARCHIVE_SHA256 + '")']
    for path in libraries:
        lines += ['file(SHA256 "' + path.as_posix() + '" _chemv_actual_sha)',
                  'if(NOT _chemv_actual_sha STREQUAL "' + sha(path) + '")',
                  '  message(FATAL_ERROR "CHEMV reference library changed after build")', 'endif()']
    lines += ['set(REFBLAS_INCLUDE_DIR "' + (source / "CBLAS/include").as_posix() + '")',
              'include_directories("' + (build / "include").as_posix() + '")',
              'set(REFBLAS_LIB "' + ';'.join(path.as_posix() for path in libraries) + '")']
    (work / "netlib-reference.cmake").write_text('\n'.join(lines) + '\n', encoding="utf-8")
    logging.info("Built pinned Netlib reference: %s", manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="New isolated directory")
    parser.add_argument("--archive", type=Path, required=True, help="Local Netlib 3.10.0 source archive")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / "lapack-3.10.0.tar.gz"
    shutil.copyfile(args.archive.resolve(strict=True), archive)
    if sha(archive) != ARCHIVE_SHA256:
        raise ValueError("Netlib archive SHA256 mismatch")
    extract(archive, work)
    source = work / "lapack-3.10.0"
    manifest = {"netlib_version": "3.10.0", "source_archive_name": archive.name,
                "source_archive_sha256": ARCHIVE_SHA256, "built": False}
    manifest_path = work / "netlib-reference.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.extract_only:
        logging.info("Pinned source extracted: %s", source)
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "Device reference build requires Linux and a working Fortran compiler; use --extract-only here")
    build_reference(args, work, source, manifest, manifest_path)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
