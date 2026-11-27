# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build the shared test reference from unchanged Netlib sources, without installing system packages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile

PROFILE = "netlib-3.10.0-flang18.1.8-gcc9.4.0-strict-v1"
SOURCE_URL = "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.10.0.tar.gz"
SOURCE_SHA256 = "328c1bea493a32cac5257d84157dc686cc3ab0b004e2bea22044e0a59f6f8a19"
FLAGS = ["-O2", "-fPIC", "-fno-fast-math", "-ffp-contract=off"]
LOGGER = logging.getLogger(__name__)


def checked(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(" ".join(command) + "\n" + result.stdout + result.stderr)
    return result.stdout + result.stderr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help=SOURCE_URL)
    parser.add_argument("--output", type=Path, default=Path("third_party/reference-blas"))
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--fc", default="flang-new-18")
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args()


def executable_path(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError("Required executable not found: " + name)
    return str(Path(path).resolve(strict=True))


def prepare_toolchain(args):
    cc = executable_path(args.cc)
    fc = executable_path(args.fc)
    cc_version = checked([cc, "-dumpfullversion"]).strip()
    fc_version = checked([fc, "--version"])
    if cc_version != "9.4.0" or not re.search(r"flang(?:-new)? version 18\.1\.8\b", fc_version):
        raise RuntimeError("The reference profile requires GCC 9.4.0 and LLVM Flang 18.1.8")
    if platform.system() != "Linux" or args.jobs < 1:
        raise RuntimeError("A Linux host and a positive --jobs value are required")
    return {"cc": cc, "fc": fc, "fc_version": fc_version, "ar": executable_path("ar")}


def prepare_sources(archive_path, out):
    # Validate the release archive once; do not depend on a mutable working source tree.
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Expected the unchanged Netlib LAPACK v3.10.0 source archive: " + SOURCE_URL)
    for name in ("include", "lib", "objects", "source"):
        (out / name).mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError("Unexpected source archive entry: " + member.name)
        archive.extractall(out / "source", members=members)
    source = out / "source/lapack-3.10.0"
    for name in ("cblas.h", "cblas_f77.h"):
        shutil.copy2(source / "CBLAS/include" / name, out / "include" / name)
    (out / "include/cblas_mangling.h").write_text(
        "#define F77_GLOBAL(lc,UC) lc ## _\n#define F77_GLOBAL_(lc,UC) lc ## _\n")
    return source


def component_commands(component, directory, tools, out):
    sources = sorted(p for p in directory.iterdir() if p.suffix in (".f", ".f90", ".c"))
    objects = []
    commands = []
    for path in sources:
        obj = out / "objects" / (component + "_" + path.name + ".o")
        objects.append(str(obj))
        command = [tools["cc"] if path.suffix == ".c" else tools["fc"], *FLAGS]
        if path.suffix == ".c":
            command += ["-DHAS_ATTRIBUTE_WEAK_SUPPORT", "-I" + str(out / "include")]
        command += ["-c", str(path), "-o", str(obj)]
        commands.append(command)
    return commands, objects


def build_component(component, directory, tools, out, jobs):
    commands, objects = component_commands(component, directory, tools, out)
    with ThreadPoolExecutor(max_workers=jobs) as pool, (out / "build.log").open("a") as log:
        for command, output in zip(commands, pool.map(checked, commands)):
            log.write(" ".join(command) + "\n" + output)
    archive_command = [tools["ar"], "rcsD", str(out / "lib" / ("lib" + component + ".a")), *objects]
    checked(archive_command)
    commands.append(archive_command)
    LOGGER.info("%s: compiled %d unmodified source files", component, len(objects))
    return commands


def copy_runtime(source, out, fc):
    runtime = Path(fc).parent.parent / "lib"
    for name in ("libFortranRuntime.a", "libFortranDecimal.a"):
        shutil.copy2(runtime / name, out / "lib" / name)
    shutil.copy2(source / "LICENSE", out / "NETLIB-LICENSE")
    runtime_license = Path("/usr/share/doc/libflang-18-dev/copyright")
    if runtime_license.exists():
        shutil.copy2(runtime_license, out / "LLVM-RUNTIME-COPYRIGHT")


def write_manifest(out, tools, commands):
    manifest = {"profile": PROFILE, "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
                "machine": platform.machine(), "cc": checked([tools["cc"], "--version"]),
                "fc": tools["fc_version"], "flags": FLAGS, "commands": commands}
    (out / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_cmake_config(out):
    # Relative paths make the built reference reusable on another host of the same architecture.
    config = [
        'set(OPS_BLAS_REFERENCE_ID "' + PROFILE + '")',
        'set(OPS_BLAS_REFERENCE_ARCH "' + platform.machine() + '")',
        'set(REFBLAS_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")',
        'set(REFBLAS_LIB',
        '    "${CMAKE_CURRENT_LIST_DIR}/lib/libcblas.a"',
        '    "${CMAKE_CURRENT_LIST_DIR}/lib/libblas.a"',
        '    "${CMAKE_CURRENT_LIST_DIR}/lib/libFortranRuntime.a"',
        '    "${CMAKE_CURRENT_LIST_DIR}/lib/libFortranDecimal.a"',
        '    stdc++ m)',
    ]
    (out / "reference.cmake").write_text("\n".join(config) + "\n")
    (out / "reference-cache.cmake").write_text(
        'include("${CMAKE_CURRENT_LIST_DIR}/reference.cmake")\n'
        'set(REFBLAS_INCLUDE_DIR "${REFBLAS_INCLUDE_DIR}" CACHE PATH "Fixed reference headers" FORCE)\n'
        'set(REFBLAS_LIB "${REFBLAS_LIB}" CACHE STRING "Fixed reference libraries" FORCE)\n'
        'set(OPS_BLAS_REFERENCE_ROOT "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Fixed reference root" FORCE)\n'
        'set(OPS_BLAS_REFBLAS_MODE "source" CACHE STRING "Fixed reference mode" FORCE)\n'
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    tools = prepare_toolchain(args)
    out = args.output.resolve()
    source = prepare_sources(args.archive, out)
    commands = []
    for component, directory in (("blas", source / "BLAS/SRC"), ("cblas", source / "CBLAS/src")):
        commands.extend(build_component(component, directory, tools, out, args.jobs))
    copy_runtime(source, out, tools["fc"])
    write_manifest(out, tools, commands)
    write_cmake_config(out)
    LOGGER.info("Reference profile ready: %s", out / "reference.cmake")


if __name__ == "__main__":
    main()
