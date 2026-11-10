#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Build the pinned Netlib reference without invoking a command shell."""

from pathlib import Path
import argparse
import hashlib
import logging
import json
import os
import shutil
import subprocess
import sys
import urllib.request


HERE = Path(__file__).resolve().parent


def tool_path(name):
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"Required executable is not on PATH: {name}")
    return str(Path(executable).absolute())


def prepare_sources(output, inputs):
    """Verify pinned Netlib source bytes before compiling, including offline inputs."""
    entries = json.loads((HERE / "sources.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        name = entry["file"]
        if Path(name).name != name:
            raise ValueError("Invalid reference filename")
        dest = output / name
        if inputs:
            data = (inputs / name).read_bytes()
        elif dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == entry["sha256"]:
            data = dest.read_bytes()
        else:
            with urllib.request.urlopen(entry["source"], timeout=60) as response:
                data = response.read()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("Netlib source size/hash mismatch: " + name)
        dest.write_bytes(data)
        logging.info("[REFERENCE] verified %s %s", name, entry["sha256"])


def compile_objects(output, source, fortran, cc, fortran_names, c_names):
    for name in fortran_names:
        subprocess.run(
            [
                fortran,
                "-O2",
                "-fPIC",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-c",
                str(source / (name + ".f")),
                "-o",
                str(output / (name + ".o")),
            ],
            check=True,
        )
    for name in c_names:
        subprocess.run(
            [
                cc,
                "-O2",
                "-fPIC",
                "-I" + str(output),
                "-I" + str(source),
                "-c",
                str(source / (name + ".c")),
                "-o",
                str(output / (name + ".o")),
            ],
            check=True,
        )
    subprocess.run(
        [
            cc,
            "-O2",
            "-fPIC",
            "-c",
            str(HERE / "reference_environment.c"),
            "-o",
            str(output / "reference_environment.o"),
        ],
        check=True,
    )


def compile_reference(output, source):
    fortran, cc = tool_path("gfortran"), tool_path("gcc")
    fortran_names = ["sgemm", "cgemm", "lsame", "xerbla"]
    c_names = ["cblas_sgemm", "cblas_cgemm", "cblas_globals", "cblas_xerbla"]
    compile_objects(output, source, fortran, cc, fortran_names, c_names)

    objects = [str(output / (name + ".o")) for name in fortran_names + c_names + ["reference_environment"]]
    library = output / "libnetlib_sgemm.so"
    subprocess.run([fortran, "-shared", "-o", str(library), *objects, "-lm"], check=True)
    shutil.copyfile(source / "cblas.h", output / "cblas.h")
    return library


def build_reference(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source-inputs"
    inputs = os.environ.get("HGEMM_REFERENCE_INPUTS")
    prepare_sources(source, Path(inputs) if inputs else None)
    (output / "cblas_mangling.h").write_text("#define F77_GLOBAL(name,NAME) name ## _\n")
    library = compile_reference(output, source)

    return library


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    library = build_reference(parser.parse_args().output)
    logging.info(hashlib.sha256(library.read_bytes()).hexdigest() + "  " + str(library) + "\n")
