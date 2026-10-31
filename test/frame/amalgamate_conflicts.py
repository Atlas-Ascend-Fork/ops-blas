#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""合并测试 Runner（blas_all_test，issue #457 全量耗时优化）的链接冲突检测。

扫描各 *_test.cpp 的文件级（列首、非 static、非模板）全局函数/变量定义，
找出跨文件重名的全局符号清单。blas_all_test 链接全部算子的目标文件时，
同名强符号会触发 "multiple definition" 链接错误；编译 launcher 依据本脚本
输出的清单对每个 .o 执行 objcopy --localize-symbols，把冲突符号降为局部
符号（各文件的重定位自动绑定到各自的副本），从而安全合并。

不纳入清单的符号：
  - static / 匿名 namespace 成员：内部链接，天然不冲突；
  - 模板及其实例化（vague linkage）：弱符号，链接器自动合并；
  - 函数体内局部定义：局部符号；
  - gtest/框架头内联符号：弱符号或唯一命名。

用法：
    python3 amalgamate_conflicts.py <src1.cpp> <src2.cpp> ...
输出：
    每行一个冲突符号名（stdout）；无冲突时输出为空。
"""

import re
import sys

# 文件级（列首）定义识别：只匹配无前导空白的行，避免误收函数体/类内成员。
# mods 里的 static / template 定义分别属于内部链接与弱符号，均排除。
FUNC_DEF = re.compile(
    r'^(?P<mods>(?:(?:static|inline|constexpr|extern|virtual)\s+)*)'
    r'[A-Za-z_][\w:]*[^;{()=]*?[\s*&](?P<name>[A-Za-z_]\w*)\s*\([^;{]*$',
    re.M)
VAR_DEF = re.compile(
    r'^(?P<mods>(?:(?:static|inline|constexpr|extern)\s+)*)'
    r'[A-Za-z_][\w:]*[^;{()=]*?[\s*&](?P<name>[A-Za-z_]\w*)\s*=\s*[^=]',
    re.M)
# template 声明行（可能与定义同行或独立一行）
TEMPLATE_LINE = re.compile(r'^\s*template\s*<', re.M)

# 框架/头文件内符号（inline 或模板），永不产生强符号冲突
WHITELIST = {
    'main',
}


def strip_comments(text):
    """去掉块注释与行注释，避免把注释里的示例代码误判为定义。"""
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text = re.sub(r'//[^\n]*', ' ', text)
    return text


def defined_names(path):
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            text = strip_comments(f.read())
    except OSError:
        return set()

    # 标记 template 修饰的行：模板与实例化是弱符号，直接排除。
    # 做法：把 "template <...>" 行整体替换为 static 修饰符行，让 mods 过滤生效。
    text = re.sub(r'^template\s*<[^>]*>\s*$', 'static ', text, flags=re.M)
    text = re.sub(r'^template\s*<[^>]*>\s*(?=\S)', 'static ', text, flags=re.M)

    names = set()
    for pat in (FUNC_DEF, VAR_DEF):
        for m in pat.finditer(text):
            mods = (m.group('mods') or '').split()
            # issue #720: 仅 static 修饰的符号需要 objcopy localize；constexpr 变量在
            # C++ 中本就是内部链接、无跨文件符号冲突，无需列入清单（此前恒假子句
            # `and False` 属调试遗留，语义失真且误导读者）
            if 'static' in mods:
                continue
            name = m.group('name')
            if name in WHITELIST or name in (
                    'if', 'for', 'while', 'switch', 'return', 'catch'):
                continue
            names.add(name)

    # 剔除匿名 namespace 段内的定义（内部链接，不参与跨文件冲突）
    for m in re.finditer(r'^namespace\s*\{', text, re.M):
        start = m.end()
        tail = text[start:start + 200000]
        end_m = re.search(r'^\}(\s*//.*)?$', tail, re.M)
        seg = tail[:end_m.start()] if end_m else tail
        for pat in (FUNC_DEF, VAR_DEF):
            for dm in pat.finditer(seg):
                names.discard(dm.group('name'))
    return names


def main():
    args = sys.argv[1:]
    out_path = None
    if args and args[0] == '-o':
        if len(args) < 2:
            print('usage: amalgamate_conflicts.py [-o <out_file>] <src...>', file=sys.stderr)
            return 2
        out_path = args[1]
        args = args[2:]

    per_file = [defined_names(p) for p in args]
    seen = {}
    for names in per_file:
        for n in names:
            seen[n] = seen.get(n, 0) + 1
    lines = [n for n, c in sorted(seen.items()) if c > 1]

    if out_path:
        # CMake add_custom_command 场景：显式写输出文件（shell 重定向在
        # VERBATIM 模式下会被转义为普通参数，不能用 >）
        with open(out_path, 'w', encoding='utf-8') as f:
            for n in lines:
                f.write(n + '\n')
    else:
        for n in lines:
            print(n)
    return 0


if __name__ == '__main__':
    sys.exit(main())
