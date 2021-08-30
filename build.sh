#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_PATH="${BASEPATH}/output"
BUILD_PATH="${BASEPATH}/build"
PYTHON=$(which python3)

mk_new_dir() {
    local create_dir="$1"  # the target to make

    if [[ -d "${create_dir}" ]];then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}

write_checksum() {
    cd "$OUTPUT_PATH" || exit
    PACKAGE_LIST=$(ls mindquantum-*.whl) || exit
    for PACKAGE_NAME in $PACKAGE_LIST; do
        echo $PACKAGE_NAME
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
}

mkdir -pv "${BUILD_PATH}"
cd ${BUILD_PATH}
#TODO: 🔥🔥🔥🔥🔥《项目编译》↪️2.优化build.sh的编译逻辑，将build.sh的编译指令传递给cmake，制定时候支持GPU，编译时的线程等
# cmake ${BASEPATH} -DGPUACCELERATED=1 -DGPU_COMPUTE_CAPABILITY=70 -DMULTITHREADED=0
cmake ${BASEPATH}
make -j12

cd ${BASEPATH}
mk_new_dir "${OUTPUT_PATH}"

${PYTHON} ${BASEPATH}/setup.py bdist_wheel

mv ${BASEPATH}/dist/*whl ${OUTPUT_PATH}

write_checksum


echo "------Successfully created mindquantum package------"
