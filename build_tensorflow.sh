#!/bin/bash

__py_ver=${1:-3}
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

NUKE_EXISTING=no
BRANCH=master

set -euxo pipefail

# Python settings
export PY_VER="${__py_ver}"
export PYTHON_PKG_PREFIX="$(python${PY_VER}-config --prefix)"
export PYTHON_BIN_PATH="$(which python${PY_VER})"
export PYTHON="${PYTHON_BIN_PATH}"
export PIP="${PYTHON} -m pip"

if [[ "darwin" == "${OS}" ]]; then
    brew upgrade bazel swig || brew install bazel swig
else
    module load linuxbrew bazel python || echo "environment modules failed to load"
    # if [[ "2" == "${PY_VER}" ]]; then
    #     module load python
    # else
    #     module unload python
    # fi    
    export JAVA_HOME=$(java -showversion -verbose 2>&1 | \
                           perl -ne 'print "$1\n" if /((\/.+)+)\/jre\/lib\/.+\.jar/' | \
                           uniq)
fi

${PIP} install -U six numpy wheel
git submodule update --init --recursive --remote
(cd dev; echo "Update google/tensorflow repo"
 # Ignore reset for the time being
 #git reset --hard "origin/${BRANCH}"
 
 export USE_DEFAULT_PYTHON_LIB_PATH=1
 export TF_NEED_GCP=0
 export TF_NEED_HDFS=0
 export TF_NEED_CUDA=0
 export TF_NEED_OPENCL=0
 export TF_NEED_JEMALLOC=1
 export TF_NEED_S3=1
 export TF_NEED_GDR=0
 export TF_ENABLE_XLA=1
 export TF_NEED_MKL=1
 export TF_NEED_MPI=0
 export TF_NEED_VERBS=0
 export TF_NEED_OPENCL_SYCL=0
 export TF_SET_ANDROID_WORKSPACE=0
 
 export CC_OPT_FLAGS="-march=native"
 ./configure

 # Installing as a python package
 [[ "yes" == "${NUKE_EXISTING}" ]] && bazel clean --expunge
 bazel build \
       --config=mkl \
       --copt=-march=native \
       //tensorflow/tools/pip_package:build_pip_package
 
 tf_pkg_dir="/tmp/tensorflow_pkg"
 rm -fr "${tf_pkg_dir}" && mkdir -p "${tf_pkg_dir}"
 ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "${tf_pkg_dir}"
 ${PIP} uninstall -y tensorflow
 find "${tf_pkg_dir}/" -name 'tensorflow-*.whl' -exec ${PIP} install -U {} \;
)
