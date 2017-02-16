#!/bin/bash

__py_ver=${1:-3}
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

NUKE_EXISTING=yes

set -ex

# Python settings
export PY_VER=${__py_ver}
export PYTHON_PKG_PREFIX="$(python${PY_VER}-config --prefix)"
export PYTHON_BIN_PATH="$(which python${PY_VER})"
export PYTHON="$(which python${PY_VER})"
export PIP="$(which pip${PY_VER})"

if [ "darwin" == "${OS}" ]; then
    brew install bazel swig
else
    module load linuxbrew bazel python
    # if [[ "2" == "${PY_VER}" ]]; then
    #     module load python
    # else
    #     module unload python
    # fi    
    export JAVA_HOME=$(java -showversion -verbose 2>&1 | \
                           perl -ne 'print "$1\n" if /((\/.+)+)\/jre\/lib\/.+\.jar/' | \
                           uniq)
fi

"${PIP}" install -U six numpy wheel
(cd dev; echo "Update google/tensorflow repo"

 git remote add forigink https://github.com/tensorflow/tensorflow.git || echo "already there"
 git fetch --all
 git checkout forigink/master
 git submodule update --init --recursive 

 export USE_DEFAULT_PYTHON_LIB_PATH=1
 export TF_NEED_GCP=0
 export TF_NEED_HDFS=0
 export TF_NEED_CUDA=0
 export TF_NEED_OPENCL=0
 export TF_NEED_JEMALLOC=1
 export TF_ENABLE_XLA=0
 export PYTHON_BIN_PATH="$(which python3)"
 export CC_OPT_FLAGS="-march=native"
 ./configure
#  ./configure <<EOF
# $(find "${PYTHON_PKG_PREFIX}" -type d -name 'site-packages' | head -n1)
# EOF

 # Installing as a python package
 [ "yes" == "${NUKE_EXISTING}" ] && bazel clean --expunge
 bazel build -c opt --copt=-march=native //tensorflow/tools/pip_package:build_pip_package
 
 tf_pkg_dir="/tmp/tensorflow_pkg"
 rm -fr "${tf_pkg_dir}" && mkdir -p "${tf_pkg_dir}"
 ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "${tf_pkg_dir}"
 "${PIP}" uninstall -y tensorflow
 find "${tf_pkg_dir}/" -name 'tensorflow-*.whl' -exec "${PIP}" install {} \;

 # # Building locally for development
 # rm -fr _python_build && mkdir _python_build
 # cd _python_build
 # ln -s ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/* .
 # ln -s ../tensorflow/tools/pip_package/* .
 # "${PYTHON}" setup.py develop
)
