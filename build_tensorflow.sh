#!/bin/bash

ver=${1:-dev}
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

NUKE_EXISTING=yes

set -ex

if [ "darwin" == "${OS}" ]; then
    brew install bazel swig
else
    module load bazel swig python
    export JAVA_HOME=$(java -showversion -verbose 2>&1 | \
                           perl -ne 'print "$1\n" if /((\/.+)+)\/jre\/lib\/.+\.jar/' | \
                           uniq)

    # module load toolchain/gcc6.1.0-glibc2.23-binutils2.26-native-integ
    # export CC=gcc
    # export CXX=g++
    # export LDFLAGS="-Wl,-rpath=${TOOLCHAIN_ROOT}/lib64 -L${TOOLCHAIN_ROOT}/lib64 -static-libgcc -static-libstdc++"
fi

if [ "dev" != "${ver}" ]; then
    pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-${ver}-py3-none-any.whl
    exit
fi

pip3 install -U six numpy wheel
git submodule update --init --recursive
(cd dev; echo "Update google/tensorflow repo"

 git submodule update --init --recursive
 git pull 
 ./configure <<EOF
$(which python3)
N
EOF

 echo $JAVA_HOME

 # Installing as a python package
 [ "yes" == "${NUKE_EXISTING}" ] && bazel clean --expunge
 bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
 ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
 pip3 install /tmp/tensorflow_pkg/tensorflow-*-py3-none-linux_x86_64.whl

 # Building locally for development
 rm -fr _python_build && mkdir _python_build
 cd _python_build
 ln -s ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/* .
 ln -s ../tensorflow/tools/pip_package/* .
 python3 setup.py develop
)
