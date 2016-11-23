#!/bin/bash

ver=${1:-dev}
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

NUKE_EXISTING=yes

set -ex

if [ "darwin" == "${OS}" ]; then
    brew install bazel swig
else
    module load linuxbrew bazel python
    export JAVA_HOME=$(java -showversion -verbose 2>&1 | \
                           perl -ne 'print "$1\n" if /((\/.+)+)\/jre\/lib\/.+\.jar/' | \
                           uniq)
fi

pip3 install -U six numpy wheel
(cd dev; echo "Update google/tensorflow repo"

 git checkout master
 git submodule update --init --recursive
 git pull
 alias python="$(which python3)"

 export PYTHON_BIN_PATH="$(which python3)"
 export USE_DEFAULT_PYTHON_LIB_PATH=1
 export TF_NEED_GCP=0
 export TF_NEED_HDFS=0
 export TF_NEED_CUDA=0
 export TF_NEED_OPENCL=0
 ./configure <<EOF
$(find "$(python3-config --prefix)" -type d -name 'site-packages' | head -n1)
EOF

 # Installing as a python package
 [ "yes" == "${NUKE_EXISTING}" ] && bazel clean --expunge
 bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
 ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
 find /tmp/tensorflow_pkg -name 'tensorflow-*.whl' -exec pip3 install {} \;

 # Building locally for development
 rm -fr _python_build && mkdir _python_build
 cd _python_build
 ln -s ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/* .
 ln -s ../tensorflow/tools/pip_package/* .
 python3 setup.py develop
)
