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
# ============================================================================
"""Setup."""

import os
import stat
from setuptools import setup
from setuptools import find_packages
from setuptools.command.build_py import build_py

cur_dir = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(cur_dir, 'build')


def read_version():
    """generate python file"""
    version_file = os.path.join(cur_dir, 'mindquantum/', 'version.py')
    with open(version_file, 'r') as f:
        version_ = f.readlines()[-1].strip().split()[-1][1:-1]
    return version_


version = read_version()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(
                dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC
                | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


class BuildPy(build_py):
    """BuildPy."""
    def run(self):
        super().run()
        mindquantum_dir = os.path.join(pkg_dir, 'lib', 'mindquantum')
        update_permissions(mindquantum_dir)


with open('requirements.txt', 'r') as f_requirements:
    requirements = f_requirements.readlines()
requirements = [r.strip() for r in requirements]

#TODO: 🔥🔥🔥🔥🔥《项目编译》↪️1.优化编译、打包流程，将项目编译成与编译时的python版本相关的whl包
setup(name='mindquantum',
      version=version,
      author='The MindSpore Authors',
      author_email='contact@mindspore.cn',
      url='https://www.mindspore.cn/',
      download_url='https://gitee.com/mindspore/mindquantum/tags',
      project_urls={
          'Sources': 'https://gitee.com/mindspore/mindquantum',
          'Issue Tracker': 'https://gitee.com/mindspore/mindquantum/issues',
      },
      description="A hybrid quantum-classic framework for quantum computing",
      license='Apache 2.0',
      packages=find_packages(),
      package_data={'': ['*.so*']},
      include_package_data=True,
      cmdclass={
          'build_py': BuildPy,
      },
      install_requires=requirements,
      classifiers=['License :: OSI Approved :: Apache Software License'])
print(find_packages())
