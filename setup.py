# -*- coding: utf-8 -*-
# The MIT License (MIT)

# Copyright (c) 2014 Julien-Charles Lévesque

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from setuptools import setup
from setuptools.command.install import install
from distutils.command.build import build
from subprocess import call
from multiprocessing import cpu_count

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()
version = '1.0'

class build_bbsvm(build):
    def run(self):
        # run original build code
        build.run(self)

        build_path = os.path.abspath(self.build_temp)

        cmd = [
            'make',
            'OUT_DIR=' + build_path,
            'V=' + str(self.verbose),
        ]

        try:
            cmd.append('-j%d' % cpu_count())
        except NotImplementedError:
            print('Unable to determine number of CPUs. Using single threaded make.')

        options = [
            'DEBUG=n',
            'ENABLE_SDL=n',
        ]
        cmd.extend(options)

        targets = ['all']
        cmd.extend(targets)

        bsgd_path = 'budgetedsvm-code'
        target_files = [os.path.join(build_path, 'budgetedsvm-predict'),
            os.path.join(build_path, 'budgetedsvm-train')]

        def compile():
            print('*' * 80)
            call(cmd, cwd=bsgd_path)
            print('*' * 80)

        self.execute(compile, [], 'Compiling budgetedsvm toolbox.')

        # copy resulting tool to library build folder
        self.mkpath(self.build_lib)

        if not self.dry_run:
            for target in target_files:
                self.copy_file(target, self.build_lib)


class install_bbsvm(install):
    def initialize_options (self):
        install.initialize_options(self)
        self.build_scripts = None

    def finalize_options (self):
        install.finalize_options(self)
        self.set_undefined_options('build', ('build_scripts', 'build_scripts'))

    def run(self):
        # run original install code
        install.run(self)

        # install BBSVM executables
        print('running install_bbsvm')
        self.copy_tree(self.build_lib, self.install_lib)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


install_requires = [
    # List your project dependencies here.
    # For more details, see:
    # http://packages.python.org/distribute/setuptools.html#declaring-dependencies
]


setup(name='bagged-budget-svm',
    version=version,
    description='''Simple implementation of method described in
 Lévesque, J.-C., Gagné, C., & Sabourin, R. (2013). Ensembles
 of Budgeted Kernel Support Vector Machines for Parallel Large Scale
 Learning. In NIPS 2013 Workshop on Big Learning: Advances in
 Algorithms and Data Management (pp. 1–5).''',
    long_description=README,
    classifiers=[
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    keywords='ensemble methods,machine learning',
    author='Julien-Charles Lévesque',
    author_email='levesque.jc@gmail.com',
    license='MIT',
    packages=['bbsvm'],
    install_requires=install_requires,
    entry_points={
        #'console_scripts':
        #    ['mpt-sample=mptsample:main']
    },
    cmdclass={
        'build': build_bbsvm,
        'install': install_bbsvm
    }
)
