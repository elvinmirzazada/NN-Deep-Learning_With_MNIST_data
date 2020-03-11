#!/usr/bin/env python3

"""Project setup file for the fabric8 analytics worker project."""

import os
from setuptools import setup, find_packages


def get_requirements():
    """Parse all packages mentioned in the 'requirements.txt' file."""
    with open('requirements.txt') as fd:
        return fd.read().splitlines()


setup(
    name='f8a_worker',
    version='0.2',
    scripts=[
        'hack/queue_conf.py',
        'hack/workers.sh',
        'hack/worker-queues-env.sh',
        'hack/worker-pre-hook.sh',
        'hack/worker-liveness.sh',
        'hack/worker-readiness.sh'
    ],
    package_data={},
    packages=find_packages(exclude=['model']),
    include_package_data=True,
    install_requires=get_requirements(),

    url='https://github.com/elvinmirzazadeh/NN-Deep-Learning_With_MNIST_data.git'
)
