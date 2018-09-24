import logging
import os
from os import path
import re
import sys

import imageio
import numpy as np

from mlboardclient.api import client


mlboard = client.Client()


LOG = logging.getLogger(__name__)
pattern = re.compile('--saved_model_dir[ =].*\s')


def main():
    app = mlboard.apps.get()
    train = app.tasks.get('train')
    convert = app.tasks.get('convert')

    train = train.start()
    LOG.info('Started task %s', train.name)
    train.wait()

    LOG.info('Task %s completed with status %s', (train.name, train.status))

    if train.status != 'Succeeded':
        LOG.error('Fail workflow')
        sys.exit(1)

    # Train is completed
    timestamp = train.exec_info.get('timestamp')
    model_path = train.exec_info.get('model_path')

    convert_cmd = convert.resource('worker')['command']
    replacement = '--saved_model_dir $TRAINING_DIR/%s' % timestamp
    if pattern.findall(convert_cmd):
        # Replace
        convert.resource('worker')['command'] = pattern.sub(
            replacement,
            convert_cmd
        )
    else:
        # Add
        convert.resource('worker')['args'] = {'saved_model_dir': replacement}

    convert = convert.start()
    LOG.info('Started task %s' % convert.name)
    convert.wait()
    LOG.info('Task %s completed with status %s' % (convert.name, convert.status))

    if convert.status != 'Succeeded':
        LOG.error('Fail workflow')
        sys.exit(1)

    serving = app.servings[0]
    serving = serving.start(train.name, train.build)

    LOG.info('Started serving %s' % serving.name)

    images_dir = 'mnist-images'
    files = [path.join(images_dir, p) for p in os.listdir(images_dir)]
    for f_name in files:
        test = imageio.imread(f_name).astype(np.float32).reshape(1, 1, 28, 28)
        data = {
            'inputs': {
                'x': {'dtype': 1, 'data': test.tolist()}
            }
        }

        resp = mlboard.servings.call(serving.name, 'any', data)
        LOG.info('Response code %s' % resp.status_code)


if __name__ == '__main__':
    main()
