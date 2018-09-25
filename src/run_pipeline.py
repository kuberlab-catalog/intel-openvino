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
    version = train.exec_info.get('version')

    convert_cmd = convert.resource('worker')['command']
    replacement = '--saved_model_dir $TRAINING_DIR/%s/%s' % (train.build, timestamp)
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
    serving = serving.start(convert.name, convert.build)

    full_name = '%s-%s-%s-%s' % (
        os.environ.get('PROJECT_NAME'), serving.name, serving.task, serving.build
    )
    LOG.info('Started serving %s' % full_name)

    images_dir = 'mnist-images'
    files = [path.join(images_dir, p) for p in os.listdir(images_dir)]
    for f_name in files:
        test = imageio.imread(f_name).astype(np.float32).reshape(1, 1, 28, 28)
        data = {
            'inputs': {
                'x': {'dtype': 1, 'data': test.tolist()}
            }
        }

        resp = mlboard.servings.call(full_name, 'any', data)
        LOG.info('Response code %s' % resp.status_code)

        if resp.status_code >= 400:
            LOG.error(
                'Serving request failed with status %s: %s' % (
                    resp.status_code, resp.text
                )
            )
            sys.exit(1)

        answer = resp.json()
        softmax = list(answer.values())[0]

        base = path.basename(f_name)
        digit = np.array(softmax).reshape([10]).argmax()
        LOG.info('File %s, digit %s' % (base, digit))

        want_digit = base[:base.rfind('.')]
        if want_digit != digit:
            LOG.error('Serving has invalid result.')
            sys.exit(1)

    # Serving validated, need to export the model.
    export_path = path.join(os.environ['TRAINING_DIR'], convert.build)
    mlboard.model_upload('openvino-mnist', version, export_path)


if __name__ == '__main__':
    main()
