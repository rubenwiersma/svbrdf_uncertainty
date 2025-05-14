# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import join, basename
from absl import app, flags
from tqdm import tqdm

from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil, io as ioutil, \
    config as configutil
from third_party.xiuminglib import xiuminglib as xm


flags.DEFINE_string(
    'ckpt', '/path/to/ckpt-100', "path to checkpoint (prefix only)")
flags.DEFINE_boolean('debug', False, "debug mode switch")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="nerf_test")


def main(_):
    if FLAGS.debug:
        logger.warn("Debug mode: on")

    # Config
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    # Output directory
    outroot = join(config_ini[:-4], 'vis_test', basename(FLAGS.ckpt))

    # Make dataset
    logger.info("Making the actual data pipeline")
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, 'test', debug=FLAGS.debug)
    n_views = dataset.get_n_views()
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)

    # Restore model
    logger.info("Restoring trained model")
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config, debug=FLAGS.debug)
    ioutil.restore_model(model, FLAGS.ckpt)

    # For all test views
    logger.info("Running inference")
    for batch_i, batch in enumerate(
            tqdm(datapipe, desc="Inferring Views", total=n_views)):
        # Inference
        _, _, _, to_vis = model.call(batch, mode='test')
        # Visualize
        outdir = join(outroot, 'batch{i:09d}'.format(i=batch_i))
        model.vis_batch(to_vis, outdir, mode='test')
        # Break if debugging
        if FLAGS.debug:
            break

    # Compile all visualized batches into a consolidated view (e.g., an
    # HTML or a video)
    batch_vis_dirs = xm.os.sortglob(outroot, 'batch?????????')
    outpref = outroot # proper extension should be added in the function below
    view_at = model.compile_batch_vis(batch_vis_dirs, outpref, mode='test')
    logger.info("Compilation available for viewing at\n\t%s", view_at)


if __name__ == '__main__':
    app.run(main)
