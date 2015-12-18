import os
import shutil

from ae.utils.flags import FLAGS, home_out
import ae.autoencoder as autoencoder
from ae.utils.start_tensorboard import start


_data_dir = FLAGS.data_dir
_summary_dir = FLAGS.summary_dir
_chkpt_dir = FLAGS.chkpt_dir


def _check_and_clean_dir(d):
  if os.path.exists(d):
    shutil.rmtree(d)
  os.mkdir(d)


def main():
  home = home_out('')
  if not os.path.exists(home):
    os.makedirs(home)
  if not os.path.exists(_data_dir):
    os.mkdir(_data_dir)

  _check_and_clean_dir(_summary_dir)
  _check_and_clean_dir(_chkpt_dir)

  os.mkdir(os.path.join(_chkpt_dir, '1'))
  os.mkdir(os.path.join(_chkpt_dir, '2'))
  os.mkdir(os.path.join(_chkpt_dir, '3'))
  os.mkdir(os.path.join(_chkpt_dir, 'fine_tuning'))

  start()

  ae = autoencoder.main_unsupervised()
  autoencoder.main_supervised(ae)

if __name__ == '__main__':
    main()
