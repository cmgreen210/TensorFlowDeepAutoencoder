from __future__ import division
from __future__ import print_function

import tensorflow as tf
from autoencoder import AutoEncoder


class AutoEncoderTest(tf.test.TestCase):

  def test_constructor(self):
    with self.test_session() as sess:

      ae_shape = [10, 20, 30, 2]
      self.assertTrue(AutoEncoder(ae_shape, sess))

  def test_get_variables(self):
    with self.test_session() as sess:
      ae_shape = [10, 20, 30, 2]
      ae = AutoEncoder(ae_shape, sess)

      with self.assertRaises(AssertionError):
        ae.get_variables_to_init(0)
      with self.assertRaises(AssertionError):
        ae.get_variables_to_init(4)

      v1 = ae.get_variables_to_init(1)
      self.assertEqual(len(v1), 3)

      v2 = ae.get_variables_to_init(2)
      self.assertEqual(len(v2), 5)

      v3 = ae.get_variables_to_init(3)
      self.assertEqual(len(v3), 2)

  def test_nets(self):
    with self.test_session() as sess:
      ae_shape = [10, 20, 30, 2]
      ae = AutoEncoder(ae_shape, sess)

      input_pl = tf.placeholder(tf.float32, shape=(100, 10))
      with self.assertRaises(AssertionError):
        ae.pretrain_net(input_pl, 0)
      with self.assertRaises(AssertionError):
        ae.pretrain_net(input_pl, 3)

      net1 = ae.pretrain_net(input_pl, 1)
      net2 = ae.pretrain_net(input_pl, 2)

      self.assertEqual(net1.get_shape().dims[1].value, 10)
      self.assertEqual(net2.get_shape().dims[1].value, 20)

      net1_target = ae.pretrain_net(input_pl, 1, is_target=True)
      self.assertEqual(net1_target.get_shape().dims[1].value, 10)
      net2_target = ae.pretrain_net(input_pl, 2, is_target=True)
      self.assertEqual(net2_target.get_shape().dims[1].value, 20)

      sup_net = ae.supervised_net(input_pl)
      self.assertEqual(sup_net.get_shape().dims[1].value, 2)
