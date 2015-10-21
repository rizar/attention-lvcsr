import numpy
from numpy.testing import assert_equal, assert_allclose
from lvsr.error_rate import (
    _edit_distance_matrix, wer,
    reward_matrix, gain_matrix)
from lvsr.ops import RewardOp


def test_edit_distance_matrix():
    dist, action = _edit_distance_matrix('abdce', 'abcd')
    dist_should_be = numpy.array(
        [[0, 1, 2, 3, 4],
         [1, 0, 1, 2, 3],
         [2, 1, 0, 1, 2],
         [3, 2, 1, 1, 1],
         [4, 3, 2, 1, 2],
         [5, 4, 3, 2, 2]])
    assert_equal(dist, dist_should_be)
    action_should_be = (
        [[0, 0, 0, 0, 0],
         [0, 0, 2, 2, 2],
         [0, 0, 0, 2, 2],
         [0, 0, 0, 3, 0],
         [0, 0, 0, 0, 3],
         [0, 0, 0, 0, 3]])
    assert_equal(action, action_should_be)


def test_reward_matrix():
    assert_equal(reward_matrix('abc', 'cab', 'abc'),
        numpy.array([[ 1, -1, -1],
            [ 0,  1, -2],
            [-1,  1, -1],
            [ 0,  0,  0]]))
    assert_equal(reward_matrix('abc', 'abc', 'abc'),
        numpy.array([[ 1, -1, -1],
            [ 0,  2,  0],
            [ 1,  1,  3],
            [ 2,  2,  2]]))
    assert_equal(reward_matrix('abc', 'xxxabc', 'abcx'),
        numpy.array([[ 1, -1, -1, -1],
            [ 0,  1, -2, -2],
            [-1,  0,  1, -3],
            [-2, -1,  0, -4],
            [-3, -1, -3, -3],
            [-2, -2,  0, -2],
            [-1, -1, -1, -1]]))
    assert_equal(reward_matrix('xaxbxc', 'abc', 'abcx'),
        numpy.array([[-1, -1, -1,  1],
                [ 1, -2, -2,  0],
                [ 0,  2, -3, -1],
                [-4, -4, -4, -4]]))


def test_gain_matrix():
    assert_equal(gain_matrix('abc', 'cab', 'abc'),
            numpy.array([[ 1, -1, -1],
                  [ 1,  2, -1],
                  [ -1,  1, -1],
                  [ -1,  -1, -1]]))
    assert_equal(gain_matrix('abc', 'abc', 'abc'),
        numpy.array([[ 1, -1, -1],
            [-1,  1, -1],
            [-1, -1,  1],
            [-1, -1, -1]]))


def test_wer():
    assert_allclose(wer('abc', 'adc'), 0.333333, rtol=1e-4)


def test_reward_op():
    op = RewardOp(4, 7)
    groundtruth = [
        [0, 0, 0],
        [1, 2, 1],
        [2, 1, 4],
        [4, 3, 0],
        [0, 4, 0]]
    recognized = [
        [0, 0, 0],
        [2, 1, 1],
        [1, 2, 4],
        [3, 4, 0],
        [4, 0, 0]]
    rewards_var, gains_var = op(groundtruth, recognized)
    rewards = rewards_var.eval()
    gains = gains_var.eval()
    rewards_should_be = numpy.array([[[ 1, -1, -1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1, -1, -1]],

       [[ 0,  2,  0,  0,  0,  0,  0],
        [ 0,  0,  2,  0,  0,  0,  0],
        [ 0,  2,  0,  0,  0,  0,  0]],

       [[-1,  1,  2, -1,  3, -1, -1],
        [-1,  2,  1,  3, -1, -1, -1],
        [ 1,  1,  1,  1,  3,  1,  1]],

       [[ 0,  0,  2,  0,  0,  0,  0],
        [ 0,  2,  0,  0,  0,  0,  0],
        [ 2,  2,  2,  2,  2,  2,  2]],

       [[-1, -1,  1, -1,  2, -1, -1],
        [-1,  1, -1,  2, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1]]])
    assert_equal(rewards, rewards_should_be)
    gains_should_be = numpy.array([[[ 1, -1, -1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1, -1, -1]],

       [[-1,  1, -1, -1, -1, -1, -1],
        [-1, -1,  1, -1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1]],

       [[-1,  1,  2, -1,  3, -1, -1],
        [-1,  2,  1,  3, -1, -1, -1],
        [-1, -1, -1, -1,  1, -1, -1]],

       [[-1, -1,  1, -1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]],

       [[-1, -1,  1, -1,  2, -1, -1],
        [-1,  1, -1,  2, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]]])
    assert_equal(gains, gains_should_be)

