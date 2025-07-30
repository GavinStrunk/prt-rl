from prt_rl.common.evaluators import RewardEvaluator
from prt_rl.env.wrappers import GymnasiumWrapper

def test_should_evaluate_based_on_num_steps():
    env = GymnasiumWrapper("CartPole-v1")
    evaluator = RewardEvaluator(env=env, num_episodes=5, eval_freq=2)

    assert evaluator._should_evaluate(0) is False
    assert evaluator._should_evaluate(1) is True

def test_should_evaluate_without_matching_frequency():
    env = GymnasiumWrapper("CartPole-v1")
    evaluator = RewardEvaluator(env=env, num_episodes=5, eval_freq=1000)

    assert evaluator._should_evaluate(0) is False
    assert evaluator._should_evaluate(990) is False
    assert evaluator._should_evaluate(1007) is True
    assert evaluator._should_evaluate(1500) is False
    assert evaluator._should_evaluate(2003) is True
    assert evaluator._should_evaluate(3000) is True