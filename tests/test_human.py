# @pytest.mark.skip(reason="Requires a keyboard press")
# def test_keyboard_blocking_policy():
#     # Create a fake environment that has 1 discrete action [0,1] and 1 discrete state [0,...,3]
#     params = EnvParams(
#         action_shape=(1,),
#         action_continuous=False,
#         action_min=0,
#         action_max=1,
#         observation_shape=(1,),
#         observation_continuous=False,
#         observation_min=0,
#         observation_max=3,
#     )

#     policy = KeyboardPolicy(
#         env_params=params,
#         key_action_map={
#             'down': 0,
#             'up': 1,
#         }
#     )

#     # Create fake observation TensorDict
#     td = TensorDict({}, batch_size=[1])

#     # You have to press up for this to pass
#     print("Press up arrow key to pass")
#     td = policy.get_action(td)
#     assert td['action'][0] == 1


# @pytest.mark.skip(reason="Requires a keyboard press")
# def test_keyboard_nonblocking_policy():
#     # Create a fake environment that has 1 discrete action [0,1] and 1 discrete state [0,...,3]
#     params = EnvParams(
#         action_shape=(1,),
#         action_continuous=False,
#         action_min=0,
#         action_max=1,
#         observation_shape=(1,),
#         observation_continuous=False,
#         observation_min=0,
#         observation_max=3,
#     )

#     policy = KeyboardPolicy(
#         env_params=params,
#         key_action_map={
#             'down': 0,
#             'up': 1,
#         },
#         blocking=False,
#     )

#     # Create fake observation TensorDict
#     td = TensorDict({}, batch_size=[1])

#     # You have to press up for this to pass
#     action = 0
#     while action == 0:
#         td = policy.get_action(td)
#         action = td['action']
#         print(f"action: {action}")
#     assert td['action'][0] == 1


# @pytest.mark.skip(reason="Requires a game controller input")
# def test_processing_joystick():
#     params = EnvParams(
#         action_shape=(3,),
#         action_continuous=True,
#         action_min=[-2, -2, -2],
#         action_max=[2, 2, 2],
#         observation_shape=(1,),
#         observation_continuous=False,
#         observation_min=0,
#         observation_max=3,
#     )

#     policy = GameControllerPolicy(
#         env_params=params,
#         key_action_map={
#             GameControllerPolicy.Key.JOYSTICK_RIGHT_X: 0,
#             GameControllerPolicy.Key.JOYSTICK_RIGHT_Y: 1,
#             GameControllerPolicy.Key.JOYSTICK_LEFT_Y: (2, 'positive'),
#         },
#         blocking=False,
#     )

#     # Initial values are 0
#     assert torch.equal(policy.latest_values, torch.tensor([0.0, 0.0, 0.0]))

#     # Input only action index
#     action_map = 1
#     norm_joy_value = 1.0
#     policy._process_joystick(action_map, norm_joy_value)
#     action_value = policy.latest_values
#     assert torch.allclose(action_value, torch.tensor([0.0, 2.0, 0.0]))

#     action_map = 0
#     norm_joy_value = -0.5
#     policy._process_joystick(action_map, norm_joy_value)
#     assert torch.allclose(policy.latest_values, torch.tensor([-1.0, 2.0, 0.0]))

# @pytest.mark.skip(reason="Requires a game controller input")
# def test_processing_positive_clipped_joystick():
#     params = EnvParams(
#         action_shape=(3,),
#         action_continuous=True,
#         action_min=[-2, -2, 0],
#         action_max=[2, 2, 2],
#         observation_shape=(1,),
#         observation_continuous=False,
#         observation_min=0,
#         observation_max=3,
#     )

#     policy = GameControllerPolicy(
#         env_params=params,
#         key_action_map={
#             GameControllerPolicy.Key.JOYSTICK_RIGHT_X: 0,
#             GameControllerPolicy.Key.JOYSTICK_RIGHT_Y: 1,
#             GameControllerPolicy.Key.JOYSTICK_LEFT_Y: (2, 'positive'),
#         },
#         blocking=False,
#     )

#     action_map = (2, 'positive')
#     norm_joy_value = 0.5
#     policy._process_joystick(action_map, norm_joy_value)
#     assert torch.allclose(policy.latest_values, torch.tensor([0.0, 0.0, 1.0]))

#     # Negative values are clipped to zero
#     norm_joy_value = -0.5
#     policy._process_joystick(action_map, norm_joy_value)
#     assert torch.allclose(policy.latest_values, torch.tensor([0.0, 0.0, 0.0]))

# @pytest.mark.skip(reason="Requires a game controller input")
# def test_processing_negative_clipped_joystick():
#     params = EnvParams(
#         action_shape=(3,),
#         action_continuous=True,
#         action_min=[-2, -2, -2],
#         action_max=[2, 2, 0],
#         observation_shape=(1,),
#         observation_continuous=False,
#         observation_min=0,
#         observation_max=3,
#     )

#     policy = GameControllerPolicy(
#         env_params=params,
#         key_action_map={
#             GameControllerPolicy.Key.JOYSTICK_RIGHT_X: 0,
#             GameControllerPolicy.Key.JOYSTICK_RIGHT_Y: 1,
#             GameControllerPolicy.Key.JOYSTICK_LEFT_Y: (2, 'negative'),
#         },
#         blocking=False,
#     )

#     action_map = (2, 'negative')
#     norm_joy_value = -0.5
#     policy._process_joystick(action_map, norm_joy_value)
#     assert torch.allclose(policy.latest_values, torch.tensor([0.0, 0.0, -1.0]))

#     norm_joy_value = 0.5
#     policy._process_joystick(action_map, norm_joy_value)
#     assert torch.allclose(policy.latest_values, torch.tensor([0.0, 0.0, 0.0]))

# @pytest.mark.skip(reason="Requires a game controller input")
# def test_game_controller_blocking_policy_with_discrete_actions():
#     # Create a fake environment that has 1 discrete action [0,1] and 1 discrete state [0,...,3]
#     params = EnvParams(
#         action_shape=(1,),
#         action_continuous=False,
#         action_min=0,
#         action_max=1,
#         observation_shape=(1,),
#         observation_continuous=False,
#         observation_min=0,
#         observation_max=3,
#     )

#     policy = GameControllerPolicy(
#         env_params=params,
#         blocking=True,
#         key_action_map={
#             GameControllerPolicy.Key.BUTTON_DPAD_UP: 0,
#             GameControllerPolicy.Key.BUTTON_DPAD_DOWN: 1,
#             GameControllerPolicy.Key.BUTTON_DPAD_RIGHT: 2,
#             GameControllerPolicy.Key.BUTTON_DPAD_LEFT: 3,
#             GameControllerPolicy.Key.BUTTON_X: 4,
#         }
#     )

#     # Create fake observation TensorDict
#     td = TensorDict({}, batch_size=[1])

#     # Press dpad up to pass
#     out_td = policy.get_action(td)
#     assert out_td['action'][0] == 0

#     # Press dpad down to pass
#     out_td = policy.get_action(td)
#     assert out_td['action'][0] == 1

#     # Press dpad right to pass
#     out_td = policy.get_action(td)
#     assert out_td['action'][0] == 2

#     # Press dpad left to pass
#     out_td = policy.get_action(td)
#     assert out_td['action'][0] == 3


# @pytest.mark.skip(reason="Requires a game controller input")
# def test_game_controller_nonblocking_policy_with_continuous_actions():
#     params = EnvParams(
#         action_shape=(2,),
#         action_continuous=True,
#         action_min=-2,
#         action_max=2,
#         observation_shape=(1,),
#         observation_continuous=False,
#         observation_min=0,
#         observation_max=3,
#     )

#     policy = GameControllerPolicy(
#         env_params=params,
#         key_action_map={
#             GameControllerPolicy.Key.JOYSTICK_RIGHT_X: 0,
#             GameControllerPolicy.Key.JOYSTICK_RIGHT_Y: 1,
#         },
#         blocking=False,
#     )

#     # Create fake observation TensorDict
#     td = TensorDict({}, batch_size=[1])

#     # You have to press up for this to pass
#     action = 0
#     while action == 0:
#         td = policy.get_action(td)
#         action = td['action'][0][0]
#         print(f"action: {action}")
#     assert td['action'][0][0] > 0