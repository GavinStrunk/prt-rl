# Getting Started


```python
spec = PPOPolicySpec(
    encoder=MLPEncoderSpec(hidden_sizes=(256, 256), activation="relu"),
    encoder_options=EncoderOptions(trainable=False, force_eval=True),  # frozen encoder
    backbone=MLPBackboneSpec(hidden_sizes=(128, 128), activation="relu"),
    heads=PPOHeadSpec(distribution="auto"),
)
policy = PPOPolicyFactory().make(env_params, spec)
```