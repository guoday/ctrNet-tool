import tensorflow as tf
def create_fm_hparams():
    return tf.contrib.training.HParams(
            model='fm',
            k=16,
            hash_ids=int(1e7),
            batch_size=64,
            optimizer="adam",
            learning_rate=0.0002,
            num_display_steps=100,
            num_eval_steps=1000,
            epoch=3,
            metric='auc',
            init_method='uniform',
            init_value=0.1)