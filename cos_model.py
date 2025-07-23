import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

TRUE_A = 0.7
TRUE_B = -0.5


def data_generator(num_samples=1000):
    """Yield random x values and cos outputs."""
    rng = np.random.default_rng()
    for _ in range(num_samples):
        x = rng.uniform(-np.pi, np.pi)
        y = np.cos(TRUE_A * x + TRUE_B)
        yield np.float32(x), np.float32(y)


class CosModel(tf.keras.Model):
    """Model y = cos(a * x + b) with trainable a, b."""

    def __init__(self):
        super().__init__()
        # Explicitly pass the parameter names to avoid conflicts across
        # different TensorFlow versions where ``shape`` might be the second
        # positional argument. Using keyword arguments ensures the code works
        # regardless of the expected parameter order.
        self.a = self.add_weight(name="a", shape=(), initializer="ones")
        self.b = self.add_weight(name="b", shape=(), initializer="zeros")

    def call(self, inputs):
        return tf.cos(self.a * inputs + self.b)


@tf.function
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def main():
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(1000),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    ).batch(32).repeat()

    model = CosModel()
    if len(model.trainable_variables) != 2:
        raise RuntimeError("Model should have exactly two trainable variables")
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss=mse)

    model.fit(dataset, epochs=200, verbose=0)
    print(f"Trained parameters: a={model.a.numpy():.3f}, b={model.b.numpy():.3f}")
    xs = tf.linspace(-np.pi, np.pi, 200)
    ys_true = tf.cos(TRUE_A * xs + TRUE_B)
    ys_pred = model(xs)

    plt.plot(xs, ys_true, label="True")
    plt.plot(xs, ys_pred, label="Predicted", linestyle="dashed")
    plt.legend()
    plt.savefig("cosine_plot.png")


if __name__ == "__main__":
    main()
