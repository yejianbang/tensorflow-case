import tensorflow as tf
import numpy as np
import time
import os

# 🔧 设置 TensorFlow 多线程参数（按你的 CPU 核心数配置）
tf.config.threading.set_intra_op_parallelism_threads(16)  # 同一操作的线程数
tf.config.threading.set_inter_op_parallelism_threads(4)   # 不同操作间的线程数

# ✅ 更深更复杂的 CNN 模型
def create_big_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(28, 28, 1)))

    # 堆叠更多的 Conv + BN + ReLU + Dropout 层
    for filters in [32, 64, 128]:
        model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    for _ in range(3):  # 3 个 Dense 层
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

# ✅ 加载 MNIST 数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

# ✅ 构建模型、定义损失和优化器
model = create_big_model()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# ✅ 自定义训练 step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ⏱️ 开始训练计时
start_train = time.time()
EPOCHS = 5

for epoch in range(EPOCHS):
    print(f"\n📦 Epoch {epoch+1}/{EPOCHS}")
    for step, (x_batch, y_batch) in enumerate(train_ds):
        loss = train_step(x_batch, y_batch)
        if step % 100 == 0:
            print(f"  Step {step}, Loss: {loss.numpy():.4f}")

end_train = time.time()
print(f"\n✅ Training completed in {end_train - start_train:.2f} seconds")

# ⏱️ Evaluation
start_eval = time.time()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

for x_batch, y_batch in test_ds:
    preds = model(x_batch, training=False)
    accuracy.update_state(y_batch, preds)

end_eval = time.time()
print(f"✅ Evaluation accuracy: {accuracy.result().numpy():.4f}")
print(f"✅ Evaluation time: {end_eval - start_eval:.2f} seconds")

# ⏱️ Prediction test
start_pred = time.time()
model.predict(x_test[:1000])
end_pred = time.time()
print(f"✅ Prediction time for 1000 samples: {end_pred - start_pred:.2f} seconds")
