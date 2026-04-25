import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import Xception

# =========================
# PATHS
# =========================
train_dir = "data/train"
test_dir = "data/test"

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

base_train = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128,128),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical"
)

# =========================
# MIXUP AUGMENTATION
# =========================
def mixup(generator, alpha=0.2):
    while True:
        x, y = next(generator)
        lam = np.random.beta(alpha, alpha)
        index = np.random.permutation(len(x))
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        yield mixed_x, mixed_y

train_data = mixup(base_train)

# =========================
# CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(base_train.classes),
    y=base_train.classes
)
class_weights = dict(enumerate(class_weights))

# =========================
# FOCAL LOSS
# =========================
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    return loss

# =========================
# MODEL (XCEPTION)
# =========================
base_model = Xception(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(7, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

# =========================
# CALLBACKS
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

def lr_schedule(epoch):
    return 1e-4 if epoch < 10 else 1e-5

lr_scheduler = LearningRateScheduler(lr_schedule)

# =========================
# STEP 1: INITIAL TRAINING
# =========================
model.compile(
    optimizer=Adam(),
    loss=focal_loss(),
    metrics=['accuracy']
)

print("\n🚀 Initial Training Started...")

history = model.fit(
    train_data,
    steps_per_epoch=len(base_train),
    validation_data=test_data,
    epochs=15,
    callbacks=[early_stop, lr_scheduler],
    class_weight=class_weights
)

# =========================
# STEP 2: FINE-TUNING
# =========================
print("\n🔥 Fine-Tuning Started...")

base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=['accuracy']
)

history_fine = model.fit(
    train_data,
    steps_per_epoch=len(base_train),
    validation_data=test_data,
    epochs=10,
    class_weight=class_weights
)

# =========================
# FINAL RESULTS
# =========================

# Training accuracy (last epoch)
final_train_acc = history_fine.history['accuracy'][-1]

# Validation accuracy (last epoch)
final_val_acc = history_fine.history['val_accuracy'][-1]

# Reset test generator before evaluation
test_data.reset()

# Test accuracy (true performance)
test_loss, test_acc = model.evaluate(test_data)

print("\n📊 FINAL RESULTS")
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# =========================
# SAVE MODEL
# =========================
model.save("models/emotion_model.h5")

print("✅ Model trained and saved!")