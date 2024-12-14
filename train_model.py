# Importar librerías necesarias
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import matplotlib.pyplot as plt
import json

train_dir = 'Datasets/OCT2017/train'
val_dir = 'Datasets/OCT2017/val'
output_model_path = 'modelo/modelo_retiniano_gpu.h5'

datagen_train = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen_val = ImageDataGenerator(rescale=1.0/255)

train_generator = datagen_train.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = datagen_val.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Construcción del modelo utilizando MobileNet como base
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

csv_logger = CSVLogger('training_log_gpu.csv', append=True)
checkpoint = ModelCheckpoint(
    filepath=output_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[csv_logger, checkpoint]
)

# Guardar el modelo final
model.save(output_model_path)
print(f"Modelo guardado como '{output_model_path}'")

with open('train_history_gpu.json', 'w') as f:
    json.dump(history.history, f)

history_dict = history.history

#Precision
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Precisión de entrenamiento')
plt.plot(history_dict['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión de Entrenamiento y Validación')

#perdida
plt.subplot(1, 2, 2)
plt.plot(history_dict['loss'], label='Pérdida de entrenamiento')
plt.plot(history_dict['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida de Entrenamiento y Validación')

plt.tight_layout()
plt.show()
