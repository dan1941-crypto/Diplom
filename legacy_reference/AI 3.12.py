import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import joblib

# Пути к файлам
files = [
    r'S:\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    r'S:\MachineLearningCVE\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    r'S:\MachineLearningCVE\Friday-WorkingHours-Morning.pcap_ISCX.csv',
    r'S:\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv',
    r'S:\MachineLearningCVE\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    r'S:\MachineLearningCVE\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    r'S:\MachineLearningCVE\Tuesday-WorkingHours.pcap_ISCX.csv',
    r'S:\MachineLearningCVE\Wednesday-workingHours.pcap_ISCX.csv'
]

# === 0. Собираем все уникальные метки для LabelEncoder ===
all_labels = []
for file in files:
    df = pd.read_csv(file, usecols=[' Label'])
    all_labels.extend(df[' Label'].astype(str).unique())
all_labels = np.unique(all_labels)

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
joblib.dump(label_encoder, 'label_encoder.save')

# === 1. ОБУЧЕНИЕ НА ПЕРВОМ ФАЙЛЕ ===
data = pd.read_csv(files[0])
data = data.replace([np.inf, -np.inf], np.nan).dropna()
data['is_attack'] = data[' Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
data['traffic_type'] = label_encoder.transform(data[' Label'].astype(str))

features = data.drop([' Label', 'is_attack', 'traffic_type'], axis=1)
features = features.apply(pd.to_numeric, errors='coerce')
features = features.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')

scaler = StandardScaler()
X = scaler.fit_transform(features)
y_type = data['traffic_type'].values
y_attack = data['is_attack'].values

joblib.dump(scaler, 'scaler.save')

X_train, X_test, y_type_train, y_type_test, y_attack_train, y_attack_test = train_test_split(
    X, y_type, y_attack, test_size=0.3, random_state=42
)

input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)

inputs = tf.keras.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)

output_type = tf.keras.layers.Dense(num_classes, activation='softmax', name='traffic_type')(x)
output_attack = tf.keras.layers.Dense(1, activation='sigmoid', name='attack_detection')(x)

model = tf.keras.Model(inputs=inputs, outputs=[output_type, output_attack])

model.compile(
    optimizer='adam',
    loss={
        'traffic_type': 'sparse_categorical_crossentropy',
        'attack_detection': 'binary_crossentropy'
    },
    metrics={
        'traffic_type': 'accuracy',
        'attack_detection': 'accuracy'
    }
)

history = model.fit(
    X_train,
    {
        'traffic_type': y_type_train,
        'attack_detection': y_attack_train
    },
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

model.save('my_model.keras')  # Используем современный формат

# === 2. ПОСЛЕДОВАТЕЛЬНОЕ ДОУЧИВАНИЕ НА ОСТАЛЬНЫХ ФАЙЛАХ ===
scaler = joblib.load('scaler.save')
label_encoder = joblib.load('label_encoder.save')
model = tf.keras.models.load_model('my_model.keras')

for file in files[1:]:
    print(f"\n=== Дообучение на файле: {file} ===")
    new_data = pd.read_csv(file)
    new_data = new_data.replace([np.inf, -np.inf], np.nan).dropna()
    new_data['is_attack'] = new_data[' Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
    new_data['traffic_type'] = label_encoder.transform(new_data[' Label'].astype(str))

    new_features = new_data.drop([' Label', 'is_attack', 'traffic_type'], axis=1)
    new_features = new_features.apply(pd.to_numeric, errors='coerce')
    new_features = new_features.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')

    X_new = scaler.transform(new_features)
    y_type_new = new_data['traffic_type'].values
    y_attack_new = new_data['is_attack'].values

    model.fit(
        X_new,
        {
            'traffic_type': y_type_new,
            'attack_detection': y_attack_new
        },
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

model.save('my_model_finetuned.keras')

# === 3. ОЦЕНКА НА ПОСЛЕДНЕМ ФАЙЛЕ ===
test_loss = model.evaluate(X_new, {
    'traffic_type': y_type_new,
    'attack_detection': y_attack_new
})
print(f"Test Accuracy (Traffic Type): {test_loss[3] * 100:.2f}%")
print(f"Test Accuracy (Attack Detection): {test_loss[4] * 100:.2f}%")

# === 4. ВИЗУАЛИЗАЦИЯ ===
plt.plot(history.history['traffic_type_accuracy'], label='Traffic Type Train')
plt.plot(history.history['val_traffic_type_accuracy'], label='Traffic Type Val')
plt.plot(history.history['attack_detection_accuracy'], label='Attack Detection Train')
plt.plot(history.history['val_attack_detection_accuracy'], label='Attack Detection Val')
plt.legend()
plt.show()