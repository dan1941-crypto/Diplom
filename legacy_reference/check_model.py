import tensorflow as tf, joblib, numpy as np

m = tf.keras.models.load_model('my_model.h5')
print('MODEL_INPUT_SHAPE:', m.input_shape, 'NUM_OUTPUTS:', len(m.outputs))

sc = joblib.load('scaler.save')
n = getattr(sc, 'n_features_in_', None)
print('SCALER_FEATURES:', n)

cols = [x for x in open('feature_names.txt','r',encoding='utf-8').read().strip().split(',') if x and x!=' Label']
print('FEATURE_NAMES_COUNT:', len(cols))
print('MATCH_SCALER_FEATURES:', (n == len(cols)))

X = np.zeros((1, n), dtype='float32')
outs = m.predict(X, verbose=0)
outs = outs if isinstance(outs, (list, tuple)) else [outs]
print('PREDICT_OUTPUT_SHAPES:', [o.shape for o in outs])