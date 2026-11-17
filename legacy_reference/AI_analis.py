import pyshark
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from collections import defaultdict, Counter

# Пути к файлам
MODEL_PATH = 'S:/PythonProject/my_model.h5'
SCALER_PATH = 'S:/PythonProject/scaler.save'
ENCODER_PATH = 'S:/PythonProject/label_encoder.save'
PCAP_PATH = 'S:/PythonProject/test.pcapng'

# Количество результатов для вывода (None = все)
N_RESULTS = None

# Загрузка модели и препроцессоров
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Функция для агрегации пакетов в потоки
# Ключ потока: (src_ip, dst_ip, src_port, dst_port, protocol)
def get_flow_key(pkt):
    try:
        proto = pkt.transport_layer if hasattr(pkt, 'transport_layer') else 'NONE'
        src_ip = pkt.ip.src if hasattr(pkt, 'ip') else '0.0.0.0'
        dst_ip = pkt.ip.dst if hasattr(pkt, 'ip') else '0.0.0.0'
        src_port = pkt[pkt.transport_layer].srcport if proto in pkt else 0
        dst_port = pkt[pkt.transport_layer].dstport if proto in pkt else 0
        return (src_ip, dst_ip, src_port, dst_port, proto)
    except Exception:
        return None

# Сбор потоков
class OnlineStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = None
        self.max = None

    def add(self, x: float):
        self.count += 1
        if self.min is None or x < self.min:
            self.min = x
        if self.max is None or x > self.max:
            self.max = x
        delta = x - self.mean
        self.mean += delta * (1.0 / self.count)
        delta2 = x - self.mean
        self.M2 += delta2 * delta

    @property
    def variance(self):
        return self.M2 / self.count if self.count > 0 else 0.0

    @property
    def std(self):
        return self.variance ** 0.5

class FlowAgg:
    def __init__(self, first_src_ip: str):
        self.first_src_ip = first_src_ip
        self.first_time = None
        self.last_time = None
        self.total_packets = 0
        self.total_bytes = 0
        # overall sizes
        self.size_stats = OnlineStats()
        # directional sizes
        self.fwd_count = 0
        self.bwd_count = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.fwd_size_stats = OnlineStats()
        self.bwd_size_stats = OnlineStats()
        # IAT
        self.prev_time = None
        self.iat_stats = OnlineStats()
        # dir IAT
        self.prev_time_fwd = None
        self.prev_time_bwd = None
        self.fwd_iat_stats = OnlineStats()
        self.bwd_iat_stats = OnlineStats()
        # TCP flags
        self.flag_counts = Counter()
        # protocol
        self.proto = 'NONE'

    def add_packet(self, pkt):
        try:
            t = float(pkt.sniff_timestamp)
        except Exception:
            return
        try:
            sz = int(pkt.length)
        except Exception:
            sz = 0

        # times
        if self.first_time is None:
            self.first_time = t
        self.last_time = t

        # overall counts
        self.total_packets += 1
        self.total_bytes += sz
        self.size_stats.add(sz)

        # protocol
        self.proto = pkt.transport_layer if hasattr(pkt, 'transport_layer') else self.proto

        # direction
        this_src_ip = pkt.ip.src if hasattr(pkt, 'ip') else '0.0.0.0'
        is_fwd = (this_src_ip == self.first_src_ip)
        if is_fwd:
            self.fwd_count += 1
            self.fwd_bytes += sz
            self.fwd_size_stats.add(sz)
        else:
            self.bwd_count += 1
            self.bwd_bytes += sz
            self.bwd_size_stats.add(sz)

        # IAT overall
        if self.prev_time is not None:
            self.iat_stats.add(t - self.prev_time)
        self.prev_time = t

        # IAT directional
        if is_fwd:
            if self.prev_time_fwd is not None:
                self.fwd_iat_stats.add(t - self.prev_time_fwd)
            self.prev_time_fwd = t
        else:
            if self.prev_time_bwd is not None:
                self.bwd_iat_stats.add(t - self.prev_time_bwd)
            self.prev_time_bwd = t

        # TCP flags
        if hasattr(pkt, 'tcp'):
            flags = pkt.tcp.flags if hasattr(pkt.tcp, 'flags') else ''
            if '0x0001' in flags: self.flag_counts['FIN'] += 1
            if '0x0002' in flags: self.flag_counts['SYN'] += 1
            if '0x0004' in flags: self.flag_counts['RST'] += 1
            if '0x0008' in flags: self.flag_counts['PSH'] += 1
            if '0x0010' in flags: self.flag_counts['ACK'] += 1
            if '0x0020' in flags: self.flag_counts['URG'] += 1
            if '0x0080' in flags: self.flag_counts['CWR'] += 1
            if '0x0040' in flags: self.flag_counts['ECE'] += 1

def extract_flows(pcap_path):
    cap = pyshark.FileCapture(pcap_path, only_summaries=False, keep_packets=False)
    aggs = {}
    for pkt in tqdm(cap, desc="Обработка пакетов", unit="pkt"):
        key = get_flow_key(pkt)
        if not key:
            continue
        src_ip = key[0]
        if key not in aggs:
            aggs[key] = FlowAgg(first_src_ip=src_ip)
        aggs[key].add_packet(pkt)
    cap.close()
    return aggs

# Извлечение признаков для потока (демо-набор)
def flow_features_from_agg(key, agg: FlowAgg):
    duration = (agg.last_time - agg.first_time) if (agg.first_time is not None and agg.last_time is not None) else 0.0
    duration_pos = duration if duration > 0 else 1e-9

    # Overall packet length stats
    pkt_len_min = agg.size_stats.min or 0
    pkt_len_max = agg.size_stats.max or 0
    pkt_len_mean = agg.size_stats.mean or 0
    pkt_len_std = agg.size_stats.std or 0
    pkt_len_var = agg.size_stats.variance or 0

    # Directional packet length stats
    fwd_len_max = agg.fwd_size_stats.max or 0
    fwd_len_min = agg.fwd_size_stats.min or 0
    fwd_len_mean = agg.fwd_size_stats.mean or 0
    fwd_len_std = agg.fwd_size_stats.std or 0

    bwd_len_max = agg.bwd_size_stats.max or 0
    bwd_len_min = agg.bwd_size_stats.min or 0
    bwd_len_mean = agg.bwd_size_stats.mean or 0
    bwd_len_std = agg.bwd_size_stats.std or 0

    # IAT
    flow_iat_mean = agg.iat_stats.mean or 0
    flow_iat_std = agg.iat_stats.std or 0
    flow_iat_max = agg.iat_stats.max or 0
    flow_iat_min = agg.iat_stats.min or 0

    fwd_iat_total = (agg.fwd_iat_stats.mean * agg.fwd_iat_stats.count) if agg.fwd_iat_stats.count else 0
    fwd_iat_mean = agg.fwd_iat_stats.mean or 0
    fwd_iat_std = agg.fwd_iat_stats.std or 0
    fwd_iat_max = agg.fwd_iat_stats.max or 0
    fwd_iat_min = agg.fwd_iat_stats.min or 0

    bwd_iat_total = (agg.bwd_iat_stats.mean * agg.bwd_iat_stats.count) if agg.bwd_iat_stats.count else 0
    bwd_iat_mean = agg.bwd_iat_stats.mean or 0
    bwd_iat_std = agg.bwd_iat_stats.std or 0
    bwd_iat_max = agg.bwd_iat_stats.max or 0
    bwd_iat_min = agg.bwd_iat_stats.min or 0

    # Rates
    flow_bytes_s = agg.total_bytes / duration_pos
    flow_pkts_s = agg.total_packets / duration_pos
    fwd_pkts_s = agg.fwd_count / duration_pos
    bwd_pkts_s = agg.bwd_count / duration_pos

    down_up_ratio = (agg.bwd_bytes / agg.fwd_bytes) if agg.fwd_bytes > 0 else 0
    avg_pkt_size = pkt_len_mean

    feature_dict = {
        ' Flow Duration': duration,
        ' Total Fwd Packets': agg.fwd_count,
        ' Total Backward Packets': agg.bwd_count,
        'Total Length of Fwd Packets': agg.fwd_bytes,
        ' Total Length of Bwd Packets': agg.bwd_bytes,
        ' Fwd Packet Length Max': fwd_len_max,
        ' Fwd Packet Length Min': fwd_len_min,
        ' Fwd Packet Length Mean': fwd_len_mean,
        ' Fwd Packet Length Std': fwd_len_std,
        'Bwd Packet Length Max': bwd_len_max,
        ' Bwd Packet Length Min': bwd_len_min,
        ' Bwd Packet Length Mean': bwd_len_mean,
        ' Bwd Packet Length Std': bwd_len_std,
        'Flow Bytes/s': flow_bytes_s,
        ' Flow Packets/s': flow_pkts_s,
        ' Flow IAT Mean': flow_iat_mean,
        ' Flow IAT Std': flow_iat_std,
        ' Flow IAT Max': flow_iat_max,
        ' Flow IAT Min': flow_iat_min,
        'Fwd IAT Total': fwd_iat_total,
        ' Fwd IAT Mean': fwd_iat_mean,
        ' Fwd IAT Std': fwd_iat_std,
        ' Fwd IAT Max': fwd_iat_max,
        ' Fwd IAT Min': fwd_iat_min,
        'Bwd IAT Total': bwd_iat_total,
        ' Bwd IAT Mean': bwd_iat_mean,
        ' Bwd IAT Std': bwd_iat_std,
        ' Bwd IAT Max': bwd_iat_max,
        ' Bwd IAT Min': bwd_iat_min,
        'Fwd Packets/s': fwd_pkts_s,
        ' Bwd Packets/s': bwd_pkts_s,
        ' Min Packet Length': pkt_len_min,
        ' Max Packet Length': pkt_len_max,
        ' Packet Length Mean': pkt_len_mean,
        ' Packet Length Std': pkt_len_std,
        ' Packet Length Variance': pkt_len_var,
        'FIN Flag Count': agg.flag_counts['FIN'],
        ' SYN Flag Count': agg.flag_counts['SYN'],
        ' RST Flag Count': agg.flag_counts['RST'],
        ' PSH Flag Count': agg.flag_counts['PSH'],
        ' ACK Flag Count': agg.flag_counts['ACK'],
        ' URG Flag Count': agg.flag_counts['URG'],
        ' CWE Flag Count': agg.flag_counts['CWR'],
        ' ECE Flag Count': agg.flag_counts['ECE'],
        ' Down/Up Ratio': down_up_ratio,
        ' Average Packet Size': avg_pkt_size,
        ' Avg Fwd Segment Size': fwd_len_mean,
        ' Avg Bwd Segment Size': bwd_len_mean,
        ' Fwd Header Length': 0,
        # Остальные поля заполним нулями ниже через выравнивание
    }
    return feature_dict

# Основной анализ
def main():
    flows = extract_flows(PCAP_PATH)
    features_list = []
    flow_keys = []
    iterable = list(flows.items()) if N_RESULTS is None else list(flows.items())[:N_RESULTS]
    for key, agg in iterable:
        feats = flow_features_from_agg(key, agg)
        features_list.append(feats)
        flow_keys.append(key)
    # === Жёсткое сопоставление признаков ===
    # Читаем точный список признаков из файла feature_names.txt
    with open('S:/PythonProject/feature_names.txt', 'r', encoding='utf-8') as f:
        feature_names_line = f.read().strip()
    feature_names = [x for x in feature_names_line.split(',') if x and x != ' Label']
    # Принудительно добавляем пробел к первому признаку, если его нет
    if not feature_names[0].startswith(' '):
        feature_names[0] = ' ' + feature_names[0]
    # Исправляем дубликат ' Fwd Header Length' -> второй экземпляр считаем ' Fwd Header Length.1'
    try:
        first_idx = feature_names.index(' Fwd Header Length')
        # Ищем второй экземпляр
        second_idx = feature_names.index(' Fwd Header Length', first_idx + 1)
        feature_names[second_idx] = ' Fwd Header Length.1'
    except ValueError:
        pass

    features_list_full = []
    for feats in features_list:
        feats_full = {}
        for fname in feature_names:
            if fname in feats:
                feats_full[fname] = feats[fname]
            elif fname == ' Fwd Header Length.1':
                feats_full[fname] = feats.get(' Fwd Header Length', 0)
            else:
                feats_full[fname] = 0
        features_list_full.append(feats_full)
    print('feature_names:', len(feature_names), feature_names)
    print('features_list_full[0]:', len(features_list_full[0]), list(features_list_full[0].keys()))
    features_df = pd.DataFrame(features_list_full)
    # Гарантируем наличие столбца-дубликата
    if ' Fwd Header Length.1' not in features_df.columns:
        features_df[' Fwd Header Length.1'] = 0
    features_df = features_df[feature_names]
    # Заполнение NaN нулями (если есть)
    features_df = features_df.fillna(0)
    # Приведение к нужному формату (добавьте/уберите признаки по необходимости)
    try:
        X = scaler.transform(features_df)
    except Exception as e:
        print('Ошибка при масштабировании признаков:', e)
        print('Проверьте, совпадает ли набор признаков с обучающим!')
        return
    # Предсказание
    y_pred_type, y_pred_attack = model.predict(X)
    y_pred_type = np.argmax(y_pred_type, axis=1)
    y_pred_attack = (y_pred_attack > 0.5).astype(int).flatten()
    # Декодирование типа трафика
    traffic_types = label_encoder.inverse_transform(y_pred_type)
    # Вывод только атакующих потоков
    total = len(flow_keys)
    attack_indices = [i for i, v in enumerate(y_pred_attack) if v == 1]
    for idx in attack_indices:
        key = flow_keys[idx]
        print(f'Поток {idx+1}: {key}')
        print(f'  Тип трафика: {traffic_types[idx]}')
        print(f'  Атака: ДА')
        print('-'*40)
    print(f'Найдено атак: {len(attack_indices)} из {total} потоков')

if __name__ == '__main__':
    main() 