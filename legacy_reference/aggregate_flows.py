import pandas as pd
import numpy as np

# Путь к вашему .csv из tshark
CSV_PATH = 'S:/PythonProject/packets.csv'  # замените на свой путь
N_RESULTS = 100

# Чтение .csv с пропуском битых строк
df = pd.read_csv(CSV_PATH, sep=',', engine='python', on_bad_lines='skip')

# Приведение типов
for col in ['tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport', 'frame.len']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Время в pandas.Timestamp
if 'frame.time' in df.columns:
    df['frame.time'] = pd.to_datetime(df['frame.time'], errors='coerce')

# Ключ потока: src_ip, dst_ip, src_port, dst_port, protocol
def get_flow_key(row):
    proto = row['_ws.col.Protocol'] if '_ws.col.Protocol' in row else 'NONE'
    src_ip = row['ip.src'] if 'ip.src' in row else '0.0.0.0'
    dst_ip = row['ip.dst'] if 'ip.dst' in row else '0.0.0.0'
    # TCP/UDP порты, если есть
    if proto == 'TCP':
        src_port = row['tcp.srcport']
        dst_port = row['tcp.dstport']
    elif proto == 'UDP':
        src_port = row['udp.srcport']
        dst_port = row['udp.dstport']
    else:
        src_port = 0
        dst_port = 0
    return (src_ip, dst_ip, src_port, dst_port, proto)

df['flow_key'] = df.apply(get_flow_key, axis=1)

# Группировка по потокам
flows = df.groupby('flow_key')

# Агрегация признаков для каждого потока
def flow_features(group):
    times = group['frame.time'].dropna()
    duration = (times.max() - times.min()).total_seconds() if len(times) > 1 else 0
    sizes = group['frame.len'].dropna()
    return pd.Series({
        'total_packets': len(group),
        'total_bytes': sizes.sum(),
        'mean_pkt_size': sizes.mean(),
        'min_pkt_size': sizes.min(),
        'max_pkt_size': sizes.max(),
        'flow_duration': duration,
        'protocol': group['_ws.col.Protocol'].iloc[0] if '_ws.col.Protocol' in group else 'NONE',
        'src_ip': group['ip.src'].iloc[0] if 'ip.src' in group else '',
        'dst_ip': group['ip.dst'].iloc[0] if 'ip.dst' in group else '',
        'src_port': group['tcp.srcport'].iloc[0] if 'tcp.srcport' in group else (group['udp.srcport'].iloc[0] if 'udp.srcport' in group else 0),
        'dst_port': group['tcp.dstport'].iloc[0] if 'tcp.dstport' in group else (group['udp.dstport'].iloc[0] if 'udp.dstport' in group else 0),
    })

agg_df = flows.apply(flow_features).reset_index(drop=True)

# Топ-100 потоков по количеству байт
top_flows = agg_df.sort_values('total_bytes', ascending=False).head(N_RESULTS)

print('Топ-100 потоков по объёму данных:')
for i, row in top_flows.iterrows():
    src_port = int(row['src_port']) if not pd.isna(row['src_port']) else 0
    dst_port = int(row['dst_port']) if not pd.isna(row['dst_port']) else 0
    print(f"Поток {i+1}: {row['src_ip']}:{src_port} -> {row['dst_ip']}:{dst_port} | Протокол: {row['protocol']}")
    print(f"  Пакетов: {int(row['total_packets'])}, Байт: {int(row['total_bytes'])}, Длительность: {row['flow_duration']:.3f} сек, Средний размер: {row['mean_pkt_size']:.1f}")
    print('-'*50)