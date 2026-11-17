import pandas as pd
import matplotlib.pyplot as plt

# Путь к файлу Excel
file_path = r'C:\Users\antiRasssist\Downloads\данные.xlsx'

# --- Лист1 (Инвертирующий) ---
# DC данные: строки 2-3 (заголовок в строке 1)
inverting_dc = pd.read_excel(file_path, sheet_name='Лист1', header=1, usecols='B:T', nrows=1)
inverting_dc = inverting_dc.melt(var_name='Uвх, В', value_name='Uвых, В').dropna()
inverting_dc['Uвх, В'] = pd.to_numeric(inverting_dc['Uвх, В'], errors='coerce')

# AC данные: строки 7-8 (заголовок в строке 7, значения в строке 8)
ac_raw = pd.read_excel(file_path, sheet_name='Лист1', header=6, usecols='B:L', nrows=2)
ac_raw = ac_raw.transpose()
ac_raw.columns = ['Uвх, В', 'Uвых, В']
ac_raw = ac_raw.dropna()
ac_raw['Uвх, В'] = pd.to_numeric(ac_raw['Uвх, В'], errors='coerce')
ac_raw['Uвых, В'] = pd.to_numeric(ac_raw['Uвых, В'], errors='coerce')
inverting_ac = ac_raw.reset_index(drop=True)

# АЧХ данные: строки 14-15 (заголовок в строке 14, значит header=13)
inverting_freq = pd.read_excel(file_path, sheet_name='Лист1', header=13, usecols='B:K', nrows=1)
inverting_freq = inverting_freq.melt(var_name='f, Гц', value_name='Uвых, В').dropna()
inverting_freq['f, Гц'] = pd.to_numeric(inverting_freq['f, Гц'], errors='coerce')

# --- Лист2 (Неинвертирующий) ---
# DC данные: строки 2-3 (заголовок в строке 1)
noninverting_dc = pd.read_excel(file_path, sheet_name='Лист2', header=1, usecols='B:T', nrows=1)
noninverting_dc = noninverting_dc.melt(var_name='Uвх, В', value_name='Uвых, В').dropna()
noninverting_dc['Uвх, В'] = pd.to_numeric(noninverting_dc['Uвх, В'], errors='coerce')

# AC данные: строки 7-8 (заголовок в строке 7, значения в строке 8)
ac_raw2 = pd.read_excel(file_path, sheet_name='Лист2', header=6, usecols='B:L', nrows=2)
ac_raw2 = ac_raw2.transpose()
ac_raw2.columns = ['Uвх, В', 'Uвых, В']
ac_raw2 = ac_raw2.dropna()
ac_raw2['Uвх, В'] = pd.to_numeric(ac_raw2['Uвх, В'], errors='coerce')
ac_raw2['Uвых, В'] = pd.to_numeric(ac_raw2['Uвых, В'], errors='coerce')
noninverting_ac = ac_raw2.reset_index(drop=True)

# АЧХ данные: строки 14-15 (заголовок в строке 14, значит header=13)
noninverting_freq = pd.read_excel(file_path, sheet_name='Лист2', header=13, usecols='B:K', nrows=1)
noninverting_freq = noninverting_freq.melt(var_name='f, Гц', value_name='Uвых, В').dropna()
noninverting_freq['f, Гц'] = pd.to_numeric(noninverting_freq['f, Гц'], errors='coerce')

# --- Построение графиков ---
plt.figure(figsize=(18, 6))

# 1. DC Амплитудная характеристика
plt.subplot(1, 3, 1)
plt.plot(inverting_dc['Uвх, В'], inverting_dc['Uвых, В'], label='Инвертирующий', color='blue')
plt.plot(noninverting_dc['Uвх, В'], noninverting_dc['Uвых, В'], label='Неинвертирующий', color='green', linestyle='--')
plt.xlabel('Uвх, В')
plt.ylabel('Uвых, В')
plt.title('DC Амплитудная характеристика')
plt.legend()
plt.grid(True)

# 2. AC Амплитудная характеристика
plt.subplot(1, 3, 2)
plt.plot(inverting_ac['Uвх, В'], inverting_ac['Uвых, В'], label='Инвертирующий', color='blue')
plt.plot(noninverting_ac['Uвх, В'], noninverting_ac['Uвых, В'], label='Неинвертирующий', color='green', linestyle='--')
plt.xlabel('Uвх, В')
plt.ylabel('Uвых, В')
plt.title('AC Амплитудная характеристика')
plt.legend()
plt.grid(True)

# 3. АЧХ
plt.subplot(1, 3, 3)
plt.plot(inverting_freq['f, Гц'], inverting_freq['Uвых, В'], label='Инвертирующий', color='blue')
plt.plot(noninverting_freq['f, Гц'], noninverting_freq['Uвых, В'], label='Неинвертирующий', color='green', linestyle='--')
plt.xlabel('f, Гц')
plt.ylabel('Uвых, В')
plt.title('Амплитудно-частотная характеристика (АЧХ)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()