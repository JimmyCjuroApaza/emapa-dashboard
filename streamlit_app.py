import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
#import folium as folium
import pydeck as pdk
# ----------------------------------------------
# Algoritmo de detección de bloques (exactamente como provisto)
# ----------------------------------------------

def unir_h(dia_promedio):
    """
    Rellena huecos de hasta 3 intervalos de 10 minutos (30 min) entre bloques activos (1)
    en la columna 'indicador_final', considerando continuidad circular en el día.
    """
    dia_promedio = dia_promedio.sort_values('hora')
    valores = dia_promedio['indicador_final'].values.copy()
    n = len(valores)
    valores_extended = np.concatenate([valores, valores[:3]])
    for i in range(n):
        if valores_extended[i] == 1:
            j = i + 1
            count_zeros = 0
            while j < i + 4 and valores_extended[j] == 0:
                count_zeros += 1
                j += 1
                if 1 <= count_zeros <= 3 and valores_extended[j % n] == 1:
                    for k in range(i + 1, i + 1 + count_zeros):
                        valores[k % n] = 1
    return valores


def detectar_intervalos(dia_promedio):
    """
    Detecta y devuelve los intervalos horarios donde el 'indicador_final' es 1,
    agrupando periodos contiguos en el día.
    """
    dia_promedio = dia_promedio.copy()
    dia_promedio['hora_dt'] = pd.to_datetime(dia_promedio['hora'], format='%H:%M')
    dia_promedio['hora_time'] = dia_promedio['hora_dt'].dt.time
    dia_promedio['grupo'] = (dia_promedio['indicador_final'].diff() != 0).cumsum()
    intervalos = []
    for _, g in dia_promedio.groupby('grupo'):
        if g['indicador_final'].iloc[0] == 1:
            h_inicio = g['hora_time'].iloc[0]
            h_fin = g['hora_time'].iloc[-1]
            intervalos.append((h_inicio, h_fin))
    return intervalos


def duracion_en_minutos(start, end):
    start_dt = datetime.combine(datetime.today(), start)
    end_dt = datetime.combine(datetime.today(), end)
    if end_dt < start_dt:
        end_dt += timedelta(days=1)
    return (end_dt - start_dt).total_seconds() / 60


def total_minutos(intervalos):
    total = 0
    for a, b in intervalos:
        total += duracion_en_minutos(a, b)
    return total


def mejor_par(intervalos):
    if len(intervalos) <= 2:
        return intervalos
    duraciones = [(i, duracion_en_minutos(a, b)) for i, (a, b) in enumerate(intervalos)]
    mayores = sorted(duraciones, key=lambda x: x[1], reverse=True)[:2]
    return [intervalos[i] for i, _ in mayores]


def reducir(intervalos, dia_promedio, umbral):
    dia_promedio['hora_dt'] = pd.to_datetime(dia_promedio['hora'], format='%H:%M')
    dia_promedio['hora_time'] = dia_promedio['hora_dt'].dt.time
    total_min1 = total_minutos(intervalos)
    while len(intervalos) > 2:
        min_gap = np.inf
        pares_min = []
        for i in range(len(intervalos)):
            a_end = intervalos[i][1]
            for j in range(i+1, len(intervalos)):
                b_start = intervalos[j][0]
                gap = duracion_en_minutos(a_end, b_start)
                if gap < min_gap:
                    min_gap = gap
                    pares_min = [(i, j)]
                elif gap == min_gap:
                    pares_min.append((i, j))
        if not pares_min:
            break
        fusion = False
        for i, j in pares_min:
            a_end = intervalos[i][1]
            b_start = intervalos[j][0]
            def in_gap(t):
                if a_end < b_start:
                    return a_end < t < b_start
                return t > a_end or t < b_start
            hueco = dia_promedio[dia_promedio['hora_time'].apply(in_gap)]
            if (hueco['indicador'] >= umbral).mean() > 0.7:
                mask = dia_promedio['hora_time'].apply(in_gap)
                dia_promedio.loc[mask, 'indicador_final'] = 1
                fusion = True
        dia_promedio['indicador_final'] = unir_h(dia_promedio)
        intervalos = detectar_intervalos(dia_promedio)
        intervalos = mejor_par(intervalos)
        if not fusion:
            break
    intervalos = mejor_par(intervalos)
    if len(intervalos) == 2:
        durs = [duracion_en_minutos(a, b) for a, b in intervalos]
        if durs[0] < 15:
            intervalos = [intervalos[1]]
        elif durs[1] < 15:
            intervalos = [intervalos[0]]
    total_min2 = total_minutos(intervalos)
    alerta = (total_min1 - 60) > total_min2
    return intervalos, alerta


def redondear_abajo(hora):
    m = (hora.minute // 10) * 10
    h = hora.hour + (m // 60)
    return time(h%24, m%60)


def redondear_arriba(hora):
    m = ((hora.minute + 9) // 10) * 10
    h = hora.hour + (m // 60)
    return time(h%24, m%60)


def mostrar_horarios(intervalos):
    if len(intervalos) == 2 and (intervalos[0][1] == time(0) or intervalos[0][1] > intervalos[1][0]):
        intervalos = intervalos[::-1]
    salida = []
    for inicio, fin in intervalos:
        i_r = redondear_abajo(inicio)
        f_r = redondear_arriba(fin)
        ini_str = i_r.strftime('%H:%M').lstrip('0') if i_r.hour!=0 else '0'+i_r.strftime(':%M')
        fin_str = '24:00' if f_r==time(0) else (f_r.strftime('%H:%M').lstrip('0') if f_r.hour!=0 else '0'+f_r.strftime(':%M'))
        salida.append(f"{ini_str} A {fin_str}")
    return ' - '.join(salida)


def horario(logger, df_presion, fecha_inicio, fecha_final, umbral_1=0.7, umbral_2=0.5):
    caso = df_presion[(df_presion['fecha_dia']>=fecha_inicio) &
                      (df_presion['fecha_dia']<=fecha_final) &
                      (df_presion['alias']==logger)].copy()
    if caso.empty:
        return None, 'Datalogger inactivo', 0, False
    caso['hora'] = caso['d_time'].dt.strftime('%H:%M')
    caso['indicador'] = (caso['presion_mca']>5).astype(float)
    dia_prom = caso.groupby('hora')['indicador'].mean().reset_index()
    dia_prom['indicador_final'] = unir_h(dia_prom.assign(indicador_final=(dia_prom['indicador']>umbral_1).astype(float)))
    intervalos = detectar_intervalos(dia_prom)
    intervalos, alerta = reducir(intervalos, dia_prom, umbral_2)
    sched = mostrar_horarios(intervalos) if intervalos else None
    mins = total_minutos(intervalos)
    return dia_prom, sched, mins, alerta

# ----------------------------------------------
# Carga de datos
# ----------------------------------------------
@st.cache_data
def load_data():
    presion = pd.read_csv('datos_presion.csv')
    presion.rename(columns={'time_stamp':'d_time'}, inplace=True)
    presion['d_time'] = pd.to_datetime(presion['d_time'], errors='coerce')
    presion['fecha_dia'] = presion['d_time'].dt.floor('D')
    presion['presion_mca'] = presion['value'] * 10.2

    conex = pd.read_csv('conex_log_df.csv')
    ubi   = pd.read_csv('ubi_df.csv')
    # Asegurar coordenadas numéricas
    ubi['latitude']  = pd.to_numeric(ubi['latitude'], errors='coerce')
    ubi['longitude'] = pd.to_numeric(ubi['longitude'], errors='coerce')
    ubi = ubi.dropna(subset=['latitude','longitude'])
    return presion, conex, ubi

# Inicializar datos
df_presion, df_conex, df_ubi = load_data()

# ----------------------------------------------
# Interfaz Streamlit
# ----------------------------------------------
st.title("Horario de Abastecimiento EMAPA San Martin")

# Barra lateral: filtros
st.sidebar.header("Filtros")
# Rango de fechas
fecha_inicio = st.sidebar.date_input("Fecha inicio", value=datetime.today() - timedelta(days=30))
fecha_final  = st.sidebar.date_input("Fecha fin",    value=datetime.today())
fecha_inicio = pd.to_datetime(fecha_inicio)
fecha_final  = pd.to_datetime(fecha_final)
if fecha_final < fecha_inicio:
    st.sidebar.error("Fecha fin debe ser posterior a Fecha inicio")

# Selector de datalogger
alias_list = sorted(df_ubi['alias'].unique())
logger_sel = st.sidebar.selectbox("Datalogger", alias_list, index=0)

# Cálculo de horario
if logger_sel:
    dia_prom, sched, minutos, alerta = horario(logger_sel, df_presion, fecha_inicio, fecha_final)
    num_conex = (df_conex['datalogger']==logger_sel).sum()
else:
    dia_prom, sched, minutos, alerta = None, None, 0, False
    num_conex = 0

# Métricas
st.metric("Número de conexiones", num_conex)
st.metric("Bloque horario", sched or "No activo")
horas = minutos / 60
st.metric("Horas estimadas", f"{horas:.2f} h")

# Mostrar coordenadas
if logger_sel:
    loc = df_ubi[df_ubi['alias']==logger_sel][['latitude','longitude']]
    if not loc.empty:
        lat, lon = loc.iloc[0]
        st.write(f"**Coordenadas:** Latitud: {lat:.6f}, Longitud: {lon:.6f}")
    else:
        st.write("**Coordenadas:** No disponibles")

# Mapa con Pydeck: todos los dataloggers, resaltando el seleccionado
df_map = df_ubi.assign(
    color=df_ubi['alias'].apply(lambda x: [255, 0, 0] if x == logger_sel else [0, 128, 255])
)
layer = pdk.Layer(
    'ScatterplotLayer',
    data=df_map,
    get_position='[longitude, latitude]',
    get_fill_color='color',
    get_radius=100,
    pickable=True,
)
# Centrar automáticamente en el logger seleccionado
def obtener_centro():
    if logger_sel:
        sel = df_ubi[df_ubi['alias'] == logger_sel]
        if not sel.empty:
            return sel.iloc[0][['latitude', 'longitude']].tolist()
    return [df_ubi['latitude'].mean(), df_ubi['longitude'].mean()]

centro = obtener_centro()
view_state = pdk.ViewState(
    latitude=centro[0],
    longitude=centro[1],
    zoom=12,
)
r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={'text': '{alias}'})
st.pydeck_chart(r)

# Serie de tiempo de 0-24h
if isinstance(dia_prom, pd.DataFrame):
    dia_prom['hora_dt'] = pd.to_datetime(dia_prom['hora'], format='%H:%M')
    import altair as alt
    chart = alt.Chart(dia_prom).mark_line().encode(
        x='hora_dt:T',
        y='indicador:Q'
    ).properties(width=700, height=300)
    st.altair_chart(chart)

st.markdown("---")
st.caption(f"Periodo analizado: {fecha_inicio.date()} a {fecha_final.date()}")
