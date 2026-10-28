import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import gspread
from google.oauth2.service_account import Credentials
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, exceptions
from scipy.stats import t as t_dist

# Sesión que imita a un navegador Chrome. Yahoo suele bloquear a los servidores
# (datacenter) de Streamlit Cloud; presentarse como navegador normal lo destraba.
@st.cache_resource
def _yf_session():
    try:
        from curl_cffi import requests as _cffi
        return _cffi.Session(impersonate="chrome")
    except Exception:
        return None

# --- 0. CONFIGURACIÓN Y ESTILO (INTERFAZ ORIGINAL COMPLETA) ---
st.set_page_config(page_title="Financial Wealth Optimizer Pro", layout="wide")

st.markdown("""
    <style>
    th { text-align: center !important; font-weight: bold !important; text-transform: uppercase; color: #1E88E5; }
    td { text-align: center !important; }
    .stMetric { background-color: #f8f9fb; padding: 15px; border-radius: 10px; border-left: 5px solid #1E88E5; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. TASA LIBRE DE RIESGO Y DESCARGA (CON CACHÉ PARA EVITAR BANEO) ---
def _bajar_yf(tk, start, end, sesion, threads):
    """Llamada cruda a yfinance, tolerante a versiones sin 'session'."""
    try:
        try:
            return yf.download(tk, start=start, end=end, auto_adjust=True,
                               progress=False, threads=threads, session=sesion)
        except TypeError:
            return yf.download(tk, start=start, end=end, auto_adjust=True,
                               progress=False, threads=threads)
    except Exception:
        return None

def _extraer_series(raw, tks):
    """De lo que devolvió yfinance, arma {ticker: serie_de_cierres} con lo que trajo datos."""
    out = {}
    if raw is None or getattr(raw, "empty", True):
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        campo = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        sub = raw[campo]
        for t in sub.columns:
            s = sub[t].dropna()
            if not s.empty:
                out[t] = s
    else:
        campo = "Adj Close" if "Adj Close" in raw.columns else "Close"
        s = raw[campo].dropna()
        if not s.empty:
            out[tks[0]] = s
    return out

@st.cache_data(ttl=3600)
def descargar_datos_robustos(tickers, start, end):
    """Baja los precios SIN ráfaga: primero un lote secuencial (threads=False) y,
    a los que fallan, los reintenta de a uno con pausas. Así se esquiva el límite
    de pedidos que Yahoo aplica a los servidores de la nube.
    Devuelve (data_valida, fallidos, motivo)."""
    tickers = list(tickers)
    sesion = _yf_session()

    # 1) Lote secuencial (gentil con el límite de Yahoo)
    series = _extraer_series(_bajar_yf(tickers, start, end, sesion, False), tickers)

    # 2) Los que no vinieron, uno por uno con backoff
    for t in [t for t in tickers if t not in series]:
        for intento in range(2):
            time.sleep(1.0 * (intento + 1))
            got = _extraer_series(_bajar_yf(t, start, end, sesion, False), [t])
            if got:
                series.update(got)
                break

    if not series:
        return None, tickers, ("Yahoo no devolvió datos para ningún ticker. "
                               "Suele ser el límite de pedidos a servidores de Streamlit Cloud.")

    data = pd.DataFrame(series).ffill().dropna()
    fallidos = [t for t in tickers if t not in series]
    if data.empty:
        return None, tickers, "Los tickers no comparten fechas en común en el rango elegido."
    return data, fallidos, ""

def obtener_risk_free_live():
    try:
        tnx = yf.Ticker("^TNX")
        data = tnx.history(period="1d")
        if data.empty: return 0.042
        return data['Close'].iloc[-1] / 100
    except:
        return 0.042

# Mapa de ESPECIE local (BYMA) -> ticker del ADR en Yahoo Finance.
# Solo para acciones argentinas que cotizan como ADR. Si una acción NO está
# en este mapa, se asume que no tiene ADR y se le agrega ".BA" (mercado local).
# Agregá o corregí entradas acá según lo que emita tu matriz.
ADR_ARG = {
    "GGAL": "GGAL", "BMA": "BMA", "PAMP": "PAM", "YPFD": "YPF",
    "TGSU2": "TGS", "CEPU": "CEPU", "CRES": "CRESY", "EDN": "EDN",
    "IRSA": "IRS", "LOMA": "LOMA", "SUPV": "SUPV", "TECO2": "TEO",
    "BBAR": "BBAR",
}

def especie_a_yahoo(especie, seccion):
    """Convierte la ESPECIE del estado de cuenta al ticker que entiende Yahoo Finance:
    - CEDEARS: el especie ya es el ticker del subyacente (EE.UU.), va tal cual.
    - ACCIONES con ADR: usa el ADR (ej. PAMP -> PAM).
    - ACCIONES sin ADR: agrega '.BA' para traer el dato local de BYMA."""
    especie = especie.strip().upper()
    if seccion == "ACCIONES":
        return ADR_ARG.get(especie, f"{especie}.BA")
    return especie  # CEDEARS (u otras secciones) van sin cambios

# Sugerencias para tickers que Yahoo no reconoce (ticker que falló -> ticker correcto).
# Lista curada base: solo equivalencias seguras. Se complementa con las que la app
# va APRENDIENDO sola cada vez que alguien corrige un ticker a mano y funciona.
SUGERENCIAS_TICKER = {
    "BRKB": "BRK-B",    # Berkshire Hathaway clase B
    "BRKA": "BRK-A",    # Berkshire Hathaway clase A
    "BFB":  "BF-B",     # Brown-Forman clase B
    "TEN":  "TS",       # Tenaris (ADR en NYSE)
    "GOLD2": "GOLD",    # Barrick Gold
}

# Memoria permanente de correcciones en una planilla de Google (columnas: fallido | correcto).
# Las credenciales van en los Secrets de Streamlit, NO en el código ni en GitHub.
# Si los Secrets no están configurados, estas funciones fallan en silencio y la app sigue
# andando (solo que sin memoria permanente), así que podés deployar antes de terminar el setup.
@st.cache_resource
def _conectar_sheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(st.secrets["sheets"]["spreadsheet_id"]).sheet1

def cargar_correcciones():
    """Lee las correcciones aprendidas desde la planilla. Devuelve {} si no hay o falla."""
    try:
        ws = _conectar_sheet()
        d = {}
        for fila in ws.get_all_values():
            if len(fila) >= 2 and fila[0].strip() and fila[0].strip().lower() != "fallido":
                d[fila[0].strip().upper()] = fila[1].strip().upper()
        return d
    except Exception:
        return {}

def guardar_correccion(fallido, correcto):
    """Guarda (o actualiza) una corrección en la planilla. Si no hay conexión, no rompe."""
    try:
        ws = _conectar_sheet()
        fallido = str(fallido).strip().upper()
        correcto = str(correcto).strip().upper()
        fila_idx = None
        for i, fila in enumerate(ws.get_all_values(), start=1):
            if fila and fila[0].strip().upper() == fallido:
                fila_idx = i
                break
        if fila_idx:
            ws.update_cell(fila_idx, 2, correcto)   # ya existía: actualizo
        else:
            ws.append_row([fallido, correcto])       # nuevo: agrego fila
        return True
    except Exception:
        return False

def sugerir_ticker(fallido, aprendidas=None):
    """Ticker candidato de Yahoo para uno que falló. Prioriza lo aprendido, luego la
    lista curada; '' si no hay nada confiable (mejor no sugerir que sugerir mal)."""
    fallido = fallido.strip().upper()
    if aprendidas and fallido in aprendidas:
        return aprendidas[fallido]
    return SUGERENCIAS_TICKER.get(fallido, "")

def leer_tickers_desde_estado_cuenta(archivo, secciones=("ACCIONES", "CEDEARS")):
    """Lee un estado de cuenta de PPI (.xlsx) con secciones por tipo de activo
    (MONEDAS, ACCIONES, BONOS, CEDEARS, FCI, ONS...) y devuelve los tickers de
    las secciones pedidas, ya traducidos a formato Yahoo Finance. Por defecto toma
    solo ACCIONES y CEDEARS, que son los activos que el motor sabe modelar.
    Si el archivo no tiene ese formato, cae a leer la primera columna como lista."""
    try:
        df = pd.read_excel(archivo, header=None)
    except Exception as e:
        st.sidebar.error(f"No se pudo leer el Excel: {e}")
        return None

    if df.empty:
        st.sidebar.warning("El archivo está vacío.")
        return None

    secciones_norm = {s.strip().upper() for s in secciones}
    pares, seccion_actual, detecto_formato = [], None, False

    for fila in df.itertuples(index=False):
        valores = list(fila)
        primera = "" if pd.isna(valores[0]) else str(valores[0]).strip()
        resto = [v for v in valores[1:] if not (pd.isna(v) or str(v).strip() == "")]
        es_titulo_solo = primera != "" and len(resto) == 0  # celda sola = título de sección

        prim_up = primera.upper()

        if es_titulo_solo:
            if prim_up in secciones_norm:
                detecto_formato = True
                seccion_actual = prim_up
            else:
                seccion_actual = None
            continue
        if seccion_actual is None:
            continue
        if prim_up in {"ESPECIE", "MONEDA"}:     # fila de encabezados de columna
            continue
        if prim_up == "SUBTOTAL" or primera == "":
            continue
        pares.append((prim_up, seccion_actual))

    # Fallback: archivo simple con una sola columna de tickers (sin secciones)
    if not detecto_formato:
        serie = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper()
        tickers = [t for t in serie if t and t not in {"TICKER", "TICKERS", "ESPECIE", "SYMBOL"}]
        tickers = list(dict.fromkeys(tickers))
        return tickers if tickers else None

    tickers = [especie_a_yahoo(e, s) for e, s in pares]
    tickers = list(dict.fromkeys(tickers))       # saca duplicados, mantiene orden
    return tickers if tickers else None

# --- 2. MOTOR DE SIMULACIÓN DIARIA (METODOLOGÍA CANOVA-SEGURA) ---
def generar_simulacion_diaria(mu_h_anual, cov_h_anual, n_sims, dist_type):
    n_assets = len(mu_h_anual)
    N_steps = 252 # 1 año bursátil
    
    # Parámetros diarios para el input (Paso 1)
    mu_d = mu_h_anual / 252
    cov_d = cov_h_anual / 252
    sigmas_d = np.sqrt(np.diag(cov_d))
    
    # Cholesky para correlación estocástica
    L = np.linalg.cholesky(cov_d + 1e-8 * np.eye(n_assets))
    
    retornos_diarios_sim = np.zeros((N_steps, n_sims, n_assets))
    
    # Drift diferenciado según TIF: MBG aplica Hull, robustas usan media pura
    if dist_type == 'MBG':
        drift_diario = mu_d - 0.5 * (sigmas_d**2)
    else:
        drift_diario = mu_d
    
    nu, gamma = 5, 1.3 
    for t in range(N_steps):
        if dist_type == 'MBG':
            z = np.random.standard_normal((n_assets, n_sims))
        elif dist_type == 'T-Student':
            z = t_dist.rvs(df=nu, size=(n_assets, n_sims))
            z = z / np.sqrt(nu / (nu - 2)) # Normalización de varianza
        elif dist_type == 'T-Skewed':
            Y = t_dist.rvs(df=nu, size=(n_assets, n_sims))
            Z_raw = np.where(Y >= 0, Y / gamma, Y * gamma)
            # Normalización del shock asimétrico
            z = (Z_raw - Z_raw.mean(axis=1, keepdims=True)) / Z_raw.std(axis=1, keepdims=True)
        
        shock = L @ z
        retornos_diarios_sim[t] = drift_diario + shock.T
        
    # --- PASO 3: ANUALIZACIÓN DE SIMULADOS ---
    mu_sim_anual = retornos_diarios_sim.mean(axis=(0, 1)) * 252
    reshaped_rets = retornos_diarios_sim.reshape(-1, n_assets)
    cov_sim_anual = np.cov(reshaped_rets, rowvar=False) * 252
    
    return mu_sim_anual, cov_sim_anual, retornos_diarios_sim

# --- 3. OPTIMIZADOR (PASO 4: MAXIMIZAR SHARPE SIMULADO CON ROBUSTEZ) ---
def optimizar_portfolio(mu_sim, cov_sim, rets_diarios, rf_rate, asset_names, objetivo, min_weight, capital):
    mu_s = pd.Series(mu_sim, index=asset_names)
    # Regularización para evitar error de UNH o matrices singulares
    cov_s = pd.DataFrame(cov_sim + 1e-5 * np.eye(len(asset_names)), index=asset_names, columns=asset_names)
    
    weights = None
    try:
        bounds = (min_weight if min_weight else 0.0, 1.0)
        ef = EfficientFrontier(mu_s, cov_s, weight_bounds=bounds)
        if "Sharpe" in objetivo:
            ef.max_sharpe(risk_free_rate=rf_rate)
        else:
            ef.min_volatility()
        weights = ef.clean_weights()
    except:
        try:
            # Plan B: Sin restricción de peso mínimo (solo >= 0)
            ef = EfficientFrontier(mu_s, cov_s, weight_bounds=(0, 1))
            ef.min_volatility()
            weights = ef.clean_weights()
        except:
            # Plan C: Pesos igualitarios (seguridad absoluta)
            weights = {t: 1.0/len(asset_names) for t in asset_names}
            ef = EfficientFrontier(mu_s, cov_s)
            ef.set_weights(weights)

    ret_p, vol_p, _ = ef.portfolio_performance(risk_free_rate=rf_rate)
    
    # Proyección de Riqueza Acumulada
    pesos_arr = np.array([weights[t] for t in asset_names])
    ret_log_acum_p = rets_diarios.sum(axis=0) @ pesos_arr
    ret_final_pct_p = np.exp(ret_log_acum_p) - 1
    
    return {
        "pesos": weights, 
        "retorno_esperado": ret_p, 
        "volatilidad_esperada": vol_p, 
        "vaR_pct": np.percentile(ret_final_pct_p, 5), 
        "ganancia_esperada_monetaria": ret_final_pct_p.mean() * capital,
        "resultado_monetario_peor_caso": capital * np.percentile(ret_final_pct_p, 5),
        "capital_final_peor_caso": capital * (1 + np.percentile(ret_final_pct_p, 5)),
        "capital_potencial": capital * (1 + ret_final_pct_p.mean()),
        "ret_acum_p": ret_final_pct_p
    }

# --- 4. INTERFAZ (RESTABLECIDA A 230+ LÍNEAS) ---
st.title("🚀 Optimizador Portfolio: Simulaciones Montecarlo")

with st.sidebar:
    st.header("⚙️ Parámetros")
    cap_inicial = st.number_input("Capital a Invertir ($)", min_value=100.0, value=10000.0)
    n_simulaciones = st.slider("Simulaciones Monte Carlo", 1000, 10000, 2000, 500)

    # --- CARGA DE TICKERS: desde estado de cuenta (.xlsx) o a mano ---
    st.markdown("**Tickers**")
    archivo_xlsx = st.file_uploader(
        "Importar estado de cuenta (.xlsx)", type=["xlsx"],
        help="Subí el estado de cuenta de PPI. Se importan solo las ESPECIES de "
             "ACCIONES y CEDEARS; se ignoran monedas, bonos, FCI y ONs."
    )

    tickers_default = "AAPL, MSFT, NVDA, GGAL, MELI, TSLA, AMZN, UNH"
    tickers_desde_archivo = leer_tickers_desde_estado_cuenta(archivo_xlsx) if archivo_xlsx else None

    if tickers_desde_archivo:
        tickers = tickers_desde_archivo
        st.success(f"✅ {len(tickers)} especies cargadas (acciones + cedears).")
        st.caption(", ".join(tickers))
    else:
        tickers_str = st.text_input("O escribilos a mano (separados por coma)", tickers_default)
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    col_a, col_b = st.columns(2)
    with col_a: f_inicio = st.date_input("Inicio", pd.to_datetime("2021-01-01"))
    with col_b: f_fin = st.date_input("Fin", pd.to_datetime("today"))
    dist_modelo = st.selectbox("Modelo de Distribución", ["MBG", "T-Student", "T-Skewed"])
    obj_input = st.radio("Objetivo", ["Max Sharpe Ratio", "Min Volatility"])
    restr_w = st.checkbox("Mínimo 5% por activo", value=True)

if st.button("Simular y Analizar"):
    st.session_state.run = True

if st.session_state.get("run"):
    # Correcciones aprendidas en sesiones anteriores (se aplican solas, sin pasar por rojo)
    persistidas = cargar_correcciones()

    # Ticker efectivo de cada activo: corrección manual > aprendida > original.
    # Un campo vacío significa "descartar este activo".
    def _corregido(t):
        clave = f"corr_{t}"
        if clave in st.session_state:
            return (st.session_state[clave] or "").strip().upper()
        return persistidas.get(t, t)
    tickers_finales = []
    for t in tickers:
        c = _corregido(t)
        if c:
            tickers_finales.append(c)
    tickers_finales = list(dict.fromkeys(tickers_finales))

    with st.spinner("Descargando data histórica y ejecutando simulación..."):
        rf = obtener_risk_free_live()
        data, fallidos, motivo = descargar_datos_robustos(tuple(tickers_finales), f_inicio, f_fin)
        
        # --- Tickers que Yahoo no encontró: en rojo, con sugerencia pre-cargada ---
        if fallidos:
            fallidos_set = set(fallidos)
            bases_con_error = [t for t in tickers if _corregido(t) in fallidos_set]
            if bases_con_error:
                st.error(f"🔴 Yahoo no encontró {len(bases_con_error)} ticker(s). Revisá la sugerencia y reintentá:")
                cols_fix = st.columns(min(len(bases_con_error), 3))
                for i, base in enumerate(bases_con_error):
                    fallo = _corregido(base)
                    clave = f"corr_{base}"
                    # La primera vez que aparece el campo, lo pre-cargamos con la sugerencia conocida
                    if clave not in st.session_state:
                        st.session_state[clave] = sugerir_ticker(fallo, persistidas)
                    etiqueta = f"🔴 {fallo}" + ("  ·  sugerido" if st.session_state[clave] else "")
                    with cols_fix[i % len(cols_fix)]:
                        st.text_input(etiqueta, key=clave, placeholder="Ticker correcto de Yahoo")
                st.caption("Donde conocemos el equivalente, el campo ya viene con la sugerencia. "
                           "Confirmá o editá, dejá vacío para descartar el activo, y tocá Reintentar.")
                st.button("🔄 Reintentar con las correcciones")

        # Guardar automáticamente las correcciones manuales que SÍ trajeron datos
        if data is not None and not data.empty:
            guardadas = []
            for base in tickers:
                clave = f"corr_{base}"
                if clave in st.session_state:
                    corr = (st.session_state[clave] or "").strip().upper()
                    if corr and corr != base and corr in data.columns and persistidas.get(base) != corr:
                        if guardar_correccion(base, corr):
                            guardadas.append(f"{base}→{corr}")
            if guardadas:
                st.caption("💾 Guardado para la próxima: " + ", ".join(guardadas))

        if data is not None and not data.empty:
            st.success(f"✅ {len(data.columns)} ticker(s) con datos: {', '.join(data.columns)}")

        if data is None or data.empty:
            st.error("⚠️ No hay datos válidos. Corregí los tickers en rojo o revisá las fechas.")
            if motivo:
                st.caption(f"Motivo técnico: {motivo}")
        else:
            if data.shape[1] < 2:
                st.warning("⚠️ Para optimizar una cartera hacen falta al menos **2 activos** con datos. "
                           "Completá o corregí más tickers (los que quedaron en rojo) y reintentá.")
                st.stop()
            log_returns = np.log(data / data.shift(1)).dropna()
            mu_h = log_returns.mean() * 252
            
            try:
                cov_h_clean = risk_models.CovarianceShrinkage(data).ledoit_wolf()
            except:
                cov_h_clean = risk_models.sample_cov(data)
            
            final_tickers = data.columns.tolist()

            # EJECUCIÓN MOTOR (PASOS 1, 2 Y 3)
            mu_sim, cov_sim, rets_diarios = generar_simulacion_diaria(mu_h.values, cov_h_clean.values, n_simulaciones, dist_modelo)
            
            # OPTIMIZACIÓN (PASO 4)
            res = optimizar_portfolio(mu_sim, cov_sim, rets_diarios, rf, final_tickers, obj_input, 0.05 if restr_w else None, cap_inicial)

            if res:
                st.success("✅ Análisis Completado")
                
                # FILA 1: MÉTRICAS
                st.subheader("📊 Métricas Esperadas del Portfolio")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Retorno Esperado", f"{res['retorno_esperado']:.2%}", help="Media de los retornos diarios simulados anualizados.")
                m2.metric("Volatilidad Anual", f"{res['volatilidad_esperada']:.2%}", help="Riesgo calculado sobre la matriz de covarianza simulada.")
                m3.metric("Ratio de Sharpe", f"{(res['retorno_esperado']-rf)/res['volatilidad_esperada']:.2f}")
                m4.metric("VaR 95% (Anual)", f"{res['vaR_pct']:.2%}", help="Percentil 5 de los retornos acumulados de la simulación.")

                # FILA 2: AUDITORÍA
                st.subheader("🎯 Eficiencia Individual Simulada")
                vols_sim_ind = np.sqrt(np.diag(cov_sim))
                df_ind = pd.DataFrame({
                    "Retorno Anual %": mu_sim * 100,
                    "Volatilidad Anual %": vols_sim_ind * 100,
                    "Ratio de Sharpe": (mu_sim - rf) / vols_sim_ind
                }, index=final_tickers)
                st.table(df_ind.sort_values(by="Retorno Anual %", ascending=False).style.format("{:.2f}"))

                st.divider()

                # FILA 3: FRONTERA EFICIENTE (RESTABLECIDA COMPLETA)
                st.subheader("📈 Frontera Eficiente de Markowitz")
                col_fe, col_pie = st.columns([2, 1])
                with col_fe:
                    n_port = 1000
                    p_r, p_v = [], []
                    for _ in range(n_port):
                        w = np.random.random(len(final_tickers)); w /= np.sum(w)
                        p_r.append(np.dot(w, mu_sim)); p_v.append(np.sqrt(np.dot(w.T, np.dot(cov_sim, w))))
                    
                    target_rets = np.linspace(min(mu_sim), max(mu_sim), 30)
                    frontier_v = []
                    for r in target_rets:
                        ef_line = EfficientFrontier(pd.Series(mu_sim, index=final_tickers), pd.DataFrame(cov_sim, index=final_tickers, columns=final_tickers))
                        try:
                            ef_line.efficient_return(r)
                            frontier_v.append(ef_line.portfolio_performance()[1])
                        except: frontier_v.append(None)

                    fig_fe, ax_fe = plt.subplots(figsize=(10, 6))
                    scatter = ax_fe.scatter(p_v, p_r, c=(np.array(p_r)/np.array(p_v)), marker='o', s=10, alpha=0.3, cmap='viridis')
                    plt.colorbar(scatter, label='Ratio de Sharpe')
                    
                    valid_v = [v for v in frontier_v if v is not None]; valid_r = [r for v, r in zip(frontier_v, target_rets) if v is not None]
                    ax_fe.plot(valid_v, valid_r, color='black', linestyle='--', linewidth=2, label="Frontera Real")

                    for i, t in enumerate(final_tickers):
                        ax_fe.scatter(vols_sim_ind[i], mu_sim[i], color='red', marker='X', s=100)
                        ax_fe.annotate(t, (vols_sim_ind[i], mu_sim[i]), xytext=(5,5), textcoords='offset points', fontweight='bold')
                    
                    ax_fe.scatter(res['volatilidad_esperada'], res['retorno_esperado'], color='gold', marker='*', s=400, edgecolor='black', label="Portfolio Óptimo")
                    ax_fe.set_xlabel("Riesgo (Volatilidad)"); ax_fe.set_ylabel("Retorno Esperado"); ax_fe.legend(); st.pyplot(fig_fe)

                with col_pie:
                    st.write("### Composición Visual")
                    fig_pie, ax_pie = plt.subplots()
                    pesos_plot = {k: v for k, v in res['pesos'].items() if v > 0.001}
                    ax_pie.pie(pesos_plot.values(), labels=pesos_plot.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(pesos_plot)))
                    st.pyplot(fig_pie)

                st.divider()
                
                # FILA 4: MONETARIAS
                st.subheader(f"💵 Proyección Monetaria (${cap_inicial:,.0f})")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ganancia Esperada", f"+ ${res['ganancia_esperada_monetaria']:,.2f}")
                c2.metric("📈 Capital Potencial", f"${res['capital_potencial']:,.2f}")
                c3.metric("Resultado Neto (VaR)", f"${res['resultado_monetario_peor_caso']:,.2f}")
                c4.metric("📉 Capital Remanente", f"${res['capital_final_peor_caso']:,.2f}", delta=f"${res['capital_final_peor_caso']-cap_inicial:,.2f}", delta_color="inverse")
                
                st.divider()
                
                # FILA 5: PLAN E HISTOGRAMA (EJE X EN %)
                col_plan, col_hist = st.columns([1.2, 1])
                with col_plan:
                    st.subheader("📋 Plan de Inversión Sugerido")
                    df_t = pd.DataFrame.from_dict(res['pesos'], orient='index', columns=['Ponderación (%)'])
                    df_t['Ponderación (%)'] = df_t['Ponderación (%)'] * 100
                    df_t['Monto ($)'] = (df_t['Ponderación (%)'] / 100) * cap_inicial
                    st.table(df_t.sort_values(by='Monto ($)', ascending=False).style.format({'Ponderación (%)': '{:.2f}%', 'Monto ($)': '$ {:,.2f}'}))

                with col_hist:
                    st.subheader("📊 Distribución de Retornos Anuales")
                    fig_hist, ax_hist = plt.subplots()
                    sns.histplot(res['ret_acum_p'] * 100, kde=True, ax=ax_hist, color="#1E88E5")
                    ax_hist.axvline(res['vaR_pct'] * 100, color='red', linestyle='--', label=f"VaR 95% ({res['vaR_pct']:.2%})")
                    ax_hist.set_xlabel("Retorno Anual Posible (%)")
                    ax_hist.set_ylabel("Cantidad de veces (Frecuencia)")
                    ax_hist.legend(); st.pyplot(fig_hist)
