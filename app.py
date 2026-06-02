"""
Aplicación de Optimización Numérica
Métodos: Gradiente, Gradiente Conjugado (Polak-Ribière), Newton
Búsqueda de línea con condiciones de Wolfe
"""

import numpy as np
import pandas as pd
from numpy.linalg import norm, solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

# ══════════════════════════════════════════════════════════════════
#  EVALUADOR DE FUNCIÓN
# ══════════════════════════════════════════════════════════════════

def make_func(expr: str):
    """
    Convierte una expresión string en f(x: np.ndarray) -> float.
    Soporta: x[0], x[1], ..., np, sin, cos, exp, log, sqrt, pi, e, abs.
    """
    safe_globals = {
        "__builtins__": {},
        "np": np,
        "sin": np.sin,  "cos": np.cos,  "tan": np.tan,
        "exp": np.exp,  "log": np.log,  "log2": np.log2,
        "sqrt": np.sqrt, "abs": np.abs,
        "pi": np.pi,    "e": np.e,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    }
    def f(x):
        x = np.asarray(x, dtype=float)
        local = {"x": x}
        return float(eval(expr, safe_globals, local))
    return f


# ══════════════════════════════════════════════════════════════════
#  DERIVADAS NUMÉRICAS
# ══════════════════════════════════════════════════════════════════

def gradiente_numerico(f, x, h=1e-6):
    """Gradiente por diferencias centradas O(h²)."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2.0 * h)
    return g


def hessiana_numerica(f, x, h=1e-4):
    """Hessiana por diferencias finitas centradas O(h²)."""
    x  = np.asarray(x, dtype=float)
    n  = len(x)
    H  = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += h
                xm[i] -= h
                H[i, i] = (f(xp) - 2.0 * fx + f(xm)) / h ** 2
            else:
                xpp, xpm = x.copy(), x.copy()
                xmp, xmm = x.copy(), x.copy()
                xpp[i] += h
                xpp[j] += h
                xpm[i] += h
                xpm[j] -= h
                xmp[i] -= h
                xmp[j] += h
                xmm[i] -= h
                xmm[j] -= h
                val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h ** 2)
                H[i, j] = H[j, i] = val
    return H


# ══════════════════════════════════════════════════════════════════
#  BÚSQUEDA DE LÍNEA — CONDICIONES DE WOLFE
# ══════════════════════════════════════════════════════════════════

def wolfe_line_search(f, x, d, fx, gx, c1=1e-4, c2=0.9,
                      alpha0=1.0, max_ls=50):
    """
    Búsqueda de línea con condiciones de Wolfe (zoom algorithm).
    Retorna α satisfaciendo:
        f(x + α·d) ≤ f(x) + c1·α·∇f(x)ᵀd   (Armijo)
        |∇f(x+α·d)ᵀd| ≤ c2·|∇f(x)ᵀd|        (Wolfe fuerte)
    """
    alpha    = alpha0
    alpha_lo = 0.0
    alpha_hi = np.inf
    phi0     = fx
    dphi0    = float(gx @ d)

    if dphi0 >= 0:
        return 1e-8  # d no es dirección de descenso

    for _ in range(max_ls):
        x_new  = x + alpha * d
        f_new  = f(x_new)
        g_new  = gradiente_numerico(f, x_new)
        dphi   = float(g_new @ d)

        # Violación de Armijo → reducir alpha
        if f_new > phi0 + c1 * alpha * dphi0:
            alpha_hi = alpha
        else:
            # Condición de curvatura satisfecha
            if abs(dphi) <= -c2 * dphi0:
                return alpha
            if dphi >= 0:
                alpha_hi = alpha
            alpha_lo = alpha

        # Actualización de alpha
        if alpha_hi == np.inf:
            alpha *= 2.0
        else:
            alpha = (alpha_lo + alpha_hi) / 2.0

        if abs(alpha_hi - alpha_lo) < 1e-14:
            break

    return max(alpha, 1e-10)


# ══════════════════════════════════════════════════════════════════
#  MÉTODOS DE OPTIMIZACIÓN
# ══════════════════════════════════════════════════════════════════

def metodo_gradiente(f, x0, tol=1e-6, max_iter=500,
                     c1=1e-4, c2=0.9, alpha0=1.0, max_ls=50):
    """
    Método del descenso de gradiente (steepest descent).
    Dirección: d_k = -∇f(x_k)
    """
    x       = np.asarray(x0, dtype=float).copy()
    history = []

    for k in range(max_iter):
        gk    = gradiente_numerico(f, x)
        fk    = f(x)
        gnorm = norm(gk)
        history.append({"iter": k, "fx": fk, "gnorm": gnorm, "x": x.copy()})

        if gnorm < tol:
            return {"x": x, "fx": fk, "iters": k + 1,
                    "gnorm": gnorm, "converged": True, "history": history}

        d     = -gk
        alpha = wolfe_line_search(f, x, d, fk, gk, c1, c2, alpha0, max_ls)
        x     = x + alpha * d

        if not np.all(np.isfinite(x)):
            break

    gk = gradiente_numerico(f, x)
    return {"x": x, "fx": f(x), "iters": max_iter,
            "gnorm": norm(gk), "converged": False, "history": history}


def metodo_gradiente_conjugado(f, x0, tol=1e-6, max_iter=500,
                                c1=1e-4, c2=0.9, alpha0=1.0, max_ls=50):
    """
    Gradiente conjugado no lineal (Polak-Ribière con reinicio).
    β_k = max(0, (∇f_{k+1} - ∇f_k)ᵀ ∇f_{k+1} / ‖∇f_k‖²)
    """
    x       = np.asarray(x0, dtype=float).copy()
    g       = gradiente_numerico(f, x)
    d       = -g.copy()
    history = []
    n       = len(x0)

    for k in range(max_iter):
        fk    = f(x)
        gnorm = norm(g)
        history.append({"iter": k, "fx": fk, "gnorm": gnorm, "x": x.copy()})

        if gnorm < tol:
            return {"x": x, "fx": fk, "iters": k + 1,
                    "gnorm": gnorm, "converged": True, "history": history}

        alpha  = wolfe_line_search(f, x, d, fk, g, c1, c2, alpha0, max_ls)
        x_new  = x + alpha * d
        g_new  = gradiente_numerico(f, x_new)

        # β Polak-Ribière con clamping positivo
        beta = max(0.0, float((g_new - g) @ g_new) / (float(g @ g) + 1e-14))

        # Reinicio periódico cada n*5 iteraciones
        if (k + 1) % max(1, n * 5) == 0:
            beta = 0.0

        d   = -g_new + beta * d
        x   = x_new
        g   = g_new

        if not np.all(np.isfinite(x)):
            break

    return {"x": x, "fx": f(x), "iters": max_iter,
            "gnorm": norm(g), "converged": False, "history": history}


def metodo_newton(f, x0, tol=1e-6, max_iter=500,
                  c1=1e-4, c2=0.9, alpha0=1.0, max_ls=50):
    """
    Método de Newton con búsqueda de línea Wolfe.
    Dirección: d_k = -H_k⁻¹ ∇f(x_k)
    Regularización espectral si H no es definida positiva.
    """
    x       = np.asarray(x0, dtype=float).copy()
    history = []

    for k in range(max_iter):
        gk    = gradiente_numerico(f, x)
        fk    = f(x)
        gnorm = norm(gk)
        history.append({"iter": k, "fx": fk, "gnorm": gnorm, "x": x.copy()})

        if gnorm < tol:
            return {"x": x, "fx": fk, "iters": k + 1,
                    "gnorm": gnorm, "converged": True, "history": history}

        Hk = hessiana_numerica(f, x)

        # Regularización: garantizar H definida positiva
        lam_min = np.linalg.eigvalsh(Hk).min()
        if lam_min <= 1e-8:
            Hk += (-lam_min + 1e-6) * np.eye(len(x))

        try:
            d = -solve(Hk, gk)
        except np.linalg.LinAlgError:
            d = -gk  # fallback al gradiente

        # Asegurar dirección de descenso
        if float(gk @ d) >= 0:
            d = -gk

        alpha = wolfe_line_search(f, x, d, fk, gk, c1, c2, alpha0, max_ls)
        x     = x + alpha * d

        if not np.all(np.isfinite(x)):
            break

    gk = gradiente_numerico(f, x)
    return {"x": x, "fx": f(x), "iters": max_iter,
            "gnorm": norm(gk), "converged": False, "history": history}


# ══════════════════════════════════════════════════════════════════
#  GRÁFICOS
# ══════════════════════════════════════════════════════════════════

def plot_convergencia(history, metodo):
    """Gráfico log(‖∇f‖) vs iteraciones."""
    iters  = [h["iter"] for h in history]
    gnorms = [h["gnorm"] for h in history]
    fxvals = [h["fx"]    for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#060610")
        ax.tick_params(colors="#8080c0")
        ax.spines[:].set_color("#2a2a3e")
        ax.xaxis.label.set_color("#8080c0")
        ax.yaxis.label.set_color("#8080c0")
        ax.title.set_color("#a0a0e0")

    # — Panel izquierdo: ‖∇f‖ en escala log
    axes[0].semilogy(iters, gnorms, color="#4a6adf", linewidth=1.5)
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("‖∇f(x_k)‖")
    axes[0].set_title(f"Norma del gradiente — {metodo}")
    axes[0].grid(True, color="#1a1a2e", linewidth=0.5)
    axes[0].axhline(y=1e-6, color="#c05050", linewidth=0.8,
                    linestyle="--", label="Tolerancia 1e-6")
    axes[0].legend(facecolor="#0f0f1a", labelcolor="#8080c0",
                   edgecolor="#2a2a3e", fontsize=8)

    # — Panel derecho: f(x_k)
    axes[1].plot(iters, fxvals, color="#40c080", linewidth=1.5)
    axes[1].set_xlabel("Iteración")
    axes[1].set_ylabel("f(x_k)")
    axes[1].set_title(f"Valor función objetivo — {metodo}")
    axes[1].grid(True, color="#1a1a2e", linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_trayectoria_2d(f, history, x_range=None, y_range=None):
    """Curvas de nivel + trayectoria (solo para n=2)."""
    xs = [h["x"][0] for h in history]
    ys = [h["x"][1] for h in history]

    margin = 1.0
    if x_range is None:
        x_range = (min(xs) - margin, max(xs) + margin)
    if y_range is None:
        y_range = (min(ys) - margin, max(ys) + margin)

    xg = np.linspace(x_range[0], x_range[1], 200)
    yg = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(xg, yg)
    Z    = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
            except Exception:
                Z[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#060610")
    ax.tick_params(colors="#8080c0")
    ax.spines[:].set_color("#2a2a3e")
    ax.xaxis.label.set_color("#8080c0")
    ax.yaxis.label.set_color("#8080c0")
    ax.title.set_color("#a0a0e0")

    levels = np.percentile(Z[np.isfinite(Z)],
                           np.linspace(5, 95, 20))
    ax.contourf(X, Y, Z, levels=levels, cmap="Blues_r", alpha=0.5)
    ax.contour(X, Y, Z, levels=levels, colors="#2a3a6e",
               linewidths=0.5, alpha=0.7)

    ax.plot(xs, ys, "o-", color="#4a6adf", markersize=3,
            linewidth=1.2, label="Trayectoria", zorder=3)
    ax.plot(xs[0], ys[0], "s", color="#e0c040", markersize=8,
            label="x₀ inicio", zorder=4)
    ax.plot(xs[-1], ys[-1], "*", color="#40c080", markersize=12,
            label="x* final", zorder=4)

    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_title("Trayectoria de iteraciones")
    ax.legend(facecolor="#0f0f1a", labelcolor="#e0e0e0",
              edgecolor="#2a2a3e", fontsize=9)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
#  INTERFAZ STREAMLIT
# ══════════════════════════════════════════════════════════════════

EJEMPLOS = {
    "Rosenbrock":   ("100*(x[1]-x[0]**2)**2 + (1-x[0])**2",  "-1.2, 1.0",  "2"),
    "Cuadrática":   ("x[0]**2 + 2*x[1]**2 - x[0]*x[1]",      "3.0, 3.0",   "2"),
    "Himmelblau":   ("(x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2", "0.0, 0.0", "2"),
    "Función 1D":   ("(x[0]-2)**4 + (x[0]-2)**2 + x[0]",     "0.0",        "1"),
    "Rastrigin 2D": ("20 + x[0]**2 - 10*np.cos(2*np.pi*x[0]) "
                     "+ x[1]**2 - 10*np.cos(2*np.pi*x[1])",   "0.5, 0.5",   "2"),
}

METODOS = {
    "Descenso de Gradiente":              metodo_gradiente,
    "Gradiente Conjugado (Polak-Ribière)": metodo_gradiente_conjugado,
    "Método de Newton":                    metodo_newton,
}

st.set_page_config(
    page_title="Optimización Numérica",
    page_icon="⚙",
    layout="wide",
)

# ── CSS personalizado ──────────────────────────────────────────────
css = (
    "<style>"
    "[data-testid='stAppViewContainer'] {"
    "  background: #0a0a0f; color: #e0e0e8;"
    "}"
    "[data-testid='stSidebar'] {"
    "  background: #0f0f1a; border-right: 1px solid #2a2a3e;"
    "}"
    "h1, h2, h3 {"
    "  color: #7c9aff !important;"
    "  font-family: Courier New, monospace;"
    "}"
    ".stButton > button {"
    "  background: #1a2050; border: 1px solid #4060c0;"
    "  color: #7c9aff; font-family: Courier New, monospace;"
    "}"
    ".stButton > button:hover { background: #20285a; }"
    ".metric-box {"
    "  background: #0f0f1a; border: 1px solid #2a2a3e;"
    "  border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: .5rem;"
    "}"
    ".metric-label {"
    "  font-size: .65rem; color: #5060b0;"
    "  letter-spacing: .2em; text-transform: uppercase; margin-bottom: .3rem;"
    "}"
    ".metric-value {"
    "  font-size: 1.1rem; color: #7c9aff;"
    "  font-weight: 700; font-family: Courier New, monospace;"
    "}"
    ".converged { color: #40c080 !important; }"
    ".no-converge { color: #c08040 !important; }"
    "</style>"
)
st.markdown(css, unsafe_allow_html=True)


# ── Encabezado ────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;letter-spacing:.2em;margin-bottom:.2rem'>
  OPTIMIZACIÓN NUMÉRICA
</h1>
<p style='text-align:center;color:#4050a0;font-family:Courier New;font-size:.8rem;
          letter-spacing:.25em;margin-bottom:2rem'>
  GRADIENTE - GRADIENTE CONJUGADO - NEWTON - CONDICIONES DE WOLFE
</p>
""", unsafe_allow_html=True)


# ── Sidebar: parámetros ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuración")

    ejemplo_sel = st.selectbox("Cargar ejemplo", ["(personalizado)"] + list(EJEMPLOS.keys()))
    if ejemplo_sel != "(personalizado)":
        _expr, _x0, _n = EJEMPLOS[ejemplo_sel]
    else:
        _expr, _x0, _n = "x[0]**2 + 2*x[1]**2", "2.0, 3.0", "2"

    st.markdown("---")
    n_vars = st.selectbox("Número de variables", [1, 2, 3, 4, 5],
                          index=int(_n) - 1)
    func_expr = st.text_area("Función objetivo f(x)",
                              value=_expr, height=90,
                              help="Use x[0], x[1], ... · operadores: **, *, +, - · funciones: np.sin, np.cos, np.exp, np.log, np.sqrt")
    x0_str    = st.text_input("Punto inicial x₀ (separado por comas)", value=_x0)

    st.markdown("---")
    metodo_nombre = st.selectbox("Método de optimización", list(METODOS.keys()))

    st.markdown("---")
    st.markdown("**Parámetros de convergencia**")
    max_iter = st.number_input("Máx. iteraciones",    min_value=10, max_value=5000, value=500)
    tol      = st.number_input("Tolerancia ‖∇f‖",    min_value=1e-12, max_value=1e-1,
                                value=1e-6, format="%.2e")

    st.markdown("**Búsqueda de línea (Wolfe)**")
    c1       = st.number_input("c₁ — Armijo",         min_value=1e-6, max_value=0.49,
                                value=1e-4, format="%.4f",
                                help="c₁ controla suficiente descenso (típico: 1e-4)")
    c2       = st.number_input("c₂ — Curvatura",      min_value=0.01, max_value=0.999,
                                value=0.9,  format="%.3f",
                                help="c₂ > c₁ garantiza Wolfe (típico: 0.9 gradiente, 0.1 Newton)")
    alpha0   = st.number_input("α₀ inicial",           min_value=1e-4, max_value=10.0,
                                value=1.0,  format="%.4f")
    max_ls   = st.number_input("Máx. pasos búsqueda", min_value=5,    max_value=200,
                                value=50)

    ejecutar = st.button("▶  Ejecutar Optimización", use_container_width=True)


# ── Área principal ────────────────────────────────────────────────
if ejecutar:
    # Parseo punto inicial
    try:
        x0 = np.array([float(v.strip()) for v in x0_str.split(",")])
        if len(x0) != n_vars:
            st.error(f"El punto inicial tiene {len(x0)} componentes pero se declararon {n_vars} variables.")
            st.stop()
    except Exception as e:
        st.error(f"Error en punto inicial: {e}")
        st.stop()

    # Parseo función
    try:
        f = make_func(func_expr)
        _ = f(x0)  # prueba rápida
    except Exception as e:
        st.error(f"Error en la función objetivo: {e}")
        st.stop()

    if not (c1 < c2):
        st.warning("Se recomienda c₁ < c₂ para garantizar las condiciones de Wolfe.")

    # Ejecución
    with st.spinner("Calculando..."):
        metodo_fn = METODOS[metodo_nombre]
        resultado = metodo_fn(
            f, x0,
            tol=tol, max_iter=int(max_iter),
            c1=c1, c2=c2, alpha0=alpha0, max_ls=int(max_ls)
        )

    r = resultado

    # ── Métricas resumen ──────────────────────────────────────────
    st.markdown("### Resultados")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-box'>
          <div class='metric-label'>Punto mínimo x*</div>
          <div class='metric-value' style='font-size:.82rem'>[{', '.join(f'{v:.6g}' for v in r['x'])}]</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-box'>
          <div class='metric-label'>f(x*)</div>
          <div class='metric-value'>{r['fx']:.6e}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-box'>
          <div class='metric-label'>Iteraciones</div>
          <div class='metric-value'>{r['iters']}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        conv_cls  = "converged" if r["converged"] else "no-converge"
        conv_text = "Convergió ✓" if r["converged"] else "Máx. iter. ✗"
        st.markdown(f"""
        <div class='metric-box'>
          <div class='metric-label'>Estado</div>
          <div class='metric-value {conv_cls}'>{conv_text}</div>
        </div>""", unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown(f"""
        <div class='metric-box'>
          <div class='metric-label'>Error final ‖∇f(x*)‖</div>
          <div class='metric-value'>{r['gnorm']:.4e}</div>
        </div>""", unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class='metric-box'>
          <div class='metric-label'>Método</div>
          <div class='metric-value' style='font-size:.75rem;color:#8090d0'>{metodo_nombre}</div>
        </div>""", unsafe_allow_html=True)

    # ── Gráficos ──────────────────────────────────────────────────
    st.markdown("### Gráficos de convergencia")
    fig_conv = plot_convergencia(r["history"], metodo_nombre)
    st.pyplot(fig_conv)
    plt.close(fig_conv)

    if n_vars == 2:
        st.markdown("### Trayectoria en el espacio de búsqueda")
        fig_tray = plot_trayectoria_2d(f, r["history"])
        st.pyplot(fig_tray)
        plt.close(fig_tray)

    # ── Tabla de iteraciones ──────────────────────────────────────
    st.markdown("### Historial de iteraciones")
    hist = r["history"]
    step = max(1, len(hist) // 50)   # mostrar máx ~50 filas
    filas_mostradas = hist[::step]
    if hist[-1] not in filas_mostradas:
        filas_mostradas.append(hist[-1])

    df = pd.DataFrame([{
        "Iter":  h["iter"],
        "f(x)":  f"{h['fx']:.8e}",
        "‖∇f‖": f"{h['gnorm']:.4e}",
        "x":     "[" + ", ".join(f"{v:.6g}" for v in h["x"]) + "]",
    } for h in filas_mostradas])
    st.dataframe(df, use_container_width=True, hide_index=True)

    if len(hist) > len(filas_mostradas):
        st.caption(f"Mostrando {len(filas_mostradas)} de {len(hist)} iteraciones (paso={step})")

else:
    st.info("Configura los parámetros en el panel izquierdo y presiona **▶ Ejecutar Optimización**.")

    st.markdown("""
    #### Funciones de ejemplo
    | Función | Expresión | Mínimo conocido |
    |---------|-----------|-----------------|
    | Rosenbrock | `100*(x[1]-x[0]**2)**2 + (1-x[0])**2` | x*=(1,1), f=0 |
    | Cuadrática | `x[0]**2 + 2*x[1]**2 - x[0]*x[1]` | x*=(0,0), f=0 |
    | Himmelblau | `(x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2` | múltiples mínimos |
    | Rastrigin 2D | `20 + x[0]**2 - 10*np.cos(2*pi*x[0]) + ...` | x*=(0,0), f=0 |

    #### Condiciones de Wolfe
    La búsqueda de línea satisface simultáneamente:
    - **Armijo (suficiente descenso):** `f(x+αd) ≤ f(x) + c₁·α·∇f(x)ᵀd`
    - **Curvatura (Wolfe fuerte):** `|∇f(x+αd)ᵀd| ≤ c₂·|∇f(x)ᵀd|`
    """)
