# =============================================================================
# ANÁLISIS FINANCIERO — BANCO SANTANDER 2019-2024
# Curso: Visualización de Datos y Data Storytelling
# Autora: Rosa Madelim Mallma Moreno
# Herramientas: Python | pandas | matplotlib | seaborn | numpy
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Paleta corporativa Santander ──────────────────────────────────────────────
ROJO       = '#CC0000'
ROJO_DARK  = '#8B0000'
NAVY       = '#1F3864'
GRIS_CLARO = '#F2F2F2'
GRIS_MED   = '#AAAAAA'
VERDE      = '#1A6630'
AMBER      = '#C9873A'

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'axes.grid.axis':     'y',
    'grid.alpha':         0.3,
    'grid.color':         '#CCCCCC',
    'figure.dpi':         150,
    'savefig.dpi':        150,
    'savefig.bbox':       'tight',
})

# ── Carga de datos ─────────────────────────────────────────────────────────────
df   = pd.read_csv('santander_resultados_anuales.csv')
dfr  = pd.read_csv('santander_por_region.csv')
dft  = pd.read_csv('santander_trimestral.csv')

print("=" * 60)
print("DATOS CARGADOS CORRECTAMENTE")
print(f"  Anual:      {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"  Regional:   {dfr.shape[0]} filas × {dfr.shape[1]} columnas")
print(f"  Trimestral: {dft.shape[0]} filas × {dft.shape[1]} columnas")
print("=" * 60)

# ── Estadísticas descriptivas básicas ─────────────────────────────────────────
print("\n--- Estadísticas descriptivas (dataset anual) ---")
print(df[['beneficio_neto_ME','ingresos_totales_ME','ROTE_pct','eficiencia_pct']].describe().round(2))

# =============================================================================
# VISUALIZACIÓN 1 — HEATMAP DE CORRELACIONES
# Objetivo: identificar relaciones entre los indicadores financieros clave
# =============================================================================
print("\n[1/6] Generando heatmap de correlaciones...")

numericas = ['beneficio_neto_ME','ingresos_totales_ME','margen_intereses_ME',
             'comisiones_netas_ME','clientes_M','ROTE_pct','eficiencia_pct','CET1_pct']

corr = df[numericas].corr()

etiquetas = {
    'beneficio_neto_ME':    'Beneficio\nneto',
    'ingresos_totales_ME':  'Ingresos\ntotales',
    'margen_intereses_ME':  'Margen de\nintereses',
    'comisiones_netas_ME':  'Comisiones\nnetas',
    'clientes_M':           'Clientes\n(M)',
    'ROTE_pct':             'ROTE\n(%)',
    'eficiencia_pct':       'Eficiencia\n(%)',
    'CET1_pct':             'CET1\n(%)',
}
corr.columns = [etiquetas[c] for c in corr.columns]
corr.index   = [etiquetas[c] for c in corr.index]

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
    vmin=-1, vmax=1, center=0, linewidths=0.5, linecolor='white',
    annot_kws={'size': 10, 'weight': 'bold'},
    cbar_kws={'shrink': 0.8, 'label': 'Coeficiente de correlación (Pearson)'},
    ax=ax
)

ax.set_title('Mapa de Correlaciones — Indicadores Financieros Santander 2019–2024',
             fontsize=14, fontweight='bold', color=NAVY, pad=15)
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='both', labelsize=9)

# Anotación clave
ax.annotate('Correlación clave:\nEficiencia vs Ingresos = -0.94',
            xy=(0.02, 0.02), xycoords='axes fraction',
            fontsize=9, color=ROJO, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F0', edgecolor=ROJO, alpha=0.8))

plt.tight_layout()
plt.savefig('vis1_heatmap_correlaciones.png')
plt.close()
print("   ✓ vis1_heatmap_correlaciones.png guardado")

# =============================================================================
# VISUALIZACIÓN 2 — BARRAS + LÍNEA: BENEFICIO vs EFICIENCIA (multi-eje)
# Objetivo: mostrar la recuperación post-COVID y la mejora de eficiencia
# =============================================================================
print("[2/6] Generando gráfico beneficio + eficiencia...")

fig, ax1 = plt.subplots(figsize=(12, 7))

colores_barras = [VERDE if v >= 0 else ROJO for v in df['beneficio_neto_ME']]
barras = ax1.bar(df['año'], df['beneficio_neto_ME'],
                 color=colores_barras, alpha=0.85, width=0.55,
                 edgecolor='white', linewidth=0.8, zorder=3)

# Etiquetas sobre barras
for bar, val in zip(barras, df['beneficio_neto_ME']):
    ypos = bar.get_height() + 180 if val >= 0 else bar.get_height() - 480
    ax1.text(bar.get_x() + bar.get_width()/2, ypos,
             f'{val:,.0f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold',
             color=VERDE if val >= 0 else ROJO)

ax1.axhline(0, color=GRIS_MED, linewidth=0.8, linestyle='--')
ax1.set_xlabel('Año', fontsize=11)
ax1.set_ylabel('Beneficio neto atribuido (M€)', fontsize=11, color=NAVY)
ax1.tick_params(axis='y', labelcolor=NAVY)
ax1.set_ylim(-12000, 16500)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Eje secundario: ratio de eficiencia
ax2 = ax1.twinx()
ax2.plot(df['año'], df['eficiencia_pct'],
         color=ROJO, linewidth=2.5, marker='o', markersize=8,
         markerfacecolor='white', markeredgewidth=2.5,
         label='Ratio de eficiencia (%)', zorder=4)

for x, y in zip(df['año'], df['eficiencia_pct']):
    ax2.text(x, y + 0.4, f'{y}%', ha='center', va='bottom',
             fontsize=9, color=ROJO, fontweight='bold')

ax2.set_ylabel('Ratio de eficiencia (%)\n← Menor es mejor', fontsize=11, color=ROJO)
ax2.tick_params(axis='y', labelcolor=ROJO)
ax2.set_ylim(38, 52)
ax2.invert_yaxis()

# Zona COVID sombreada
ax1.axvspan(2019.6, 2020.4, alpha=0.08, color=ROJO, label='Impacto COVID-19')
ax1.text(2020, -10500, 'COVID-19\n-8.771 M€', ha='center', fontsize=9,
         color=ROJO, fontweight='bold')

# Flecha de tendencia mejora
ax1.annotate('', xy=(2024, 13000), xytext=(2021, 13000),
             arrowprops=dict(arrowstyle='->', color=VERDE, lw=2))
ax1.text(2022.5, 13400, '3 años consecutivos de récord', ha='center',
         fontsize=9, color=VERDE, fontweight='bold')

ax1.set_title('Beneficio Neto y Ratio de Eficiencia — Banco Santander 2019–2024',
              fontsize=14, fontweight='bold', color=NAVY, pad=15)
ax1.set_xticks(df['año'])

legend_barra_pos = mpatches.Patch(color=VERDE, label='Beneficio positivo')
legend_barra_neg = mpatches.Patch(color=ROJO, alpha=0.85, label='Pérdida neta')
legend_linea     = plt.Line2D([0], [0], color=ROJO, lw=2.5, marker='o',
                               markerfacecolor='white', label='Ratio eficiencia (%)')
ax1.legend(handles=[legend_barra_pos, legend_barra_neg, legend_linea],
           loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('vis2_beneficio_eficiencia.png')
plt.close()
print("   ✓ vis2_beneficio_eficiencia.png guardado")

# =============================================================================
# VISUALIZACIÓN 3 — BARRAS AGRUPADAS: BENEFICIO POR REGIÓN 2023 vs 2024
# Objetivo: comparar rendimiento geográfico y mostrar crecimiento
# =============================================================================
print("[3/6] Generando gráfico por regiones...")

df23 = dfr[dfr['año'] == 2023].set_index('pais')
df24 = dfr[dfr['año'] == 2024].set_index('pais')
paises = df24.index.tolist()

x = np.arange(len(paises))
ancho = 0.38

fig, (ax_barras, ax_var) = plt.subplots(1, 2, figsize=(16, 7),
                                         gridspec_kw={'width_ratios': [2, 1]})

# Panel izquierdo: barras agrupadas
b23 = ax_barras.bar(x - ancho/2, df23.loc[paises, 'beneficio_neto_ME'],
                    ancho, label='2023', color=NAVY, alpha=0.75, zorder=3)
b24 = ax_barras.bar(x + ancho/2, df24.loc[paises, 'beneficio_neto_ME'],
                    ancho, label='2024', color=ROJO, alpha=0.85, zorder=3)

for bar in b23:
    ax_barras.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                   f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8, color=NAVY)
for bar in b24:
    ax_barras.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                   f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8,
                   color=ROJO, fontweight='bold')

ax_barras.set_xticks(x)
ax_barras.set_xticklabels(paises, rotation=30, ha='right', fontsize=10)
ax_barras.set_ylabel('Beneficio neto (M€)', fontsize=11)
ax_barras.set_title('Beneficio por Mercado — 2023 vs 2024', fontsize=13,
                     fontweight='bold', color=NAVY)
ax_barras.legend(fontsize=10)
ax_barras.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Panel derecho: variación interanual horizontal
variaciones = df24.loc[paises, 'var_interanual_pct']
colores_var  = [VERDE if v >= 0 else ROJO for v in variaciones]

barras_h = ax_var.barh(paises, variaciones, color=colores_var, alpha=0.85,
                        edgecolor='white', zorder=3)
ax_var.axvline(0, color=GRIS_MED, linewidth=1)

for bar, val in zip(barras_h, variaciones):
    xpos = val + 0.8 if val >= 0 else val - 0.8
    ha   = 'left' if val >= 0 else 'right'
    ax_var.text(xpos, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', ha=ha,
                fontsize=9, fontweight='bold',
                color=VERDE if val >= 0 else ROJO)

ax_var.set_xlabel('Variación interanual (%)', fontsize=11)
ax_var.set_title('Crecimiento 2024 vs 2023', fontsize=13,
                  fontweight='bold', color=NAVY)
ax_var.invert_yaxis()

# Resaltar España
spain_idx = paises.index('España')
ax_var.get_yticklabels()[spain_idx].set_color(VERDE)
ax_var.get_yticklabels()[spain_idx].set_fontweight('bold')

plt.suptitle('Análisis Geográfico — Banco Santander', fontsize=15,
             fontweight='bold', color=NAVY, y=1.01)
plt.tight_layout()
plt.savefig('vis3_beneficio_por_region.png')
plt.close()
print("   ✓ vis3_beneficio_por_region.png guardado")

# =============================================================================
# VISUALIZACIÓN 4 — EVOLUCIÓN DE KPIs CLAVE (panel 2×2)
# Objetivo: mostrar la tendencia de los 4 indicadores estratégicos
# =============================================================================
print("[4/6] Generando panel de KPIs...")

fig, axs = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Evolución de KPIs Estratégicos — Banco Santander 2019–2024',
             fontsize=15, fontweight='bold', color=NAVY, y=1.01)

kpis = [
    ('beneficio_neto_ME', 'Beneficio Neto Atribuido (M€)', ROJO, True),
    ('clientes_M',        'Base de Clientes (millones)',    NAVY, False),
    ('ROTE_pct',          'ROTE — Return on Tangible Equity (%)', VERDE, False),
    ('eficiencia_pct',    'Ratio de Eficiencia (%) ← menor es mejor', AMBER, True),
]

for ax, (col, titulo, color, hay_barras) in zip(axs.flat, kpis):
    valores = df[col].values
    años    = df['año'].values

    if hay_barras:
        colores = [VERDE if v >= 0 else ROJO for v in valores] if col == 'beneficio_neto_ME' else [color]*len(valores)
        ax.bar(años, valores, color=colores, alpha=0.8, width=0.55, zorder=3)
    else:
        ax.fill_between(años, valores, alpha=0.15, color=color)
        ax.plot(años, valores, color=color, linewidth=2.5,
                marker='o', markersize=7, markerfacecolor='white',
                markeredgewidth=2.5, zorder=4)

    for x_val, y_val in zip(años, valores):
        ax.text(x_val, y_val + (max(valores)-min(valores))*0.03,
                f'{y_val:,.1f}' if col in ('ROTE_pct','eficiencia_pct','clientes_M') else f'{y_val:,.0f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color)

    ax.set_title(titulo, fontsize=11, fontweight='bold', color=NAVY)
    ax.set_xticks(años)
    ax.tick_params(axis='x', labelsize=9)

    if col == 'eficiencia_pct':
        ax.invert_yaxis()
    if col == 'beneficio_neto_ME':
        ax.axhline(0, color=GRIS_MED, linewidth=0.8, linestyle='--')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Valor 2024 destacado
    ultimo = valores[-1]
    ax.annotate(f'2024: {ultimo:,.1f}' if col != 'beneficio_neto_ME' else f'2024: {ultimo:,.0f} M€',
                xy=(2024, ultimo), xytext=(2022.5, ultimo * 1.12 if ultimo > 0 else ultimo * 0.85),
                fontsize=8, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

plt.tight_layout()
plt.savefig('vis4_panel_kpis.png')
plt.close()
print("   ✓ vis4_panel_kpis.png guardado")

# =============================================================================
# VISUALIZACIÓN 5 — DONA: COMPOSICIÓN DEL BENEFICIO 2024 POR REGIÓN
# Objetivo: mostrar el peso de cada mercado en el resultado total del grupo
# =============================================================================
print("[5/6] Generando gráfico de dona por región...")

df24_region = (dfr[dfr['año'] == 2024]
               .groupby('region')['beneficio_neto_ME']
               .sum()
               .reset_index()
               .sort_values('beneficio_neto_ME', ascending=False))

etiquetas_don = df24_region['region'].tolist()
valores_don   = df24_region['beneficio_neto_ME'].tolist()
total_don     = sum(valores_don)
colores_don   = [ROJO, NAVY, AMBER, VERDE][:len(etiquetas_don)]

fig, ax = plt.subplots(figsize=(10, 7))

wedges, texts, autotexts = ax.pie(
    valores_don,
    labels=None,
    colors=colores_don,
    autopct=lambda p: f'{p:.1f}%',
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
    pctdistance=0.78,
    textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'}
)

# Centro dona
ax.text(0, 0, f'Total\n{total_don:,.0f} M€',
        ha='center', va='center', fontsize=13,
        fontweight='bold', color=NAVY)

# Leyenda con valores absolutos
leyenda_items = [
    mpatches.Patch(color=c, label=f'{r}:  {v:,.0f} M€  ({v/total_don*100:.1f}%)')
    for c, r, v in zip(colores_don, etiquetas_don, valores_don)
]
ax.legend(handles=leyenda_items, loc='lower center',
          bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10,
          framealpha=0.9, edgecolor=GRIS_MED)

ax.set_title('Distribución del Beneficio 2024 por Región del Grupo\nBanco Santander',
             fontsize=14, fontweight='bold', color=NAVY, pad=20)

plt.tight_layout()
plt.savefig('vis5_dona_regiones.png')
plt.close()
print("   ✓ vis5_dona_regiones.png guardado")

# =============================================================================
# VISUALIZACIÓN 6 — WATERFALL CHART: Crecimiento beneficio 2023 → 2024
# Objetivo: descomponer el incremento de beneficio por mercado
# =============================================================================
print("[6/6] Generando waterfall chart 2023→2024...")

# Calcular variaciones por país
df_wf = df24[['beneficio_neto_ME']].join(df23['beneficio_neto_ME'],
                                          rsuffix='_23').dropna()
df_wf['variacion'] = df_wf['beneficio_neto_ME'] - df_wf['beneficio_neto_ME_23']
df_wf = df_wf.sort_values('variacion', ascending=False)

categorias = ['Beneficio\n2023'] + df_wf.index.tolist() + ['Beneficio\n2024']
variaciones_wf = [0] + df_wf['variacion'].tolist() + [0]

beneficio_2023 = 11076
beneficio_2024 = 12574
bases = []
base = beneficio_2023

for i, v in enumerate(variaciones_wf):
    if i == 0:
        bases.append(0)
    elif i == len(variaciones_wf) - 1:
        bases.append(0)
    else:
        if v >= 0:
            bases.append(base)
            base += v
        else:
            base += v
            bases.append(base)

alturas = []
for i, (b, v) in enumerate(zip(bases, variaciones_wf)):
    if i == 0:
        alturas.append(beneficio_2023)
    elif i == len(variaciones_wf) - 1:
        alturas.append(beneficio_2024)
    else:
        alturas.append(abs(v))

colores_wf = []
for i, v in enumerate(variaciones_wf):
    if i == 0:
        colores_wf.append(NAVY)
    elif i == len(variaciones_wf) - 1:
        colores_wf.append(ROJO)
    elif v >= 0:
        colores_wf.append(VERDE)
    else:
        colores_wf.append('#CC3333')

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(categorias))
barras_wf = ax.bar(x, alturas, bottom=bases, color=colores_wf,
                    alpha=0.87, width=0.6, edgecolor='white', linewidth=0.8, zorder=3)

# Etiquetas
for i, (bar, base_b, alto, var) in enumerate(zip(barras_wf, bases, alturas, variaciones_wf)):
    if i == 0:
        label = f'{beneficio_2023:,.0f} M€'
        ypos  = beneficio_2023 + 100
    elif i == len(variaciones_wf) - 1:
        label = f'{beneficio_2024:,.0f} M€'
        ypos  = beneficio_2024 + 100
    else:
        label = f'{var:+,.0f} M€'
        ypos  = base_b + alto + 80 if var >= 0 else base_b - 280
    ax.text(x[i], ypos, label, ha='center', va='bottom',
            fontsize=9, fontweight='bold',
            color='white' if i in [0, len(variaciones_wf)-1] else colores_wf[i])

# Línea conectora entre barras
for i in range(len(x) - 1):
    if i == 0:
        y_conec = beneficio_2023
    elif i == len(x) - 2:
        continue
    else:
        y_conec = bases[i] + alturas[i] if variaciones_wf[i] >= 0 else bases[i]
    ax.plot([x[i] + 0.31, x[i+1] - 0.31], [y_conec, y_conec],
            color=GRIS_MED, linewidth=0.8, linestyle='--')

ax.set_xticks(x)
ax.set_xticklabels(categorias, fontsize=10)
ax.set_ylabel('Beneficio neto atribuido (M€)', fontsize=11)
ax.set_title('Waterfall Chart — Composición del Crecimiento de Beneficio 2023 → 2024\nBanco Santander',
             fontsize=13, fontweight='bold', color=NAVY, pad=15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.set_ylim(0, 15500)

# Anotación España
ax.annotate('España: 93% del\ncrecimiento total',
            xy=(x[1], bases[1] + alturas[1]/2),
            xytext=(x[1] + 1.5, 14000),
            fontsize=9, color=VERDE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=VERDE, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=VERDE))

# Leyenda
leg_pos = mpatches.Patch(color=VERDE, label='Contribución positiva')
leg_neg = mpatches.Patch(color='#CC3333', label='Contribución negativa')
leg_tot = mpatches.Patch(color=NAVY, label='Total 2023')
leg_rec = mpatches.Patch(color=ROJO, label='Total 2024 (récord)')
ax.legend(handles=[leg_pos, leg_neg, leg_tot, leg_rec],
          loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('vis6_waterfall_crecimiento.png')
plt.close()
print("   ✓ vis6_waterfall_crecimiento.png guardado")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 60)
print("TODAS LAS VISUALIZACIONES GENERADAS CORRECTAMENTE")
print("=" * 60)
print("\nArchivos PNG guardados:")
for i in range(1, 7):
    nombres = {
        1: 'vis1_heatmap_correlaciones.png',
        2: 'vis2_beneficio_eficiencia.png',
        3: 'vis3_beneficio_por_region.png',
        4: 'vis4_panel_kpis.png',
        5: 'vis5_dona_regiones.png',
        6: 'vis6_waterfall_crecimiento.png',
    }
    print(f"  [{i}] {nombres[i]}")

print("\nCSV de entrada utilizados:")
print("  - santander_resultados_anuales.csv")
print("  - santander_por_region.csv")
print("  - santander_trimestral.csv")
print("\nAutora: Rosa Madelim Mallma Moreno")
print("Curso:  Visualización de Datos y Data Storytelling")
print("=" * 60)
