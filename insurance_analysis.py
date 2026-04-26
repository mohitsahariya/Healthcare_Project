"""
=============================================================
  Medical Insurance Charges — Data Analytics Project
=============================================================
  Author   : Mohit Sahariya | Data Analyst, Wipro Ltd.
  Portfolio: https://mohitsahariya.github.io/MyPortfolio/
  Dataset  : insurance.csv (1,338 records, 7 features)
  Tools    : Python · Pandas · Matplotlib · Seaborn · NumPy

  HOW TO RUN:
    pip install pandas matplotlib seaborn numpy
    python insurance_analysis.py

  OUTPUT:
    charts/  → 9 publication-ready PNG charts
    Console  → Full statistical summary & key insights
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings('ignore')

# ── COLOUR PALETTE ──────────────────────────────────────────
NAVY    = '#070E1A'
DARK    = '#0C1624'
CARD    = '#0D1B2C'
BLUE    = '#1A6FEB'
BLUE2   = '#4B9EFF'
CYAN    = '#22D3EE'
GREEN   = '#10B981'
RED     = '#F43F5E'
GOLD    = '#F59E0B'
PURPLE  = '#8B5CF6'
ORANGE  = '#F97316'
MUTED   = '#6B8099'
WHITE   = '#EEF5FF'

# ── MATPLOTLIB GLOBAL STYLE ────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':    NAVY,
    'axes.facecolor':      DARK,
    'axes.edgecolor':      '#111F2E',
    'axes.labelcolor':     WHITE,
    'axes.titlecolor':     WHITE,
    'axes.titlesize':      13,
    'axes.titleweight':    'bold',
    'axes.titlepad':       14,
    'axes.grid':           True,
    'grid.color':          '#111F2E',
    'grid.linestyle':      '--',
    'grid.alpha':          0.6,
    'xtick.color':         MUTED,
    'ytick.color':         MUTED,
    'text.color':          WHITE,
    'font.family':         'DejaVu Sans',
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.spines.left':    False,
    'axes.spines.bottom':  False,
    'legend.facecolor':    DARK,
    'legend.edgecolor':    '#111F2E',
    'legend.labelcolor':   MUTED,
    'figure.dpi':          120,
})

OUTPUT_DIR = 'charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def watermark(fig):
    fig.text(0.98, 0.01,
             'Mohit Sahariya  |  mohitsahariya.github.io/MyPortfolio',
             ha='right', va='bottom', fontsize=7.5, color=MUTED, alpha=0.6)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=NAVY)
    plt.close(fig)
    print(f'  ✓  {path}')

def section(title):
    print(f'\n{"─"*55}\n  {title}\n{"─"*55}')

# ═══════════════════════════════════════════════════════════
# 1 · LOAD & ENGINEER FEATURES
# ═══════════════════════════════════════════════════════════
section('LOADING DATA')
df = pd.read_csv('insurance.csv')

df['bmi_cat'] = pd.cut(df['bmi'],
    bins=[0, 18.5, 25, 30, 100],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

df['age_group'] = pd.cut(df['age'],
    bins=[17, 30, 40, 50, 65],
    labels=['18-30', '31-40', '41-50', '51-64'])

df['smoker_bin'] = (df['smoker'] == 'yes').astype(int)
df['sex_bin']    = (df['sex'] == 'male').astype(int)

print(f'  Rows       : {len(df):,}')
print(f'  Columns    : {list(df.columns)}')
print(f'  Missing    : {df.isnull().sum().sum()}')
print(f'  Duplicates : {df.duplicated().sum()}')

# ═══════════════════════════════════════════════════════════
# 2 · CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════
section('DATASET SUMMARY')
print(df[['age', 'bmi', 'children', 'charges']].describe().round(2).to_string())

section('KEY GROUP STATISTICS')
for col in ['smoker', 'sex', 'region']:
    print(f'\n  Avg charges by {col}:')
    print(df.groupby(col)['charges'].agg(['mean','count']).round(2).to_string())

section('CORRELATION WITH CHARGES')
num_cols = ['smoker_bin', 'age', 'bmi', 'children', 'sex_bin']
corrs = df[num_cols + ['charges']].corr()['charges'].drop('charges').round(3)
for k, v in corrs.items():
    bar = '█' * int(abs(v) * 40)
    print(f'  {k:<15} r = {v:+.3f}  {bar}')

# ═══════════════════════════════════════════════════════════
# CHART 1 · Smoker Distribution (Donut)
# ═══════════════════════════════════════════════════════════
section('GENERATING CHARTS')
print('\n[1/9] Smoker Distribution')

smk  = df['smoker'].value_counts()
fig, ax = plt.subplots(figsize=(7, 6), facecolor=NAVY)
ax.set_facecolor(NAVY)
wedges, texts, autos = ax.pie(
    smk.values,
    labels=['Non-Smoker', 'Smoker'],
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops={'linewidth': 2.5, 'edgecolor': NAVY, 'width': 0.52},
    colors=[GREEN, RED]
)
for t in texts:  t.set_color(WHITE); t.set_fontsize(12); t.set_fontweight('bold')
for a in autos:  a.set_color(NAVY);  a.set_fontsize(11); a.set_fontweight('bold')
ax.text(0, 0, f'{len(df):,}\nRecords', ha='center', va='center',
        fontsize=12, fontweight='bold', color=WHITE)
ax.set_title('Smoker vs Non-Smoker Distribution', fontsize=14, fontweight='bold')
watermark(fig)
save(fig, '01_smoker_distribution.png')

# ═══════════════════════════════════════════════════════════
# CHART 2 · Avg Charges by Smoker Status
# ═══════════════════════════════════════════════════════════
print('[2/9] Charges by Smoker Status')

smk_avg = df.groupby('smoker')['charges'].mean().sort_index(ascending=False)
fig, ax = plt.subplots(figsize=(7, 5), facecolor=NAVY)
bars = ax.bar(['Non-Smoker', 'Smoker'], smk_avg.values,
              color=[GREEN, RED], width=0.5, edgecolor=NAVY, linewidth=1.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            f'${bar.get_height():,.0f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=WHITE)
ax.set_title('Average Annual Charges: Smoker vs Non-Smoker', fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Charges ($)', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${int(x):,}'))
ax.set_ylim(0, max(smk_avg.values) * 1.15)
ax.yaxis.grid(True); ax.set_axisbelow(True)

# Annotate the gap
gap = smk_avg['yes'] - smk_avg['no']
ax.annotate(f'Gap: ${gap:,.0f}\n(3.8× higher)',
            xy=(1, smk_avg['yes']/2), fontsize=10,
            color=GOLD, ha='center', va='center', fontweight='bold')
watermark(fig)
save(fig, '02_charges_by_smoker.png')

# ═══════════════════════════════════════════════════════════
# CHART 3 · Charge Distribution (Histogram)
# ═══════════════════════════════════════════════════════════
print('[3/9] Charge Distribution')

fig, ax = plt.subplots(figsize=(10, 5), facecolor=NAVY)
ax.hist(df[df['smoker']=='no']['charges'],  bins=40, color=GREEN, alpha=0.65,
        label='Non-Smoker', edgecolor=NAVY, linewidth=0.5)
ax.hist(df[df['smoker']=='yes']['charges'], bins=40, color=RED,   alpha=0.7,
        label='Smoker',     edgecolor=NAVY, linewidth=0.5)
ax.axvline(df['charges'].mean(), color=GOLD, linestyle='--', linewidth=1.5,
           label=f'Mean ${df["charges"].mean():,.0f}')
ax.axvline(df['charges'].median(), color=CYAN, linestyle=':', linewidth=1.5,
           label=f'Median ${df["charges"].median():,.0f}')
ax.set_title('Distribution of Annual Insurance Charges', fontsize=14, fontweight='bold')
ax.set_xlabel('Charges ($)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${int(x):,}'))
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.legend(fontsize=10)
watermark(fig)
save(fig, '03_charge_distribution.png')

# ═══════════════════════════════════════════════════════════
# CHART 4 · Age Group × Smoker (Grouped Bar)
# ═══════════════════════════════════════════════════════════
print('[4/9] Age Group × Smoker')

age_groups = ['18-30', '31-40', '41-50', '51-64']
smk_vals  = [df[(df['age_group']==g)&(df['smoker']=='yes')]['charges'].mean() for g in age_groups]
non_vals  = [df[(df['age_group']==g)&(df['smoker']=='no')]['charges'].mean()  for g in age_groups]

fig, ax = plt.subplots(figsize=(10, 6), facecolor=NAVY)
x = np.arange(len(age_groups))
w = 0.38
b1 = ax.bar(x - w/2, non_vals, w, color=GREEN,  alpha=0.85, label='Non-Smoker', edgecolor=NAVY)
b2 = ax.bar(x + w/2, smk_vals, w, color=RED,    alpha=0.85, label='Smoker',     edgecolor=NAVY)
for b in list(b1) + list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+200,
            f'${b.get_height():,.0f}', ha='center', va='bottom', fontsize=8.5, color=WHITE)
ax.set_xticks(x); ax.set_xticklabels(age_groups, fontsize=11)
ax.set_title('Avg Charges by Age Group × Smoking Status', fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Charges ($)', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'${int(v):,}'))
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.legend(fontsize=10)
watermark(fig)
save(fig, '04_age_group_smoker.png')

# ═══════════════════════════════════════════════════════════
# CHART 5 · BMI Category × Smoker
# ═══════════════════════════════════════════════════════════
print('[5/9] BMI Category × Smoker')

bmi_cats   = ['Underweight', 'Normal', 'Overweight', 'Obese']
bmi_smk    = [df[(df['bmi_cat']==c)&(df['smoker']=='yes')]['charges'].mean() for c in bmi_cats]
bmi_non    = [df[(df['bmi_cat']==c)&(df['smoker']=='no')]['charges'].mean()  for c in bmi_cats]

fig, ax = plt.subplots(figsize=(10, 6), facecolor=NAVY)
x = np.arange(len(bmi_cats))
b1 = ax.bar(x - w/2, bmi_non, w, color=GREEN, alpha=0.85, label='Non-Smoker', edgecolor=NAVY)
b2 = ax.bar(x + w/2, bmi_smk, w, color=RED,   alpha=0.85, label='Smoker',     edgecolor=NAVY)
for b in list(b1) + list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+300,
            f'${b.get_height():,.0f}', ha='center', va='bottom', fontsize=8.5, color=WHITE)
ax.set_xticks(x); ax.set_xticklabels(bmi_cats, fontsize=11)
ax.set_title('Avg Charges by BMI Category × Smoking Status', fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Charges ($)', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'${int(v):,}'))
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.legend(fontsize=10)

# Highlight obese-smoker bar
ax.annotate('⚠ Obese Smokers\n$41,693 avg (5× mean)',
            xy=(3 + w/2, bmi_smk[3]), xytext=(2.4, bmi_smk[3] + 2000),
            fontsize=9, color=GOLD, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GOLD))
watermark(fig)
save(fig, '05_bmi_smoker.png')

# ═══════════════════════════════════════════════════════════
# CHART 6 · Scatter: Age vs Charges (coloured by smoker)
# ═══════════════════════════════════════════════════════════
print('[6/9] Scatter: Age vs Charges')

fig, ax = plt.subplots(figsize=(10, 6), facecolor=NAVY)
for smk, colour, label, alpha, size in [
    ('no',  GREEN, 'Non-Smoker', 0.45, 18),
    ('yes', RED,   'Smoker',     0.65, 22),
]:
    sub = df[df['smoker'] == smk]
    ax.scatter(sub['age'], sub['charges'], c=colour, alpha=alpha,
               s=size, label=label, edgecolors='none')
ax.set_title('Age vs Annual Charges  (coloured by Smoker Status)', fontsize=14, fontweight='bold')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Charges ($)', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'${int(v):,}'))
ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.legend(fontsize=10, markerscale=1.8)
watermark(fig)
save(fig, '06_scatter_age_charges.png')

# ═══════════════════════════════════════════════════════════
# CHART 7 · Feature Correlation Bars
# ═══════════════════════════════════════════════════════════
print('[7/9] Feature Correlation')

corr_labels = ['Smoking', 'Age', 'BMI', 'Children', 'Sex (male)']
corr_values = [corrs['smoker_bin'], corrs['age'], corrs['bmi'],
               corrs['children'], corrs['sex_bin']]
corr_colours = [RED, GOLD, BLUE2, MUTED, PURPLE]

fig, ax = plt.subplots(figsize=(8, 5), facecolor=NAVY)
bars = ax.barh(corr_labels[::-1], corr_values[::-1],
               color=corr_colours[::-1], edgecolor=NAVY, height=0.55)
for bar, val in zip(bars, corr_values[::-1]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'r = {val:.3f}', va='center', fontsize=10, color=WHITE)
ax.axvline(0, color=MUTED, linewidth=1, alpha=0.5)
ax.set_xlim(0, 1.0)
ax.set_title('Pearson Correlation of Features with Insurance Charges',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Pearson r', fontsize=11)
ax.xaxis.grid(True); ax.set_axisbelow(True)
watermark(fig)
save(fig, '07_feature_correlation.png')

# ═══════════════════════════════════════════════════════════
# CHART 8 · Regional Risk (Combo Chart)
# ═══════════════════════════════════════════════════════════
print('[8/9] Regional Risk')

regions = ['southeast', 'northeast', 'northwest', 'southwest']
labels  = ['Southeast', 'Northeast', 'Northwest', 'Southwest']
reg_avg = [df[df['region']==r]['charges'].mean() for r in regions]
reg_smk = [df[df['region']==r]['smoker'].eq('yes').mean()*100 for r in regions]
bar_clrs = [RED, GOLD, GREEN, GREEN]

fig, ax1 = plt.subplots(figsize=(10, 6), facecolor=NAVY)
ax2 = ax1.twinx()
bars = ax1.bar(labels, reg_avg, color=[c+'BB' for c in bar_clrs],
               edgecolor=NAVY, width=0.5)
for bar in bars:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
             f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=10, color=WHITE)
ax2.plot(labels, reg_smk, color=CYAN, marker='o', markersize=8,
         linewidth=2.5, label='Smoker %', zorder=5)
for i, (lbl, pct) in enumerate(zip(labels, reg_smk)):
    ax2.text(i, pct + 0.8, f'{pct:.1f}%', ha='center', fontsize=9.5, color=CYAN, fontweight='bold')
ax1.set_title('Regional Risk: Avg Charges & Smoker Rate by Region',
              fontsize=14, fontweight='bold')
ax1.set_ylabel('Avg Charges ($)', fontsize=11)
ax2.set_ylabel('Smoker Rate (%)', fontsize=11, color=CYAN)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'${int(v):,}'))
ax2.set_ylim(0, 35); ax2.tick_params(colors=CYAN)
ax1.yaxis.grid(True, alpha=0.4); ax1.set_axisbelow(True)
lines, lbl_leg = ax2.get_legend_handles_labels()
ax2.legend(lines, lbl_leg, loc='upper right', fontsize=9)
watermark(fig)
save(fig, '08_regional_risk.png')

# ═══════════════════════════════════════════════════════════
# CHART 9 · BMI vs Charges (scatter, coloured by smoker)
# ═══════════════════════════════════════════════════════════
print('[9/9] BMI vs Charges Scatter')

fig, ax = plt.subplots(figsize=(10, 6), facecolor=NAVY)
for smk, colour, label, alpha in [
    ('no',  GREEN, 'Non-Smoker', 0.4),
    ('yes', RED,   'Smoker',     0.6),
]:
    sub = df[df['smoker'] == smk]
    ax.scatter(sub['bmi'], sub['charges'], c=colour, alpha=alpha,
               s=20, label=label, edgecolors='none')

# BMI = 30 threshold line
ax.axvline(30, color=GOLD, linestyle='--', linewidth=1.5,
           label='Obese threshold (BMI=30)')
ax.set_title('BMI vs Annual Charges  (coloured by Smoker Status)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('BMI', fontsize=11)
ax.set_ylabel('Charges ($)', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'${int(v):,}'))
ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.legend(fontsize=10, markerscale=1.8)
watermark(fig)
save(fig, '09_scatter_bmi_charges.png')

# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
section('ANALYSIS COMPLETE')
print(f'''
  ✅ 9 charts saved to ./{OUTPUT_DIR}/

  KEY INSIGHTS
  ─────────────────────────────────────────────────────────
  • Smoking is #1 charge driver   → Pearson r = {corrs["smoker_bin"]:.3f}
  • Smoker avg premium            → ${df[df["smoker"]=="yes"]["charges"].mean():,.0f}
  • Non-smoker avg premium        → ${df[df["smoker"]=="no"]["charges"].mean():,.0f}
  • Smoker multiplier             → {df[df["smoker"]=="yes"]["charges"].mean()/df[df["smoker"]=="no"]["charges"].mean():.1f}×
  • Obese smoker avg              → ${df[(df["bmi_cat"]=="Obese")&(df["smoker"]=="yes")]["charges"].mean():,.0f}
  • Southeast smoker rate         → {df[df["region"]=="southeast"]["smoker"].eq("yes").mean()*100:.1f}%
  • Portfolio avg age             → {df["age"].mean():.1f} yrs
  • Portfolio avg BMI             → {df["bmi"].mean():.1f}
  • Min charge                    → ${df["charges"].min():,.2f}
  • Max charge                    → ${df["charges"].max():,.2f}
  • Std deviation                 → ${df["charges"].std():,.0f}

  NEXT STEPS
  ─────────────────────────────────────────────────────────
  • Train XGBoost / Random Forest charge prediction model
  • Engineer bmi×smoker interaction feature
  • Deploy as FastAPI pricing endpoint
  • Build Power BI dashboard from this cleaned dataset

  Portfolio: https://mohitsahariya.github.io/MyPortfolio/
''')
