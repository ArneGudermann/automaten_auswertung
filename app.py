import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

WOCHENTAGE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
C_BLUE = "#2563EB"
C_GREEN = "#059669"
C_RED = "#DC2626"
C_SLATE = "#0F172A"

CHART_STYLE = {
    "axes.facecolor": "#F8FAFC",
    "figure.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E2E8F0",
    "axes.labelcolor": "#475569",
    "xtick.color": "#64748B",
    "ytick.color": "#64748B",
    "text.color": "#1E293B",
    "grid.color": "#E2E8F0",
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_chart_style():
    plt.rcParams.update(CHART_STYLE)


def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["Value"] = df["Value"].str.replace(",", ".").astype(float)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Datum"] = df["Timestamp"].dt.date
    df["Stunde"] = df["Timestamp"].dt.hour
    TAGE_DE = {"Monday": "Montag", "Tuesday": "Dienstag", "Wednesday": "Mittwoch",
               "Thursday": "Donnerstag", "Friday": "Freitag", "Saturday": "Samstag", "Sunday": "Sonntag"}
    df["Wochentag"] = df["Timestamp"].dt.day_name().map(TAGE_DE)
    df = df[df["Product"].notna() & (df["Transaction type"] == "SALE")]
    return df


def fig_wochentag_heatmap(df):
    apply_chart_style()
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    wt = df.groupby("Wochentag")["Value"].agg(["count", "sum"]).reindex(WOCHENTAGE).fillna(0)
    colors = [C_BLUE if v == wt["count"].max() else "#BFDBFE" for v in wt["count"]]
    bars = ax1.bar(range(len(wt)), wt["count"], color=colors, zorder=3, width=0.6)
    ax1.set_title("Verkäufe nach Wochentag", fontweight="bold", fontsize=13, pad=12)
    ax1.set_ylabel("Anzahl Verkäufe")
    ax1.set_xticks(range(len(wt.index)))
    ax1.set_xticklabels(wt.index, rotation=30, ha="right")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax1.set_axisbelow(True)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3, int(h),
                     ha="center", va="bottom", fontsize=9, color="#475569")

    ax2 = fig.add_subplot(gs[1])
    heat = df.groupby(["Wochentag", "Stunde"])["Value"].count().unstack(fill_value=0)
    heat = heat.reindex(WOCHENTAGE).fillna(0)
    im = ax2.imshow(heat.values, aspect="auto", cmap="Blues", interpolation="nearest")
    ax2.set_title("Heatmap: Uhrzeit × Wochentag", fontweight="bold", fontsize=13, pad=12)
    ax2.set_xlabel("Uhrzeit")
    ax2.set_xticks(range(len(heat.columns)))
    ax2.set_xticklabels([f"{h}h" for h in heat.columns], rotation=90, fontsize=7)
    ax2.set_yticks(range(len(WOCHENTAGE)))
    ax2.set_yticklabels(WOCHENTAGE, fontsize=9)
    plt.colorbar(im, ax=ax2, label="Verkäufe")
    fig.tight_layout()
    return fig


def fig_top_produkte(df):
    apply_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    prod = df.groupby("Product")["Value"].agg(["count", "sum"]).sort_values("count", ascending=True).tail(15)
    axes[0].barh(prod.index, prod["count"], color=C_BLUE, zorder=3, height=0.6)
    axes[0].set_title("Top 15 nach Anzahl", fontweight="bold", fontsize=13, pad=12)
    axes[0].set_xlabel("Anzahl Verkäufe")
    axes[0].xaxis.grid(True, linestyle="--", alpha=0.6)
    axes[0].set_axisbelow(True)

    prod2 = df.groupby("Product")["Value"].agg(["count", "sum"]).sort_values("sum", ascending=True).tail(15)
    axes[1].barh(prod2.index, prod2["sum"], color=C_GREEN, zorder=3, height=0.6)
    axes[1].set_title("Top 15 nach Umsatz (€)", fontweight="bold", fontsize=13, pad=12)
    axes[1].set_xlabel("Umsatz in €")
    axes[1].xaxis.grid(True, linestyle="--", alpha=0.6)
    axes[1].set_axisbelow(True)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return fig


def fig_schwache_produkte(df):
    apply_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    schwach = df.groupby("Product")["Value"].agg(["count", "sum"]).sort_values("count").head(10)
    ax.barh(schwach.index, schwach["count"], color="#FCA5A5", zorder=3, height=0.6)
    ax.set_title("Schwächste Produkte", fontweight="bold", fontsize=13, pad=12)
    ax.set_xlabel("Anzahl Verkäufe")
    ax.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    for i, (idx, row) in enumerate(schwach.iterrows()):
        ax.text(row["count"] + 0.05, i,
                f"  {row['count']:.0f}x · {row['sum']:.2f} €",
                va="center", fontsize=9, color="#475569")
    fig.tight_layout()
    return fig


def fig_slots_zahlungsart(df):
    apply_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    slot = df.groupby("Column")["Value"].agg(["count", "sum"]).sort_values("count", ascending=False).head(20)
    colors = [C_BLUE if v == slot["count"].max() else "#BFDBFE" for v in slot["count"]]
    axes[0].bar(slot.index.astype(str), slot["count"], color=colors, zorder=3, width=0.6)
    axes[0].set_title("Top 20 Slots nach Verkäufen", fontweight="bold", fontsize=13, pad=12)
    axes[0].set_xlabel("Slot")
    axes[0].set_ylabel("Anzahl Verkäufe")
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.6)
    axes[0].set_axisbelow(True)
    axes[0].tick_params(axis="x", rotation=45)

    zahl_tag = df.groupby(["Datum", "Payment"])["Value"].sum().unstack(fill_value=0)
    if "Cash" in zahl_tag.columns:
        axes[1].fill_between(range(len(zahl_tag)), zahl_tag["Cash"], alpha=0.15, color=C_BLUE)
        axes[1].plot(range(len(zahl_tag)), zahl_tag["Cash"], label="Bar", color=C_BLUE, marker="o", markersize=4, linewidth=2)
    if "Cashless" in zahl_tag.columns:
        axes[1].fill_between(range(len(zahl_tag)), zahl_tag["Cashless"], alpha=0.15, color=C_GREEN)
        axes[1].plot(range(len(zahl_tag)), zahl_tag["Cashless"], label="Karte", color=C_GREEN, marker="o", markersize=4, linewidth=2)
    axes[1].set_title("Bar vs. Karte pro Tag", fontweight="bold", fontsize=13, pad=12)
    axes[1].set_ylabel("Umsatz (€)")
    axes[1].set_xticks(range(len(zahl_tag)))
    axes[1].set_xticklabels([str(d)[5:] for d in zahl_tag.index], rotation=45, ha="right", fontsize=7)
    axes[1].legend(frameon=False)
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.6)
    axes[1].set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig_tagesumsatz(df, monat_label):
    apply_chart_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    tag = df.groupby("Datum")["Value"].sum()
    colors = [C_BLUE if v == tag.max() else "#BFDBFE" for v in tag.values]
    ax.bar(range(len(tag)), tag.values, color=colors, zorder=3, width=0.6)
    ax.set_xticks(range(len(tag)))
    ax.set_xticklabels([str(d)[5:] for d in tag.index], rotation=45, ha="right")
    ax.set_ylabel("Umsatz (€)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    for i, v in enumerate(tag.values):
        ax.text(i, v + 0.2, f"{v:.0f}€", ha="center", va="bottom", fontsize=7.5, color="#475569")
    fig.tight_layout()
    return fig


def fig_mwst(df, monat_label):
    apply_chart_style()
    df = df.copy()
    df["Netto"] = df["Value"] / (1 + df["Tax Rate"] / 100)
    df["MwSt_Betrag"] = df["Value"] - df["Netto"]
    mwst = df.groupby("Tax Rate").agg(
        Anzahl=("Value", "count"),
        Brutto=("Value", "sum"),
        Netto=("Netto", "sum"),
        MwSt=("MwSt_Betrag", "sum"),
    ).reset_index()
    mwst["Label"] = mwst["Tax Rate"].astype(int).astype(str) + " %"

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    wedges, texts, autotexts = ax1.pie(
        mwst["Brutto"], labels=mwst["Label"], autopct="%1.1f%%",
        colors=[C_BLUE, C_GREEN], startangle=90,
        wedgeprops={"linewidth": 2, "edgecolor": "white"}
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_color("white")
        t.set_fontweight("bold")
    ax1.set_title("Brutto-Umsatz nach MwSt-Satz", fontweight="bold", fontsize=13, pad=12)

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    tabelle_data = []
    for _, row in mwst.iterrows():
        tabelle_data.append([row["Label"], f"{row['Anzahl']:.0f}", f"{row['Brutto']:.2f} €",
                              f"{row['Netto']:.2f} €", f"{row['MwSt']:.2f} €"])
    tabelle_data.append(["Gesamt", f"{mwst['Anzahl'].sum():.0f}", f"{mwst['Brutto'].sum():.2f} €",
                         f"{mwst['Netto'].sum():.2f} €", f"{mwst['MwSt'].sum():.2f} €"])
    cols = ["MwSt-Satz", "Anzahl", "Brutto", "Netto", "MwSt-Betrag"]
    tbl = ax2.table(cellText=tabelle_data, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.1, 2.4)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0)
        if row == 0:
            cell.set_facecolor(C_SLATE)
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(tabelle_data):
            cell.set_facecolor("#DBEAFE")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("#FFFFFF")
    fig.tight_layout()
    return fig


PLOTLY_LAYOUT = dict(
    font=dict(family="DM Sans, sans-serif", color="#1E293B", size=13),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    margin=dict(t=50, b=40, l=20, r=20),
    hoverlabel=dict(bgcolor="white", font_size=13, font_color="#1E293B"),
    colorway=[C_BLUE, C_GREEN, "#7C3AED", "#D97706"],
)

PLOTLY_AXIS = dict(tickfont=dict(color="#475569"), title_font=dict(color="#475569"),
                   gridcolor="#E2E8F0", linecolor="#E2E8F0")


def plotly_wochentag_heatmap(df):
    wt = df.groupby("Wochentag")["Value"].count().reindex(WOCHENTAGE).fillna(0)
    colors = [C_BLUE if v == wt.max() else "#BFDBFE" for v in wt.values]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Verkäufe nach Wochentag", "Heatmap: Uhrzeit × Wochentag"))

    fig.add_trace(go.Bar(
        x=wt.index, y=wt.values, marker_color=colors,
        text=wt.values.astype(int), textposition="outside",
        hovertemplate="%{x}: %{y} Verkäufe<extra></extra>",
    ), row=1, col=1)

    heat = df.groupby(["Wochentag", "Stunde"])["Value"].count().unstack(fill_value=0).reindex(WOCHENTAGE).fillna(0)
    fig.add_trace(go.Heatmap(
        z=heat.values, x=[f"{h}h" for h in heat.columns], y=heat.index,
        colorscale="Blues", showscale=True,
        hovertemplate="Tag: %{y}<br>Uhrzeit: %{x}<br>Verkäufe: %{z}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=380)
    fig.update_xaxes(**PLOTLY_AXIS, tickangle=-30, row=1, col=1)
    fig.update_yaxes(**PLOTLY_AXIS)
    for ann in fig.layout.annotations:
        ann.font = dict(color="#0F172A", size=14)
    return fig


def plotly_top_produkte(df):
    prod_count = df.groupby("Product")["Value"].count().sort_values(ascending=True).tail(15)
    prod_sum = df.groupby("Product")["Value"].sum().sort_values(ascending=True).tail(15)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Top 15 nach Anzahl", "Top 15 nach Umsatz (€)"))

    fig.add_trace(go.Bar(
        y=prod_count.index, x=prod_count.values, orientation="h",
        marker_color=C_BLUE, text=prod_count.values.astype(int), textposition="outside",
        hovertemplate="%{y}: %{x} Verkäufe<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=prod_sum.index, x=prod_sum.values.round(2), orientation="h",
        marker_color=C_GREEN,
        text=[f"{v:.2f} €" for v in prod_sum.values], textposition="outside",
        hovertemplate="%{y}: %{x:.2f} €<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=500)
    fig.update_xaxes(**PLOTLY_AXIS)
    fig.update_yaxes(**PLOTLY_AXIS)
    for ann in fig.layout.annotations:
        ann.font = dict(color="#0F172A", size=14)
    return fig


def plotly_schwache_produkte(df):
    schwach = df.groupby("Product")["Value"].agg(["count", "sum"]).sort_values("count").head(10)
    fig = go.Figure(go.Bar(
        y=schwach.index, x=schwach["count"], orientation="h",
        marker_color="#FCA5A5",
        text=[f"{int(c)}x · {s:.2f} €" for c, s in zip(schwach["count"], schwach["sum"])],
        textposition="outside",
        hovertemplate="%{y}<br>%{x} Verkäufe<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                      title=dict(text="Schwächste Produkte", font=dict(color="#0F172A", size=15)))
    fig.update_xaxes(**PLOTLY_AXIS)
    fig.update_yaxes(**PLOTLY_AXIS)
    return fig


def plotly_slots_zahlungsart(df):
    slot = (df.dropna(subset=["Column"])
              .groupby("Column")["Value"].count()
              .sort_values(ascending=False).head(20))
    slot.index = slot.index.astype(int).astype(str)
    zahl_tag = df.groupby(["Datum", "Payment"])["Value"].sum().unstack(fill_value=0)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Top 20 Slots nach Verkäufen", "Bar vs. Karte pro Tag"))

    colors = [C_BLUE if v == slot.max() else "#BFDBFE" for v in slot.values]
    fig.add_trace(go.Bar(
        x=slot.index, y=slot.values, marker_color=colors,
        hovertemplate="Slot %{x}: %{y} Verkäufe<extra></extra>",
    ), row=1, col=1)

    if "Cash" in zahl_tag.columns:
        fig.add_trace(go.Scatter(
            x=[str(d) for d in zahl_tag.index], y=zahl_tag["Cash"],
            name="💵 Bar (Cash)", mode="lines+markers",
            line=dict(color=C_BLUE, width=3),
            marker=dict(size=7, color=C_BLUE),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.18)",
            hovertemplate="<b>Bar</b><br>%{x}: %{y:.2f} €<extra></extra>",
        ), row=1, col=2)
    if "Cashless" in zahl_tag.columns:
        fig.add_trace(go.Scatter(
            x=[str(d) for d in zahl_tag.index], y=zahl_tag["Cashless"],
            name="💳 Karte (Cashless)", mode="lines+markers",
            line=dict(color=C_GREEN, width=3),
            marker=dict(size=7, color=C_GREEN),
            fill="tozeroy", fillcolor="rgba(5,150,105,0.18)",
            hovertemplate="<b>Karte</b><br>%{x}: %{y:.2f} €<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                      legend=dict(
                          orientation="h", yanchor="bottom", y=1.02,
                          xanchor="right", x=1,
                          font=dict(size=13, color="#1E293B"),
                          bgcolor="rgba(255,255,255,0.8)",
                          bordercolor="#E2E8F0", borderwidth=1,
                      ))
    fig.update_xaxes(**PLOTLY_AXIS, tickangle=-45)
    fig.update_yaxes(**PLOTLY_AXIS)
    for ann in fig.layout.annotations:
        ann.font = dict(color="#0F172A", size=14)
    return fig


def plotly_tagesumsatz(df, monat_label):
    tag = df.groupby("Datum")["Value"].sum()
    colors = [C_BLUE if v == tag.max() else "#BFDBFE" for v in tag.values]
    fig = go.Figure(go.Bar(
        x=[str(d) for d in tag.index], y=tag.values,
        marker_color=colors,
        text=[f"{v:.0f} €" for v in tag.values], textposition="outside",
        hovertemplate="%{x}: %{y:.2f} €<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=380)
    fig.update_xaxes(**PLOTLY_AXIS, tickangle=-45)
    fig.update_yaxes(**PLOTLY_AXIS)
    return fig


def plotly_mwst(df):
    df = df.copy()
    df["Netto"] = df["Value"] / (1 + df["Tax Rate"] / 100)
    df["MwSt_Betrag"] = df["Value"] - df["Netto"]
    mwst = df.groupby("Tax Rate").agg(
        Anzahl=("Value", "count"), Brutto=("Value", "sum"),
        Netto=("Netto", "sum"), MwSt=("MwSt_Betrag", "sum"),
    ).reset_index()
    mwst["Label"] = mwst["Tax Rate"].astype(int).astype(str) + " %"

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "table"}]],
                        subplot_titles=("Brutto-Umsatz nach MwSt-Satz", "Aufschlüsselung"))

    fig.add_trace(go.Pie(
        labels=mwst["Label"], values=mwst["Brutto"].round(2),
        marker_colors=[C_BLUE, C_GREEN],
        hole=0.4,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.2f} €<extra></extra>",
    ), row=1, col=1)

    gesamt = ["Gesamt", int(mwst["Anzahl"].sum()),
              f"{mwst['Brutto'].sum():.2f} €", f"{mwst['Netto'].sum():.2f} €", f"{mwst['MwSt'].sum():.2f} €"]
    rows = [[row["Label"], int(row["Anzahl"]), f"{row['Brutto']:.2f} €",
             f"{row['Netto']:.2f} €", f"{row['MwSt']:.2f} €"] for _, row in mwst.iterrows()]
    rows.append(gesamt)

    fig.add_trace(go.Table(
        header=dict(values=["MwSt-Satz", "Anzahl", "Brutto", "Netto", "MwSt-Betrag"],
                    fill_color=C_SLATE, font=dict(color="white", size=13), align="center", height=36),
        cells=dict(
            values=[[r[i] for r in rows] for i in range(5)],
            fill_color=[["#F8FAFC", "#FFFFFF"] * len(rows)],
            align="center", height=32, font=dict(size=12),
        ),
    ), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=380)
    for ann in fig.layout.annotations:
        ann.font = dict(color="#0F172A", size=14)
    return fig


def create_pdf(df, monat_label):
    buf = io.BytesIO()
    figures = [
        fig_wochentag_heatmap(df),
        fig_top_produkte(df),
        fig_schwache_produkte(df),
        fig_slots_zahlungsart(df),
        fig_tagesumsatz(df, monat_label),
        fig_mwst(df, monat_label),
    ]
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_facecolor(C_SLATE)
        fig.patch.set_facecolor(C_SLATE)
        ax.axis("off")
        ax.text(0.5, 0.62, "Auswertung", ha="center", va="center",
                fontsize=42, fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(0.5, 0.50, monat_label, ha="center", va="center",
                fontsize=22, color="#94A3B8", transform=ax.transAxes)
        stats = f"{len(df)} Verkäufe  ·  {df['Value'].sum():.2f} € Umsatz  ·  Ø {df['Value'].mean():.2f} € pro Verkauf"
        ax.text(0.5, 0.38, stats, ha="center", va="center",
                fontsize=13, color="#64748B", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
        for f in figures:
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)
    buf.seek(0)
    return buf


# --- CSS ---
st.set_page_config(page_title="Auswertung", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1E293B;
}

.stApp {
    background: #F1F5F9;
}

.main-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94A3B8;
    margin: 0.3rem 0 0 0;
    font-size: 1rem;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border-top: 3px solid var(--accent, #2563EB);
}
.metric-card .label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 0.3rem;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #0F172A;
    padding: 0.5rem 0 0.8rem 0;
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 1rem;
}

[data-testid="stPlotlyChart"] {
    background: white;
    border-radius: 12px;
    padding: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}

[data-testid="stSidebar"] {
    background: #0F172A !important;
}
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: white !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stFileUploader"] {
    background: #1E293B;
    border-radius: 10px;
    padding: 0.5rem;
}

div[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    width: 100% !important;
    margin-top: 1rem;
}
div[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}

/* Mobile */
@media (max-width: 768px) {
    .main-header {
        padding: 1.4rem 1.2rem;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
    }
    .main-header h1 {
        font-size: 1.6rem;
    }
    .main-header p {
        font-size: 0.85rem;
    }
    .main-header div[style*="font-size:3rem"] {
        display: none;
    }
    .metric-card {
        padding: 1rem 1.1rem;
    }
    .metric-card .value {
        font-size: 1.5rem;
    }
    .section-header {
        font-size: 1rem;
    }
    [data-testid="stPlotlyChart"] {
        padding: 0.2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📊 Auswertung")
    st.markdown("---")
    uploaded_file = st.file_uploader("CSV hochladen", type="csv", label_visibility="collapsed")
    if not uploaded_file:
        st.markdown("""
        <div style='color:#64748B; font-size:0.85rem; margin-top:1rem; line-height:1.6'>
        Lade eine Verkaufs-CSV hoch um die Auswertung zu starten.<br><br>
        <b style='color:#94A3B8'>Enthaltene Auswertungen:</b><br>
        · Wochentag & Uhrzeit<br>
        · Top & schwächste Produkte<br>
        · Slots & Zahlungsart<br>
        · Tagesumsatz<br>
        · MwSt-Aufschlüsselung
        </div>
        """, unsafe_allow_html=True)

# --- Hauptbereich ---
st.markdown("""
<div class="main-header">
    <div style="flex:1; min-width:0">
        <h1 style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis">Auswertung</h1>
        <p>Verkaufsanalyse für Ihren Automaten</p>
    </div>
    <div style="font-size:3rem; flex-shrink:0">📊</div>
</div>
""", unsafe_allow_html=True)

if not uploaded_file:
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color:#94A3B8'>
        <div style='font-size:4rem; margin-bottom:1rem'>📂</div>
        <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:600; color:#475569'>
            CSV-Datei in der Seitenleiste hochladen
        </div>
        <div style='font-size:0.9rem; margin-top:0.5rem'>
            Die Auswertung startet automatisch nach dem Upload
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    with st.spinner("Daten werden ausgewertet..."):
        df = load_data(uploaded_file)

    monat_label = df["Timestamp"].dt.to_period("M").astype(str).iloc[0]
    gesamt = df["Value"].sum()
    schnitt = df["Value"].mean()
    cash_anteil = (df[df["Payment"] == "Cash"]["Value"].sum() / gesamt * 100) if gesamt > 0 else 0

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card" style="--accent:#2563EB">
            <div class="label">Verkäufe</div>
            <div class="value">{len(df)}</div>
            <div class="sub">{monat_label}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="--accent:#059669">
            <div class="label">Gesamtumsatz</div>
            <div class="value">{gesamt:,.2f} €</div>
            <div class="sub">Brutto inkl. MwSt</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card" style="--accent:#7C3AED">
            <div class="label">Ø pro Verkauf</div>
            <div class="value">{schnitt:.2f} €</div>
            <div class="sub">Durchschnitt</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card" style="--accent:#D97706">
            <div class="label">Barzahlung</div>
            <div class="value">{cash_anteil:.0f} %</div>
            <div class="sub">{100 - cash_anteil:.0f} % Kartenzahlung</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

    # PDF Export oben
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    with st.spinner("PDF wird erstellt..."):
        pdf_buf = create_pdf(df, monat_label)
    st.download_button(
        label="📄 PDF-Report herunterladen",
        data=pdf_buf,
        file_name=f"auswertung_{monat_label}.pdf",
        mime="application/pdf",
    )

    st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-header">Zeitliche Analyse</div>', unsafe_allow_html=True)

    st.plotly_chart(plotly_wochentag_heatmap(df), use_container_width=True, config={"responsive": True, "scrollZoom": False})


    st.markdown('<div class="section-header">Produkte</div>', unsafe_allow_html=True)

    st.plotly_chart(plotly_top_produkte(df), use_container_width=True)



    st.plotly_chart(plotly_schwache_produkte(df), use_container_width=True)


    st.markdown('<div class="section-header">Slots & Zahlungsart</div>', unsafe_allow_html=True)

    st.plotly_chart(plotly_slots_zahlungsart(df), use_container_width=True)


    st.markdown('<div class="section-header">Tagesumsatz</div>', unsafe_allow_html=True)

    st.plotly_chart(plotly_tagesumsatz(df, monat_label), use_container_width=True)


    st.markdown('<div class="section-header">MwSt-Aufschlüsselung</div>', unsafe_allow_html=True)

    st.plotly_chart(plotly_mwst(df), use_container_width=True)


