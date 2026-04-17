import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st

WOCHENTAGE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
FARBE = "#2563EB"
FARBE2 = "#16A34A"


def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["Value"] = df["Value"].str.replace(",", ".").astype(float)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Datum"] = df["Timestamp"].dt.date
    df["Stunde"] = df["Timestamp"].dt.hour
    df["Wochentag"] = df["Timestamp"].dt.day_name(locale="de_DE")
    df = df[df["Product"].notna() & (df["Transaction type"] == "SALE")]
    return df


def create_pdf(df, monat_label):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # --- Deckblatt ---
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.text(0.5, 0.65, "Automat Köller", ha="center", va="center",
                fontsize=36, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.52, f"Verkaufsauswertung {monat_label}", ha="center", va="center",
                fontsize=20, color="#555", transform=ax.transAxes)
        stats = f"{len(df)} Verkäufe  ·  {df['Value'].sum():.2f} € Umsatz  ·  Ø {df['Value'].mean():.2f} € pro Verkauf"
        ax.text(0.5, 0.40, stats, ha="center", va="center",
                fontsize=13, color="#777", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # --- Seite 1: Wochentag & Tageszeit ---
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Wann läuft der Automat?", fontsize=16, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        wt = df.groupby("Wochentag")["Value"].agg(["count", "sum"]).reindex(WOCHENTAGE).fillna(0)
        bars = ax1.bar(wt.index, wt["count"], color=FARBE, zorder=3)
        ax1.set_title("Verkäufe nach Wochentag", fontweight="bold")
        ax1.set_ylabel("Anzahl Verkäufe")
        ax1.set_xticks(range(len(wt.index)))
        ax1.set_xticklabels(wt.index, rotation=30, ha="right")
        ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax1.set_axisbelow(True)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, int(h),
                         ha="center", va="bottom", fontsize=9)

        ax2 = fig.add_subplot(gs[1])
        heat = df.groupby(["Wochentag", "Stunde"])["Value"].count().unstack(fill_value=0)
        heat = heat.reindex(WOCHENTAGE).fillna(0)
        im = ax2.imshow(heat.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax2.set_title("Heatmap: Uhrzeit × Wochentag", fontweight="bold")
        ax2.set_xlabel("Uhrzeit")
        ax2.set_xticks(range(len(heat.columns)))
        ax2.set_xticklabels([f"{h}h" for h in heat.columns], rotation=90, fontsize=7)
        ax2.set_yticks(range(len(WOCHENTAGE)))
        ax2.set_yticklabels(WOCHENTAGE, fontsize=9)
        plt.colorbar(im, ax=ax2, label="Verkäufe")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # --- Seite 2: Top Produkte ---
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle("Produkte", fontsize=16, fontweight="bold")
        prod = df.groupby("Product")["Value"].agg(["count", "sum"]).sort_values("count", ascending=True).tail(15)
        axes[0].barh(prod.index, prod["count"], color=FARBE, zorder=3)
        axes[0].set_title("Top 15 nach Anzahl", fontweight="bold")
        axes[0].set_xlabel("Anzahl Verkäufe")
        axes[0].xaxis.grid(True, linestyle="--", alpha=0.5)
        axes[0].set_axisbelow(True)
        prod2 = df.groupby("Product")["Value"].agg(["count", "sum"]).sort_values("sum", ascending=True).tail(15)
        axes[1].barh(prod2.index, prod2["sum"], color=FARBE2, zorder=3)
        axes[1].set_title("Top 15 nach Umsatz (€)", fontweight="bold")
        axes[1].set_xlabel("Umsatz in €")
        axes[1].xaxis.grid(True, linestyle="--", alpha=0.5)
        axes[1].set_axisbelow(True)
        for ax in axes:
            ax.tick_params(axis="y", labelsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # --- Seite 3: Schwächste Produkte ---
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        fig.suptitle("Schwächste Produkte (Kandidaten zum Austausch)", fontsize=16, fontweight="bold")
        schwach = df.groupby("Product")["Value"].agg(["count", "sum"]).sort_values("count").head(10)
        bars = ax.barh(schwach.index, schwach["count"], color="#DC2626", zorder=3)
        ax.set_xlabel("Anzahl Verkäufe")
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{w:.0f}x  ({schwach.loc[list(schwach.index)[list(schwach['count']).index(w)], 'sum']:.2f} €)",
                    va="center", fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # --- Seite 4: Slots & Zahlungsart ---
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle("Slots & Zahlungsart", fontsize=16, fontweight="bold")
        slot = df.groupby("Column")["Value"].agg(["count", "sum"]).sort_values("count", ascending=False).head(20)
        axes[0].bar(slot.index.astype(str), slot["count"], color=FARBE, zorder=3)
        axes[0].set_title("Top 20 Slots nach Verkäufen", fontweight="bold")
        axes[0].set_xlabel("Slot (Column)")
        axes[0].set_ylabel("Anzahl Verkäufe")
        axes[0].yaxis.grid(True, linestyle="--", alpha=0.5)
        axes[0].set_axisbelow(True)
        axes[0].tick_params(axis="x", rotation=45)
        zahl_tag = df.groupby(["Datum", "Payment"])["Value"].sum().unstack(fill_value=0)
        if "Cash" in zahl_tag.columns:
            axes[1].plot(range(len(zahl_tag)), zahl_tag["Cash"], label="Bar", color=FARBE, marker="o", markersize=4)
        if "Cashless" in zahl_tag.columns:
            axes[1].plot(range(len(zahl_tag)), zahl_tag["Cashless"], label="Karte", color=FARBE2, marker="o", markersize=4)
        axes[1].set_title("Umsatz: Bar vs. Karte pro Tag", fontweight="bold")
        axes[1].set_xlabel("Tag")
        axes[1].set_ylabel("Umsatz (€)")
        axes[1].set_xticks(range(len(zahl_tag)))
        axes[1].set_xticklabels([str(d)[5:] for d in zahl_tag.index], rotation=45, ha="right", fontsize=7)
        axes[1].legend()
        axes[1].yaxis.grid(True, linestyle="--", alpha=0.5)
        axes[1].set_axisbelow(True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # --- Seite 5: Tagesumsatz ---
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        fig.suptitle(f"Tagesumsatz {monat_label}", fontsize=16, fontweight="bold")
        tag = df.groupby("Datum")["Value"].sum()
        ax.bar(range(len(tag)), tag.values, color=FARBE, zorder=3)
        ax.set_xticks(range(len(tag)))
        ax.set_xticklabels([str(d)[5:] for d in tag.index], rotation=45, ha="right")
        ax.set_ylabel("Umsatz (€)")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for i, v in enumerate(tag.values):
            ax.text(i, v + 0.3, f"{v:.0f}€", ha="center", va="bottom", fontsize=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # --- Seite 6: MwSt-Aufschlüsselung ---
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle(f"Umsatzsteuer-Aufschlüsselung {monat_label}", fontsize=16, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.5)

        df["Netto"] = df["Value"] / (1 + df["Tax Rate"] / 100)
        df["MwSt_Betrag"] = df["Value"] - df["Netto"]
        mwst = df.groupby("Tax Rate").agg(
            Anzahl=("Value", "count"),
            Brutto=("Value", "sum"),
            Netto=("Netto", "sum"),
            MwSt=("MwSt_Betrag", "sum"),
        ).reset_index()
        mwst["Tax Rate"] = mwst["Tax Rate"].astype(int).astype(str) + " %"

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.pie(mwst["Brutto"], labels=mwst["Tax Rate"], autopct="%1.1f%%",
                colors=[FARBE, FARBE2], startangle=90)
        ax1.set_title("Brutto-Umsatz nach MwSt-Satz", fontweight="bold")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.pie(mwst["Anzahl"], labels=mwst["Tax Rate"], autopct="%1.1f%%",
                colors=[FARBE, FARBE2], startangle=90)
        ax2.set_title("Anzahl Verkäufe nach MwSt-Satz", fontweight="bold")

        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis("off")
        tabelle_data = []
        for _, row in mwst.iterrows():
            tabelle_data.append([
                row["Tax Rate"],
                f"{row['Anzahl']:.0f}",
                f"{row['Brutto']:.2f} €",
                f"{row['Netto']:.2f} €",
                f"{row['MwSt']:.2f} €",
            ])
        tabelle_data.append([
            "Gesamt",
            f"{mwst['Anzahl'].sum():.0f}",
            f"{mwst['Brutto'].sum():.2f} €",
            f"{mwst['Netto'].sum():.2f} €",
            f"{mwst['MwSt'].sum():.2f} €",
        ])
        cols = ["MwSt-Satz", "Anzahl", "Brutto", "Netto", "MwSt-Betrag"]
        tbl = ax3.table(cellText=tabelle_data, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 2.2)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0 or row == len(tabelle_data):
                cell.set_facecolor("#1E3A5F")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F0F4FF")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    buf.seek(0)
    return buf


# --- Streamlit UI ---
st.set_page_config(page_title="Automaten Auswertung", page_icon="🏪", layout="centered")
st.title("🏪 Automaten Auswertung")
st.write("CSV-Datei hochladen und PDF-Report herunterladen.")

uploaded_file = st.file_uploader("Verkaufs-CSV hochladen", type="csv")

if uploaded_file:
    with st.spinner("Daten werden ausgewertet..."):
        df = load_data(uploaded_file)

    monat_label = df["Timestamp"].dt.to_period("M").astype(str).iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Verkäufe", len(df))
    col2.metric("Gesamtumsatz", f"{df['Value'].sum():.2f} €")
    col3.metric("Ø pro Verkauf", f"{df['Value'].mean():.2f} €")

    st.subheader("Top 5 Produkte")
    top5 = df.groupby("Product")["Value"].agg(["count", "sum"]).sort_values("count", ascending=False).head(5)
    top5.columns = ["Anzahl", "Umsatz (€)"]
    st.dataframe(top5, use_container_width=True)

    with st.spinner("PDF wird erstellt..."):
        pdf_buf = create_pdf(df, monat_label)

    st.download_button(
        label="📄 PDF-Report herunterladen",
        data=pdf_buf,
        file_name=f"auswertung_{monat_label}.pdf",
        mime="application/pdf",
    )
