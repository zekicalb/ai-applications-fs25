import gradio as gr
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import pickle

# Lade das Modell
model_filename = "random_forest_regression.pkl"
try:
    with open(model_filename, 'rb') as f:
        random_forest_model = pickle.load(f)
    print('Modell erfolgreich geladen')
    print('Anzahl der Features: ', random_forest_model.n_features_in_)
    print('Features sind: ', ['rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income'])
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    random_forest_model = None

# Lade die BFS-Daten
try:
    df_bfs_data = pd.read_csv('bfs_municipality_and_tax_data.csv', sep=',', encoding='utf-8')
    df_bfs_data['tax_income'] = df_bfs_data['tax_income'].str.replace("'", "").astype(float)
    print('BFS-Daten erfolgreich geladen')
except Exception as e:
    print(f"Fehler beim Laden der BFS-Daten: {e}")
    df_bfs_data = None

# Orte-Dictionary
locations = {
    "Zürich": 261,
    "Kloten": 62,
    "Uster": 198,
    "Illnau-Effretikon": 296,
    "Feuerthalen": 27,
    "Pfäffikon": 177,
    "Ottenbach": 11,
    "Dübendorf": 191,
    "Richterswil": 138,
    "Maur": 195,
    "Embrach": 56,
    "Bülach": 53,
    "Winterthur": 230,
    "Oetwil am See": 157,
    "Russikon": 178,
    "Obfelden": 10,
    "Wald (ZH)": 120,
    "Niederweningen": 91,
    "Dällikon": 84,
    "Buchs (ZH)": 83,
    "Rüti (ZH)": 118,
    "Hittnau": 173,
    "Bassersdorf": 52,
    "Glattfelden": 58,
    "Opfikon": 66,
    "Hinwil": 117,
    "Regensberg": 95,
    "Langnau am Albis": 136,
    "Dietikon": 243,
    "Erlenbach (ZH)": 151,
    "Kappel am Albis": 6,
    "Stäfa": 158,
    "Zell (ZH)": 231,
    "Turbenthal": 228,
    "Oberglatt": 92,
    "Winkel": 72,
    "Volketswil": 199,
    "Kilchberg (ZH)": 135,
    "Wetzikon (ZH)": 121,
    "Zumikon": 160,
    "Weisslingen": 180,
    "Elsau": 219,
    "Hettlingen": 221,
    "Rüschlikon": 139,
    "Stallikon": 13,
    "Dielsdorf": 86,
    "Wallisellen": 69,
    "Dietlikon": 54,
    "Meilen": 156,
    "Wangen-Brüttisellen": 200,
    "Flaach": 28,
    "Regensdorf": 96,
    "Niederhasli": 90,
    "Bauma": 297,
    "Aesch (ZH)": 241,
    "Schlieren": 247,
    "Dürnten": 113,
    "Unterengstringen": 249,
    "Gossau (ZH)": 115,
    "Oberengstringen": 245,
    "Schleinikon": 98,
    "Aeugst am Albis": 1,
    "Rheinau": 38,
    "Höri": 60,
    "Rickenbach (ZH)": 225,
    "Rafz": 67,
    "Adliswil": 131,
    "Zollikon": 161,
    "Urdorf": 250,
    "Hombrechtikon": 153,
    "Birmensdorf (ZH)": 242,
    "Fehraltorf": 172,
    "Weiach": 102,
    "Männedorf": 155,
    "Küsnacht (ZH)": 154,
    "Hausen am Albis": 4,
    "Hochfelden": 59,
    "Fällanden": 193,
    "Greifensee": 194,
    "Mönchaltorf": 196,
    "Dägerlen": 214,
    "Thalheim an der Thur": 39,
    "Uetikon am See": 159,
    "Seuzach": 227,
    "Uitikon": 248,
    "Affoltern am Albis": 2,
    "Geroldswil": 244,
    "Niederglatt": 89,
    "Thalwil": 141,
    "Rorbas": 68,
    "Pfungen": 224,
    "Weiningen (ZH)": 251,
    "Bubikon": 112,
    "Neftenbach": 223,
    "Mettmenstetten": 9,
    "Otelfingen": 94,
    "Flurlingen": 29,
    "Stadel": 100,
    "Grüningen": 116,
    "Henggart": 31,
    "Dachsen": 25,
    "Bonstetten": 3,
    "Bachenbülach": 51,
    "Horgen": 295
}

# Vorhersagefunktion mit tax_per_capita Feature
def predict_apartment_with_tax_per_capita(rooms, area, town):
    """
    Vorhersage des Mietpreises einer Wohnung und Berechnung der Steuereinnahmen pro Kopf in der gewählten Gemeinde.
    
    Args:
        rooms (float): Anzahl der Zimmer
        area (float): Wohnfläche in m²
        town (str): Name der Gemeinde
    
    Returns:
        tuple: (Vorhergesagter Mietpreis, Steuereinnahmen pro Kopf)
    """
    # Prüfen, ob Modell und Daten geladen wurden
    if random_forest_model is None or df_bfs_data is None:
        return -1, -1
    
    # BFS-Nummer der gewählten Gemeinde abrufen
    bfs_number = locations[town]
    
    # Daten für die gewählte Gemeinde filtern
    df = df_bfs_data[df_bfs_data['bfs_number']==bfs_number].copy()
    
    if len(df) == 0:
        return -1, -1
    
    # Zurücksetzen des Index für einfacheren Zugriff
    df.reset_index(inplace=True)
    
    # Aktualisieren der Eingabewerte
    df.loc[0, 'rooms'] = rooms
    df.loc[0, 'area'] = area
    
    if len(df) != 1:  # wenn es mehr als einen Datensatz mit derselben bfs_number gibt
        return -1, -1
    
    # Berechnung des tax_per_capita (Steuereinnahmen pro Kopf)
    tax_per_capita = df.loc[0, 'tax_income'] / df.loc[0, 'pop']
    
    # Mietpreis vorhersagen
    prediction = random_forest_model.predict(df[['rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income']])
    
    return int(np.round(prediction[0], 0)), int(np.round(tax_per_capita, 0))

# Gradio-Interface erstellen
with gr.Blocks(title="Mietpreis-Vorhersage mit Steuerinformationen") as iface:
    gr.Markdown("# Mietpreis-Vorhersage mit Steuerinformationen")
    gr.Markdown("Dieses Tool prognostiziert den Mietpreis einer Wohnung basierend auf Anzahl Zimmer, "
               "Wohnfläche und Gemeinde. Zusätzlich wird die Steuerleistung pro Kopf angezeigt.")
    
    with gr.Row():
        with gr.Column():
            # Eingabefelder
            rooms_input = gr.Slider(minimum=1, maximum=7, step=0.5, label="Anzahl Zimmer", value=3.5)
            area_input = gr.Slider(minimum=20, maximum=300, step=5, label="Wohnfläche (m²)", value=80)
            town_input = gr.Dropdown(choices=sorted(list(locations.keys())), label="Gemeinde", value="Zürich")
            submit_button = gr.Button("Berechnen")
        
        with gr.Column():
            # Ausgabefelder
            rent_output = gr.Number(label="Prognostizierter Mietpreis (CHF/Monat)")
            tax_output = gr.Number(label="Steuerleistung pro Kopf (CHF/Jahr)")
            
            # Interpretationstext
            interpretation = gr.Markdown()
    
    # Beispiele
    examples = gr.Examples(
        examples=[
            [4.5, 120, "Dietlikon"],
            [3.5, 60, "Winterthur"],
            [2.5, 50, "Zürich"],
            [5.5, 150, "Küsnacht (ZH)"],
            [3.0, 70, "Uster"]
        ],
        inputs=[rooms_input, area_input, town_input]
    )
    
    # Funktion zur Interpretation der Ergebnisse
    def interpret_results(rent, tax):
        if rent == -1 or tax == -1:
            return "Fehler bei der Berechnung."
        
        # Durchschnittlicher Steuerertrag pro Kopf im Kanton Zürich (fiktiver Wert)
        avg_tax_per_capita = 5000
        
        if tax > avg_tax_per_capita * 1.2:
            tax_interpretation = f"Die Steuerleistung pro Kopf ({tax} CHF) ist **deutlich höher** als der kantonale Durchschnitt, was auf eine wohlhabendere Gemeinde hindeutet."
        elif tax > avg_tax_per_capita:
            tax_interpretation = f"Die Steuerleistung pro Kopf ({tax} CHF) ist **höher** als der kantonale Durchschnitt."
        elif tax > avg_tax_per_capita * 0.8:
            tax_interpretation = f"Die Steuerleistung pro Kopf ({tax} CHF) liegt **im Bereich** des kantonalen Durchschnitts."
        else:
            tax_interpretation = f"Die Steuerleistung pro Kopf ({tax} CHF) ist **niedriger** als der kantonale Durchschnitt."
        
        return f"### Mietpreis: {rent} CHF pro Monat\n\n" + tax_interpretation + "\n\n" + \
               "Die Steuerleistung pro Kopf ist ein Indikator für die wirtschaftliche Stärke einer Gemeinde und kann Hinweise auf Infrastruktur, öffentliche Dienstleistungen und sozioökonomische Bedingungen geben."
    
    # Event-Handler für den Submit-Button
    def on_submit(rooms, area, town):
        rent, tax = predict_apartment_with_tax_per_capita(rooms, area, town)
        interpretation_text = interpret_results(rent, tax)
        return rent, tax, interpretation_text
    
    submit_button.click(
        on_submit,
        inputs=[rooms_input, area_input, town_input],
        outputs=[rent_output, tax_output, interpretation]
    )

# Starten der Anwendung
iface.launch()
