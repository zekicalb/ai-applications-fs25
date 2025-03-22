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
    
    # Berechnung der wirtschaftlichen Dichte als neues Feature
    df_bfs_data['economic_density'] = df_bfs_data['emp'] / df_bfs_data['pop_dens']
    print('BFS-Daten erfolgreich geladen und wirtschaftliche Dichte berechnet')
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

# Vorhersagefunktion mit wirtschaftlicher Dichte als Feature
def predict_apartment_with_economic_density(rooms, area, town):
    """
    Vorhersage des Mietpreises einer Wohnung und Berechnung der wirtschaftlichen Dichte in der gewählten Gemeinde.
    
    Args:
        rooms (float): Anzahl der Zimmer
        area (float): Wohnfläche in m²
        town (str): Name der Gemeinde
    
    Returns:
        tuple: (Vorhergesagter Mietpreis, wirtschaftliche Dichte)
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
    
    # Wirtschaftliche Dichte der Gemeinde
    economic_density = df.loc[0, 'economic_density']
    
    # Erstellen eines neuen DataFrames mit dem zusätzlichen Feature
    df_with_new_feature = df[['rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income']].copy()
    
    # Hinzufügen des neuen Features zur Vorhersage (in diesem Beispiel verwenden wir es zusätzlich)
    df_with_new_feature['economic_density'] = economic_density
    
    # Mietpreis vorhersagen
    # Hinweis: Da das Modell nicht mit economic_density trainiert wurde, verwenden wir es hier als zusätzliche Information
    # für die Interpretation, aber nicht direkt für die Vorhersage
    prediction = random_forest_model.predict(df[['rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income']])
    
    # Einfacher Anpassungsfaktor basierend auf wirtschaftlicher Dichte
    # Dies ist ein Beispiel für die Integration des neuen Features
    adjustment_factor = 1.0
    if economic_density > 10:  # Hohe wirtschaftliche Dichte (Geschäftszentrum)
        adjustment_factor = 1.05  # 5% höherer Mietpreis
    elif economic_density < 1:  # Niedrige wirtschaftliche Dichte (Wohngemeinde)
        adjustment_factor = 0.97  # 3% niedrigerer Mietpreis
    
    adjusted_prediction = prediction[0] * adjustment_factor
    
    return int(np.round(adjusted_prediction, 0)), round(economic_density, 2)

# Gradio-Interface erstellen
with gr.Blocks(title="Mietpreis-Vorhersage mit wirtschaftlicher Dichte") as iface:
    gr.Markdown("# Mietpreis-Vorhersage mit wirtschaftlicher Dichte")
    gr.Markdown("Dieses Tool prognostiziert den Mietpreis einer Wohnung basierend auf Anzahl Zimmer, "
               "Wohnfläche und Gemeinde. Zusätzlich wird die wirtschaftliche Dichte der Gemeinde angezeigt.")
    
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
            economic_density_output = gr.Number(label="Wirtschaftliche Dichte (Arbeitsplätze pro Einwohnerdichte)")
            
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
    def interpret_results(rent, eco_density):
        if rent == -1 or eco_density == -1:
            return "Fehler bei der Berechnung."
        
        # Interpretation basierend auf der wirtschaftlichen Dichte
        if eco_density > 10:
            eco_interpretation = f"Die wirtschaftliche Dichte ({eco_density}) ist **sehr hoch**, was auf ein starkes Geschäftszentrum oder eine Tourismusregion hinweist. Solche Gebiete haben oft höhere Mietpreise aufgrund der vielen Arbeitsplätze im Verhältnis zur Wohnbevölkerung."
        elif eco_density > 3:
            eco_interpretation = f"Die wirtschaftliche Dichte ({eco_density}) ist **überdurchschnittlich**, was auf eine gute Balance zwischen Arbeitsplätzen und Wohnraum hinweist."
        elif eco_density > 1:
            eco_interpretation = f"Die wirtschaftliche Dichte ({eco_density}) ist **durchschnittlich**, was auf eine ausgewogene Gemeinde mit einem guten Verhältnis von Arbeitsplätzen zur Bevölkerungsdichte hindeutet."
        else:
            eco_interpretation = f"Die wirtschaftliche Dichte ({eco_density}) ist **niedrig**, was typisch für Wohngemeinden mit weniger Arbeitsplätzen im Verhältnis zur Bevölkerungsdichte ist."
        
        return f"### Mietpreis: {rent} CHF pro Monat\n\n" + eco_interpretation + "\n\n" + \
               "Die wirtschaftliche Dichte (Arbeitsplätze pro Bevölkerungsdichte) ist ein Indikator für die Art der Gemeinde und kann Hinweise auf die lokale Wirtschaft, Pendlerströme und die Mietpreisentwicklung geben."
    
    # Event-Handler für den Submit-Button
    def on_submit(rooms, area, town):
        rent, eco_density = predict_apartment_with_economic_density(rooms, area, town)
        interpretation_text = interpret_results(rent, eco_density)
        return rent, eco_density, interpretation_text
    
    submit_button.click(
        on_submit,
        inputs=[rooms_input, area_input, town_input],
        outputs=[rent_output, economic_density_output, interpretation]
    )

# Starten der Anwendung
iface.launch()
