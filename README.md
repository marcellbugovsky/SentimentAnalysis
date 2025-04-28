# SentimentAnalysis

Ein Python-Projekt zur Durchführung von Sentiment-Analysen (Stimmungsanalysen) für Text unter Verwendung von vortrainierten Modellen aus der Hugging Face `transformers`-Bibliothek. Dieses Projekt wurde als Lernübung für Konzepte des Natural Language Processing (NLP) konzipiert und dient als einfaches Kommandozeilen-Tool.

## Projektbeschreibung

Dieses Tool nimmt einen Text als Eingabe über die Kommandozeile entgegen und analysiert dessen Sentiment (z. B. POSITIVE oder NEGATIVE) mithilfe eines spezifizierten oder eines Standard-Modells von Hugging Face. Das Ergebnis, einschließlich des ermittelten Labels und eines Konfidenz-Scores, wird auf der Konsole ausgegeben.

## Features

* **Hugging Face Integration:** Nutzt die `pipeline`-Funktion der `transformers`-Bibliothek für einfachen Zugriff auf vortrainierte Sentiment-Analyse-Modelle.
* **Konfigurierbares Modell:** Ermöglicht die Auswahl verschiedener Sentiment-Analyse-Modelle von Hugging Face über einen Kommandozeilenparameter (`--model`).
* **Einfache Bedienung:** Klares Kommandozeilen-Interface zur Eingabe von Text und zur Auswahl des Modells.
* **Fehlerbehandlung:** Grundlegendes Logging und Fehlerbehandlung beim Laden des Modells und während der Analyse.

## Verwendete Technologien

* **Sprache:** Python 3
* **Kernbibliotheken:**
    * `transformers`: Für den Zugriff auf Modelle und Pipelines von Hugging Face.
    * `torch`: Als Backend für die `transformers`-Bibliothek.
    * `accelerate`: Wird oft von `transformers` für optimale Performance empfohlen oder benötigt.

## Setup & Installation

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/marcellbugovsky/SentimentAnalysis.git](https://github.com/marcellbugovsky/SentimentAnalysis.git)
    cd SentimentAnalysis
    ```
2.  **Virtuelle Umgebung erstellen (empfohlen):**
    ```bash
    python -m venv venv
    venv\Scripts\activate    # Windows
    ```
3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Hinweis: Beim ersten Ausführen lädt die `transformers`-Bibliothek das benötigte Modell herunter, was einige Zeit dauern und Speicherplatz benötigen kann.)*

## Verwendung

Führe das Skript über die Kommandozeile aus und übergib den zu analysierenden Text mit dem Argument `-t` oder `--text`. Optional kannst du mit `-m` oder `--model` ein anderes Hugging Face Sentiment-Analyse-Modell angeben.

**Beispiele:**

* **Analyse mit dem Standardmodell:**
    ```bash
    python main.py --text "This is a wonderful library and I love using it!"
    ```
    *(Erwartete Ausgabe: POSITIVE mit hohem Score)*

* **Analyse eines negativen Texts:**
    ```bash
    python main.py --text "I am very disappointed with the product quality."
    ```
    *(Erwartete Ausgabe: NEGATIVE mit hohem Score)*

* **Analyse mit einem anderen Modell (Beispiel):**
    *(Finde geeignete Modellnamen auf dem Hugging Face Hub, z.B. für mehrsprachige Modelle oder Modelle mit feineren Labels)*
    ```bash
    python main.py --text "Das ist eine interessante Idee." --model "bert-base-multilingual-uncased-sentiment"
    ```
    *(Hinweis: Die Verfügbarkeit und Eignung anderer Modelle muss geprüft werden.)*

## Standardmodell

Wenn kein Modell über das `--model`-Argument spezifiziert wird, verwendet das Skript standardmäßig:
`distilbert-base-uncased-finetuned-sst-2-english`

Dieses Modell ist für Englisch trainiert und gibt typischerweise `POSITIVE` oder `NEGATIVE` als Label zurück.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.