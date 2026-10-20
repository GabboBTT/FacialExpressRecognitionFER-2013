# Facial Expression Recognition su FER-2013

**Gabriel Betti — Matricola 178267** · Progetto di Computer Vision

Confronto tra una pipeline classica (**HOG + LinearSVC**) e reti convoluzionali (**ResNet-Lite** addestrata da zero, **MobileNetV2** in transfer learning) per il riconoscimento di 7 espressioni facciali su FER-2013, con analisi dei failure modes e dei bias sui sottogruppi.

## Risultati (test set, PrivateTest — 3 589 immagini)

| Modello | Accuracy [IC 95%] | Macro-F1 |
|---|---|---|
| HOG + LinearSVC | 40.6% [39.1, 42.2] | 0.349 |
| MobileNetV2 (transfer learning) | 63.2% [61.6, 64.8] | 0.609 |
| ResNet-Lite, senza mitigazioni | 66.7% [65.3, 68.1] | 0.636 |
| **ResNet-Lite + label smoothing + balanced sampling** (3 seed) | **67.3% ± 0.2** | **0.657** |

- La CNN supera nettamente HOG + SVM (McNemar, p < 10⁻¹⁵⁰) e MobileNetV2 (p ≈ 2·10⁻⁷).
- Label smoothing e campionamento bilanciato non cambiano in modo significativo l'accuratezza (p = 0.35), ma migliorano macro-F1 e recall delle classi rare.
- Le confusioni principali sono *Fear → Sad*, *Sad ↔ Neutral*, *Angry → Sad*; parte degli errori è dovuta al rumore delle etichette di FER-2013.
- Nessun bias statisticamente significativo per luminosità, contrasto o nitidezza; le immagini più scure mostrano un calo non significativo (p = 0.085).

## Protocollo
- Split ufficiale ICML 2013: `Training` → addestramento, `PublicTest` → validation (scelta del modello), `PrivateTest` → test (usato una sola volta).
- Rimosse dal training 703 immagini identiche a immagini di validation o test.
- Selezione dei modelli sulla macro-F1 di validation; intervalli di confidenza bootstrap e test di McNemar sul test.
- Seed fissi e operazioni deterministiche (TensorFlow 2.21, Keras 3.13, GPU T4).

## Struttura

| Percorso | Contenuto |
|---|---|
| `FER2013_Betti_178267.ipynb` | Notebook completo: dati, modelli, valutazione, failure modes, bias |
| `Report_FER2013_Betti_178267.pdf` | Report scientifico |
| `report/` | Sorgenti LaTeX del report (`main.tex`, `references.bib`, `figures/`) |
| `results/` | Metriche esportate dal notebook (`results.json`), audit del dataset, ambiente di esecuzione |

## Esecuzione
1. Aprire il notebook in Google Colab con runtime **GPU T4**.
2. Aggiungere ai segreti di Colab `KAGGLE_USERNAME` e `KAGGLE_KEY`: il dataset ([msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)) viene scaricato automaticamente.
3. *Runtime → Esegui tutte* (circa 3 ore di GPU). Risultati, pesi e figure vengono salvati su Google Drive in `FER2013_178267/`; gli esperimenti già completati vengono ricaricati, quindi l'esecuzione può essere divisa in più sessioni.
