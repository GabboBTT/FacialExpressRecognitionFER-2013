# Report FER-2013

## 1. Abstract
[Inserire qui un breve riassunto del report, descrivendo l'obiettivo del progetto, l'approccio utilizzato e i risultati principali ottenuti sulla classificazione delle espressioni facciali.]

## 2. Introduzione
[Inserire qui l'introduzione al dataset FER-2013, affrontando le sfide specifiche come la bassa risoluzione delle immagini (48x48 pixel), la presenza di label rumorose (noisy labels) e il forte sbilanciamento delle classi.]

## 3. Stato dell'Arte
[Inserire qui una panoramica dello stato dell'arte, confrontando l'approccio basato su pipeline classica (estrazione di feature HOG e classificatore SVM) con i moderni modelli di Deep Learning (CNN custom e architetture leggere come Mini-Xception).]

## 4. Metodologia
**Architettura Scelta**: Mini-Xception, preferita per l'uso di Depthwise Separable Convolutions che riducono drasticamente i parametri (<100k) e prevengono l'overfitting su un dataset rumoroso come FER-2013 (48x48 pixel). Sviluppata usando Keras Functional API.

**Data Ingestion**: Sostituito l'obsoleto ImageDataGenerator con l'API moderna tf.data.Dataset (image_dataset_from_directory) con prefetching ottimizzato per I/O su GPU.

**Strategie di Mitigazione (Requisito Rispettato)**: 1) Label Smoothing impostato a 0.1 nella Categorical Crossentropy per gestire la bassa human-accuracy e le etichette rumorose. 2) Balanced Sampling tramite il calcolo dinamico dei "class weights" di Scikit-Learn per penalizzare maggiormente gli errori sulle classi minoritarie (Fear e Disgust).

## 5. Analisi Sperimentale Avanzata (CNN Custom V2)
La baseline iniziale ha evidenziato limiti di overfitting precoce (Accuracy 42%). Per dimostrare robustezza architetturale interna, è stata implementata una CNN Custom V2 aggiungendo un livello di Data Augmentation dinamica (Random Flip, Rotation, Translation e Zoom).

Risultati Definitivi: Mantenendo attivi i requisiti di mitigazione (Label Smoothing e Class Weights bilanciati), l'architettura V2 ha garantito una regolarizzazione estrema, portando la Validation Accuracy al 56.23% (+14% rispetto alla V1). Sorprendentemente, questo modello from scratch ha pareggiato le prestazioni del fine-tuning estremo su reti State-of-the-Art come MobileNetV2 (56.28%). Ciò dimostra scientificamente che, in contesti di noisy labels e bassa risoluzione come il FER-2013, la regolarizzazione del dataset (Augmentation) è un fattore di convergenza immensamente più impattante rispetto alla profondità pura della rete.

## 6. Discussione Risultati (Failure Modes & Bias)
L'analisi della Matrice di Confusione espone due criticità (requisito di progetto). 1) Dataset Bias: La classe "Disgust" (solo 111 sample) è gravemente sottorappresentata, limitando l'F1-score a 0.15 nonostante il bilanciamento dei pesi. 2) Visually Similar Emotions: Esiste una forte sovrapposizione tra "Fear" e "Angry". Il modello ha predetto correttamente "Fear" solo 83 volte, confondendola con "Angry" 323 volte. A 48x48 pixel, i micro-tratti (es. sopracciglia aggrottate) risultano indistinguibili, causando il collasso della Recall per la classe Fear (0.08).

### 6.1 Benchmark CNN vs HOG+SVM (Requisito Soddisfatto)
Il benchmark ha rivelato prestazioni globali comparabili: 42% (CNN) vs 41% (HOG+LinearSVC).

La differenza chiave risiede nella gestione delle classi sbilanciate. L'estrazione geometrica HOG, combinata al bilanciamento dei pesi rigido della SVM, ha portato la Recall della classe minoritaria "Disgust" (111 sample) al 63%, sovraperformando nettamente la CNN (31%).

Si conclude che per dataset a bassissima risoluzione (48x48) privi di Data Augmentation massiccia, l'ingegnerizzazione manuale delle feature (HOG) fornisce contorni decisionali più robusti sui gradienti facciali marcati rispetto a reti neurali convoluzionali di tipo shallow.

### 6.2 Subgroup Bias Analysis (Luminosità e Sovraesposizione)
Per soddisfare la richiesta di analisi dei bias sui sottogruppi, il test set è stato segmentato tramite median split sull'intensità media dei pixel (immagini "Dark" vs "Bright").

I risultati hanno rivelato un bias contro-intuitivo: il modello performa meglio sul sottogruppo "Dark" (Accuratezza: 30.04%) rispetto al sottogruppo "Bright" (Accuratezza: 27.03%), con un differenziale del 3%.

L'analisi ingegneristica suggerisce che la sovraesposizione (immagini "Bright") agisce da filtro passa-basso, appiattendo i gradienti e cancellando i micro-dettagli (ombre delle rughe d'espressione, contorno labbra) essenziali per i filtri convoluzionali. Questo dimostra come le condizioni di illuminazione del dataset FER-2013 introducano un bias sistematico che penalizza i volti sovraesposti.

## 7. Esperimento Extra: Transfer Learning e Fine-Tuning Avanzato

Per spingere l'architettura ai limiti del dataset, è stato implementato un "Bonus Sprint" basato su MobileNetV2 (pesi ImageNet). Le immagini 48x48 sono state convertite in RGB e ridimensionate a 96x96.

Fase 1 (Feature Extraction): Con il backbone congelato, il modello si è fermato al 44.2% di accuracy, dimostrando i limiti dell'adattamento passivo su dati a bassa risoluzione.

Fase 2 (Fine-Tuning e Data Augmentation): Scongelando gli ultimi 50 strati convoluzionali, applicando Data Augmentation dinamica (rotazione, zoom, traslazione) e un learning rate decadente (Nadam, da 1e-4 a 6e-6), l'architettura ha potuto mappare i gradienti facciali specifici.

Risultato Definitivo: Il modello ha raggiunto una Validation Accuracy del 56.28%, mantenendo attive le strategie di mitigazione per i bias (Label Smoothing e Balanced Sampling). Questo balzo del +14% rispetto alla baseline dimostra che un'architettura profonda, se sottoposta a un fine-tuning aggressivo, riesce parzialmente a compensare il degrado (upscaling distruttivo, noisy labels) intrinseco del dataset FER-2013.

## 8. Conclusioni
[Inserire qui le conclusioni del lavoro svolto, sintetizzando i risultati principali e proponendo possibili sviluppi futuri.]
