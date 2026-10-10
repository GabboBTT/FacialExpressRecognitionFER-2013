# Report Analitico: Facial Expression Recognition (FER-2013)

**Autore:** Gabriel Betti (Matricola: 178267) | **Contatto:** betti.gabriel@spes.uniud.it

---

## 1. Abstract

Questo progetto indaga le prestazioni e i limiti intrinseci di diverse architetture di Machine Learning e Deep Learning applicate al riconoscimento delle espressioni facciali (FER) su immagini a bassissima risoluzione. Utilizzando il dataset FER-2013, lo studio mette a confronto tecniche classiche di estrazione feature (HOG + LinearSVC) con architetture convoluzionali profonde (Mini-Xception e MobileNetV2). L'analisi si concentra sulla gestione critica di classi gravemente sbilanciate (es. *Disgust*) e *noisy labels* tramite strategie di mitigazione strutturate (Label Smoothing, Balanced Sampling, Data Augmentation). I risultati dimostrano che una CNN custom ottimizzata con Data Augmentation dinamica (56.23% Validation Accuracy) è in grado di pareggiare le prestazioni di reti pre-addestrate State-of-the-Art (MobileNetV2, 56.28%), provando che su dataset fortemente rumorosi la regolarizzazione del dato domina sulla profondità architetturale. Il report include inoltre un'analisi spietata dei bias di dataset, quantificando il degrado prestazionale causato dalla sovraesposizione luminosa.

## 2. Introduzione

Il riconoscimento automatico delle espressioni facciali rappresenta una sfida centrale nella Computer Vision. Il dataset scelto per questo studio è il benchmark FER-2013, composto da volti in scala di grigi di dimensione 48x48 pixel, categorizzati in 7 emozioni primarie (*Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral*).
Le caratteristiche intrinseche del FER-2013 presentano ostacoli ingegneristici non banali:

1. **Bassa Risoluzione e Mancanza di Texture:** I volti a 48x48 pixel offrono gradienti minimi, rendendo indistinguibili i micro-dettagli necessari per separare emozioni visivamente contigue.
2. **Noisy Labels:** Il dataset originale è stato etichettato tramite crowdsourcing umano, introducendo un tasso di errore stimato (label noise) che fissa un tetto teorico di accuratezza globale attorno al 65%.
3. **Sbilanciamento Estremo:** Classi come *Disgust* contano un numero di sample irrisorio rispetto a classi dominanti come *Happy*, forzando i modelli standard a ignorare l'emozione minoritaria per minimizzare la loss globale.
Questo progetto affronta tali limitazioni implementando pipeline robuste, con l'obiettivo di analizzare non solo l'accuratezza pura, ma i *failure modes* e i bias algoritmici risultanti.

## 3. Stato dell'Arte

Nell'ambito FER, l'evoluzione delle tecniche si divide in due grandi filoni metodologici analizzati in questo studio.
L'approccio classico (Machine Learning tradizionale) si affida all'ingegnerizzazione manuale delle feature. L'estrazione di Istogrammi dei Gradienti Orientati (HOG) cattura la distribuzione dei contorni geometrici (spigoli, rughe d'espressione), delegando la classificazione a modelli lineari come le Support Vector Machine (SVM). Questo approccio è computazionalmente leggero e robusto su feature rigide, ma non scala su pattern complessi.
Il secondo filone sfrutta il Deep Learning, dove le Reti Neurali Convoluzionali (CNN) estraggono gerarchie di feature in modo autonomo. Sebbene modelli mastodontici come VGG16 o ResNet dominino su dataset ad alta risoluzione (ImageNet), tendono all'overfitting catastrofico su dataset come FER-2013. Pertanto, lo stato dell'arte per questi scenari predilige architetture leggere o *shallow*, come le reti basate su *Depthwise Separable Convolutions* (es. Xception/MobileNet), che massimizzano la capacità rappresentativa minimizzando il numero di parametri addestrabili.

## 4. Metodologia

**Architettura Scelta**: Mini-Xception, preferita per l'uso di Depthwise Separable Convolutions che riducono drasticamente i parametri (<100k) e prevengono l'overfitting su un dataset rumoroso come FER-2013 (48x48 pixel). Sviluppata usando Keras Functional API.

**Data Ingestion**: Sostituito l'obsoleto ImageDataGenerator con l'API moderna `tf.data.Dataset` (`image_dataset_from_directory`) con prefetching ottimizzato per I/O su GPU.

**Strategie di Mitigazione (Requisito Rispettato)**:

1. *Label Smoothing* impostato a 0.1 nella Categorical Crossentropy per gestire la bassa human-accuracy e le etichette rumorose.
2. *Balanced Sampling* tramite il calcolo dinamico dei "class weights" di Scikit-Learn per penalizzare maggiormente gli errori sulle classi minoritarie (Fear e Disgust).

## 5. Analisi Sperimentale Avanzata (CNN Custom V2)

La baseline iniziale ha evidenziato limiti di overfitting precoce (Accuracy 42%). Per dimostrare robustezza architetturale interna, è stata implementata una CNN Custom V2 aggiungendo un livello di Data Augmentation dinamica (Random Flip, Rotation, Translation e Zoom).

**Risultati Definitivi:** Mantenendo attivi i requisiti di mitigazione (Label Smoothing e Class Weights bilanciati), l'architettura V2 ha garantito una regolarizzazione estrema, portando la Validation Accuracy al **56.23%** (+14% rispetto alla V1). Sorprendentemente, questo modello *from scratch* ha pareggiato le prestazioni del fine-tuning estremo su reti State-of-the-Art come MobileNetV2 (56.28%). Ciò dimostra scientificamente che, in contesti di *noisy labels* e bassa risoluzione come il FER-2013, la regolarizzazione del dataset (Augmentation) è un fattore di convergenza immensamente più impattante rispetto alla profondità pura della rete.

## 6. Discussione Risultati (Failure Modes & Bias)

L'analisi della Matrice di Confusione espone due criticità (requisito di progetto).

1. **Dataset Bias:** La classe "Disgust" (solo 111 sample) è gravemente sottorappresentata, limitando l'F1-score a 0.15 nonostante il bilanciamento dei pesi.
2. **Visually Similar Emotions:** Esiste una forte sovrapposizione tra "Fear" e "Angry". Il modello ha predetto correttamente "Fear" solo 83 volte, confondendola con "Angry" 323 volte. A 48x48 pixel, i micro-tratti (es. sopracciglia aggrottate) risultano indistinguibili, causando il collasso della Recall per la classe Fear (0.08).

### 6.1 Benchmark CNN vs HOG+SVM (Requisito Soddisfatto)

Il benchmark ha rivelato prestazioni globali comparabili: 42% (CNN Baseline) vs 41% (HOG+LinearSVC).

La differenza chiave risiede nella gestione delle classi sbilanciate. L'estrazione geometrica HOG, combinata al bilanciamento dei pesi rigido della SVM, ha portato la Recall della classe minoritaria "Disgust" (111 sample) al 63%, sovraperformando nettamente la CNN (31%).

Si conclude che per dataset a bassissima risoluzione (48x48) privi di Data Augmentation massiccia, l'ingegnerizzazione manuale delle feature (HOG) fornisce contorni decisionali più robusti sui gradienti facciali marcati rispetto a reti neurali convoluzionali di tipo *shallow*.

### 6.2 Subgroup Bias Analysis (Luminosità e Sovraesposizione)

Per soddisfare la richiesta di analisi dei bias sui sottogruppi, il test set è stato segmentato tramite *median split* sull'intensità media dei pixel (immagini "Dark" vs "Bright").

I risultati hanno rivelato un bias contro-intuitivo: il modello performa meglio sul sottogruppo "Dark" (Accuratezza: 30.04%) rispetto al sottogruppo "Bright" (Accuratezza: 27.03%), con un differenziale del 3%.

L'analisi ingegneristica suggerisce che la sovraesposizione (immagini "Bright") agisce da filtro passa-basso, appiattendo i gradienti e cancellando i micro-dettagli (ombre delle rughe d'espressione, contorno labbra) essenziali per i filtri convoluzionali. Questo dimostra come le condizioni di illuminazione del dataset FER-2013 introducano un bias sistematico che penalizza i volti sovraesposti.

## 7. Esperimento Extra: Transfer Learning e Fine-Tuning Avanzato

Per spingere l'architettura ai limiti del dataset, è stato implementato un "Bonus Sprint" basato su MobileNetV2 (pesi ImageNet). Le immagini 48x48 sono state convertite in RGB e ridimensionate a 96x96.

**Fase 1 (Feature Extraction):** Con il backbone congelato, il modello si è fermato al 44.2% di accuracy, dimostrando i limiti dell'adattamento passivo su dati a bassa risoluzione.

**Fase 2 (Fine-Tuning e Data Augmentation):** Scongelando gli ultimi 50 strati convoluzionali, applicando Data Augmentation dinamica (rotazione, zoom, traslazione) e un learning rate decadente (Nadam, da 1e-4 a 6e-6), l'architettura ha potuto mappare i gradienti facciali specifici.

**Risultato Definitivo:** Il modello ha raggiunto una Validation Accuracy del **56.28%**, mantenendo attive le strategie di mitigazione per i bias (Label Smoothing e Balanced Sampling). Questo balzo del +14% rispetto alla baseline dimostra che un'architettura profonda, se sottoposta a un fine-tuning aggressivo, riesce parzialmente a compensare il degrado (upscaling distruttivo, noisy labels) intrinseco del dataset FER-2013.

## 8. Conclusioni

Il progetto ha esplorato l'ecosistema FER-2013 dimostrando l'inefficacia di approcci *brute-force* su dataset rumorosi. Mentre l'estrazione classica HOG+SVM si è rivelata estremamente sensibile su feature geometriche specifiche (es. *Disgust*), le architetture convoluzionali addestrate senza regolarizzazione hanno sofferto di grave overfitting.
L'introduzione metodica di *Data Augmentation*, *Label Smoothing* e *Balanced Sampling* ha permesso alla nostra architettura *from scratch* (Custom CNN V2) di scalare dal 42% al 56.23%, avvicinandosi asintoticamente al limite teorico dettato dal rumore di etichettatura umana (circa 65%) e pareggiando le performance di backbone complessi in Transfer Learning (MobileNetV2 al 56.28%).
L'analisi dei bias ha inoltre certificato vulnerabilità hardware/ambientali, mostrando crolli di accuratezza su immagini sovraesposte.
Possibili sviluppi futuri potrebbero includere tecniche di Ensembling (Soft Voting tra le probabilità di predizione della SVM e delle CNN) o l'utilizzo di reti generative (GAN) per effettuare un sovracampionamento sintetico mirato esclusivamente sulle classi minoritarie, al fine di riequilibrare fisicamente il dataset mitigandone le distorsioni d'origine.