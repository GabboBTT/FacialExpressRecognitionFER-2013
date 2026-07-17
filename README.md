# 🎭 Facial Expression Recognition (FER-2013)

**Author:** Gabriel Betti (Matricola: 178267)

## 📌 Project Overview
Questo progetto implementa e confronta diverse pipeline di Machine Learning e Deep Learning per la classificazione di espressioni facciali (7 emozioni) sul dataset FER-2013, come richiesto dalle specifiche di esame. L'obiettivo principale è analizzare criticamente le performance, i *failure modes* e il *dataset bias* in condizioni di bassa risoluzione (48x48) e *noisy labels*.

## 🎯 Objectives & Mitigations
*   **Benchmark:** Confronto tra estrattori classici (HOG + SVM) e architetture Deep CNN.
*   **Bias Analysis:** Analisi delle discrepanze di performance causate dalle condizioni di illuminazione (Subgroup bias: Dark vs Bright images).
*   **Mitigation Strategies:** Implementazione di *Label Smoothing*, *Balanced Class Weights* e *Data Augmentation* dinamica per contrastare l'overfitting e lo sbilanciamento delle classi (es. *Fear* e *Disgust*).

## 🚀 Models & Results (Validation Accuracy)
1.  **Custom CNN V2 (Scratch + Augmentation):** `56.23%` - Modello primario ottimizzato.
2.  **MobileNetV2 (Transfer Learning + Fine-Tuning):** `56.28%` - Bonus Sprint per testare i limiti teorici del dataset.
3.  **HOG + LinearSVC:** `41.00%` - Pipeline classica, eccellente nella *Recall* della classe minoritaria *Disgust* (63%).

## 📂 Repository Structure
*   `/src/` - Contiene gli script di training isolati e le pipeline (CNN, SVM, Bias Analysis).
*   `Report_FER2013_178267_Gold.md` - Documentazione scientifica completa, analisi delle metriche e discussione architetturale.
