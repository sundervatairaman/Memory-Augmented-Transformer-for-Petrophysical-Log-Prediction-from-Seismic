# Titans: Memory-Augmented Transformer for Well Log Prediction

This project implements a custom memory-augmented Transformer neural network (called `TitansModel`) for predicting petrophysical logs from seismic and geological features. The goal is to enhance log prediction accuracy by incorporating long-term memory and adaptive forgetting mechanisms into the Transformer architecture.

## 🚀 Key Features

- **Memory-Augmented Learning:** Custom `TitansMemoryModule` maintains a dynamic memory of informative samples, adapting during training.
- **Custom Transformer Layer:** Combines self-attention with memory context and adaptive forgetting to learn complex spatial and temporal dependencies.
- **Multitask Regression:** Simultaneous prediction of multiple target logs (AI_Log, RT, RHOB, NPHI).
- **Visualization:** True vs. predicted plots, and well-wise vertical log comparisons across 8 wells.

---

## 🧠 Model Architecture

- **TitansMemoryModule:** A custom memory component with surprise-based memory updates, momentum-enhanced storage, and weight decay forgetting.
- **TitansTransformerLayer:** A Transformer encoder layer integrated with the memory module.
- **TitansModel:** A full Transformer model with multiple `TitansTransformerLayer` instances stacked, followed by a linear regression output layer.

---

## 📊 Input Features & Targets

- **Features:**  
  - `TWT`, `D2`, `Quadr`, `TraceGrad`, `GradMag`, `Freq`, `Zone1`

- **Targets (Predicted Logs):**  
  - `AI_Log`, `RT`, `RHOB`, `NPHI`

---

## 🛠️ Data Preprocessing

- Reads from `merged_data1.xlsx`
- Splits data by well number: wells `<8` for training, `>=8` for testing
- Handles missing values, applies Yeo-Johnson `PowerTransformer` for normalization

---

## 🔁 Training Pipeline

1. Data loading and normalization
2. Custom PyTorch `Dataset` and `DataLoader`
3. TitansModel definition and training (100 epochs)
4. MAE and MSE evaluation on the test set
5. Save model as `gr_model.pt`

---

## 📈 Evaluation & Plots

- Loss vs. Epochs training plot
- Scatter plot of predicted vs. true `AI_Log`
- Vertical log comparison for 8 wells:
  - True and predicted logs plotted over Two-Way Time (TWT)

---

## 💾 Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
