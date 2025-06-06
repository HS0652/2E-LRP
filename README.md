# 🧠 Deep Reinforcement Learning for Two-Echelon Location-Routing Problem (2E-LRP)

---

## 📦 1. Environment Setup

### 🔧 Requirements

- Python 3.10+
- PyTorch 2.1.0+
- NumPy
- SciPy
- tqdm
- matplotlib
- Gurobi (Optional, for comparison)

---

## 📊 2. Dataset Generation & Format

We use synthetic instances generated in 2D space [10, 90]×[10, 90].

### 📁 Node Feature Format

Each node is a tuple:

```
(node_x, node_y, C_depot, O_depot, C_trans, O_trans, demand, vehicle_load)
```

- For customers: only `demand` is non-zero.
- For transfer stations: has `capacity`, `operating cost`, and `load`.
- For depots: has `capacity`, `operating cost`, and `load`.

Supported scales:

- `small`: 2 depots, 5 transfer stations, 30 customers  
- `medium`: 4 depots, 10 transfer stations, 50 customers  
- `large`: 5 depots, 15 transfer stations, 100 customers

---

## 🏋️‍♀️ 2. Training Procedure

### 🔁 Train the Model

```bash
python train.py   
```

- Checkpoints saved to: `./checkpoints/`
- Training logs saved to: `./logs/`

---

## 🧪 3. Testing & Inference

### 🔍 Run Testing

```bash
python test.py   
```

---

_We hope this work helps accelerate research in supply chain optimization using DRL._
