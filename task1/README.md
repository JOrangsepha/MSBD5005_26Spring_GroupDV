# VAST MC1 Q1 - Sailor Shift Influence Network

This project builds an interactive D3 visualization centered on Sailor Shift to explore:

- who influences Sailor Shift (`influenced_by`)
- who Sailor Shift influences (`influences`)
- what has Sailor Shift collaborated with people
- what works/evidence support each relationship

---

## 1) Prerequisites

- Python 3.9+

No third-party Python package installation is required for the current scripts.

---

## 2) Project Setup

1. Clone the repository.
2. Ensure dataset exists at:
   - `data/MC1_graph.json`

## 3) Build the Network JSON

Run the data processing script to generate D3-ready network data:

```bash
python3 scripts/build_influence_network.py
```

Default output:

- `sailor_shift_network.json`

Optional flags:

---

## 4) Run the Visualization

Start a local static server from project root:

```bash
python3 -m http.server 8000
```

Open:

- [http://localhost:8000/index.html](http://localhost:8000/index.html)

---