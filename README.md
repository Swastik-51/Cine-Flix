# CineFlix OS — Spatial Movie Recommendation System & Sentiment Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1%2B-red?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TMDB](https://img.shields.io/badge/TMDB%20API-v3-01d277?style=for-the-badge&logo=themoviedatabase&logoColor=white)
![Design](https://img.shields.io/badge/UI-Apple%20VisionOS%20Obsidian-111114?style=for-the-badge&logo=apple&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

<br/>

**Crafted with precision by [Swastik Sengupta](https://github.com/Swastik-51)**

*An intelligent, content-based movie recommendation engine paired with real-time NLP sentiment analysis, packaged inside a calm Apple VisionOS dark obsidian glass interface.*

[Features](#-key-features) • [Architecture](#-system-architecture) • [ML & NLP Pipeline](#-machine-learning--nlp-pipeline) • [Getting Started](#-installation--quickstart) • [Project Structure](#-project-structure) • [Author](#-author--credits)

</div>

---

## 🌌 Overview

**CineFlix OS** is a modern movie recommendation and sentiment analysis web application created by **Swastik Sengupta**. It bridges high-dimensional machine learning similarity models with real-world audience sentiment extraction.

Instead of generic collaborative filtering, CineFlix OS extracts comprehensive film metadata (director, cast, genres, overview, and keyword combinations) and computes high-dimensional **Cosine Similarity** matrices across 6,000+ movies. Concurrently, user reviews are ingested through a trained **TF-IDF Vectorizer** and classified using a **Multinomial Naive Bayes** NLP model to evaluate audience reception in real time.

All of this is presented through an ultra-sleek **Apple VisionOS spatial glass interface** featuring obsidian frosted panels, specular light borders, and subtle crimson highlights.

---

## ✨ Key Features

- **Apple VisionOS Obsidian Glass UI**:
  - Pure pitch black canvas (`#000000`) paired with frosted dark glass cards (`rgba(18, 18, 22, 0.75)`).
  - Refined specular white borders (`rgba(255, 255, 255, 0.08)`) and calm crimson accents (`#e50914`).
  - Mobile-responsive spatial layouts with subtle micro-interactions.

- **Intelligent Multi-Tiered Movie Matcher**:
  - Handles exact matches, sequel subtitles, and typos gracefully.
  - 5-stage resolution pipeline: Exact Match $\rightarrow$ Substring Containment $\rightarrow$ Franchise Inverse Containment (e.g. *Deadpool & Wolverine* resolves seamlessly to *Deadpool*) $\rightarrow$ `difflib` similarity $\rightarrow$ Tokenized word overlap.

- **Content-Based Cosine Similarity Engine**:
  - Pre-computed similarity matrix across bag-of-words combinations.
  - Dynamically calculates and renders the top 10 most similar movies with high-resolution poster artwork.

- **Real-Time NLP Sentiment Analysis Dashboard**:
  - Live audience sentiment percentage gauge.
  - Statistics breakdown: Total Analyzed, Positive, and Critical reviews.
  - Categorized review stream with distinct glass sentiment badges.

- **Interactive Cast Explorer & Spatial Bios**:
  - High-definition cast portrait cards with smooth hover animations.
  - Custom VisionOS spatial modal sheet detailing actor birth dates, places of birth, and comprehensive biographies.
  - Full backdrop dismiss, close button, and Escape key event listeners.

- **Infinite Recommendation Chaining**:
  - Click any of the 10 recommended movie cards to instantly trigger a full re-analysis and query cycle for that movie without reloading the page.

- **Built-in ISP & TMDB Proxy Failover**:
  - Direct DNS routing patch in Python socket layer (`api.themoviedb.org -> 99.84.152.32`) to circumvent Indian ISP DNS spoofing (`49.44.79.236`).
  - Dedicated Flask server-side proxy endpoints (`/api/tmdb/*`) guaranteeing 100% TMDB uptime client-side.

---

## 🏗️ System Architecture

```
                               ┌────────────────────────┐
                               │   Client Web Browser   │
                               │ (Apple VisionOS Glass) │
                               └───────────┬────────────┘
                                           │ AJAX / Fetch
                                           ▼
                       ┌────────────────────────────────────────┐
                       │           Flask Web Server             │
                       │               (main.py)                │
                       └─────┬────────────────────────────┬─────┘
                             │                            │
             ┌───────────────┴──────────────┐             │
             ▼                              ▼             ▼
  ┌──────────────────────┐      ┌───────────────────────────────┐
  │  TMDB Proxy Routes   │      │   Recommendation Engine       │
  │  (/api/tmdb/*)       │      │   - Multi-tier Title Matcher  │
  │  - Patched DNS layer │      │   - Cosine Similarity Matrix  │
  │  - Details, Cast     │      │   - Top 10 Ranked Output      │
  └──────────┬───────────┘      └───────────────┬───────────────┘
             │                                  │
             ▼                                  ▼
  ┌──────────────────────┐      ┌───────────────────────────────┐
  │ TMDB Cloud API (v3)  │      │   NLP Sentiment Pipeline      │
  │  - Live Posters      │      │   - TF-IDF Vectorizer         │
  │  - Actor Biographies │      │   - Multinomial Naive Bayes   │
  │  - Synopses          │      │   - Audience Review Scoring   │
  └──────────────────────┘      └───────────────────────────────┘
```

---

## 🔬 Machine Learning & NLP Pipeline

### 1. Content-Based Recommendation (Cosine Similarity)
The recommendation engine builds a composite feature vector $\vec{V}_i$ for each movie using combined metadata:
$$\text{comb} = \text{genres} + \text{director} + \text{actors} + \text{keywords}$$

Similarity between movie $A$ and movie $B$ is calculated using the Cosine Similarity metric:
$$\text{Similarity}(A, B) = \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

### 2. Audience Sentiment Classification (Multinomial Naive Bayes)
Audience reviews are vectorized through a pre-fitted **TF-IDF (Term Frequency-Inverse Document Frequency)** transformer:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{1 + n}{1 + \text{DF}(t)}\right) + 1$$

The resulting high-dimensional sparse representations are passed through a **Multinomial Naive Bayes** classifier trained to predict sentiment class $c \in \{\text{Good}, \text{Bad}\}$:
$$P(c \mid \vec{x}) \propto P(c) \prod_{j=1}^{d} P(x_j \mid c)^{x_j}$$

---

## 📁 Project Structure

```
Movie-Recommendation-System-with-Sentiment-Analysis/
├── datasets/                   # Raw & processed movie datasets
│   ├── data.csv
│   ├── final_data.csv
│   ├── main_data.csv
│   ├── movie_metadata.csv
│   ├── new_data.csv
│   └── reviews.txt
├── static/                     # Frontend styles, scripts & assets
│   ├── autocomplete.js         # Search autocomplete integration
│   ├── image.jpg               # Fallback poster asset
│   ├── recommend.js            # Async orchestration & modal handling
│   └── style.css               # Apple VisionOS obsidian glass design system
├── templates/                  # Jinja2 HTML templates
│   ├── home.html               # Main discovery landing page
│   └── recommend.html          # Spatial results, cast, reviews & cards
├── .gitignore                  # Git tracking rules
├── main.py                     # Flask server, DNS patch, similarity & proxy
├── main_data.csv               # Cleaned movie metadata for similarity matrix
├── nlp_model.pkl               # Trained Multinomial Naive Bayes sentiment model
├── requirements.txt            # Python dependencies
├── tranform.pkl                # Trained TF-IDF vectorizer
└── README.md                   # Project documentation
```

---

## 🚀 Installation & Quickstart

### Prerequisites
- **Python 3.10+** (Python 3.10, 3.11, 3.12, 3.13, 3.14 supported)
- **Git**
- TMDB API Key (included out of the box in `main.py`)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Swastik-51/Cine-Flix.git
cd Cine-Flix
```

### Step 2: Set Up Virtual Environment
* **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

* **Windows (Command Prompt)**:
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```

* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python main.py
```

### Step 5: Explore
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

Try searching for **Deadpool**, **Avatar**, **Interstellar**, **Inception**, or **The Dark Knight Rises**!

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Backend** | Python 3, Flask | Lightweight WSGI web framework and API proxy |
| **Data Processing** | Pandas, NumPy | High-performance dataframe operations |
| **Machine Learning** | Scikit-Learn | Cosine similarity & Multinomial Naive Bayes |
| **Natural Language** | TF-IDF Vectorizer | Text normalization and sparse matrix encoding |
| **Data Scraping** | BeautifulSoup4, Requests | Multi-tier review retrieval and fallbacks |
| **Frontend** | HTML5, CSS3, Vanilla JS, jQuery | Apple VisionOS Obsidian spatial glass design |
| **External API** | The Movie Database (TMDB) | Movie metadata, posters, cast photos, and bios |

---

## 👨‍💻 Author & Credits

Designed, architected, and crafted with ❤️ by **Swastik Sengupta**.

- **GitHub**: [@Swastik-51](https://github.com/Swastik-51)
- **Portfolio / Projects**: [Swastik-51 Repositories](https://github.com/Swastik-51?tab=repositories)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it for academic or personal projects.
