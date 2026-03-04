# AreoNET — Autonomous UGV Perception Platform

> **Desert terrain semantic segmentation + traversability analysis for autonomous Unmanned Ground Vehicles (UGVs), with a full-stack web platform, 3D digital twin simulation, and an AI-powered visual workflow builder.**

---

## Table of Contents

1. [What Is This?](#what-is-this)
2. [Repository Layout](#repository-layout)
3. [System Architecture](#system-architecture)
4. [Terrain Classes](#terrain-classes)
5. [Backend (Python / FastAPI)](#backend-python--fastapi)
   - [API Endpoints](#api-endpoints)
   - [Advanced Features](#advanced-features)
6. [AreoNet Web App (Next.js)](#areonet-web-app-nextjs)
   - [Pages & Features](#pages--features)
   - [AI Workflow Builder](#ai-workflow-builder)
7. [Frontend Landing Page](#frontend-landing-page)
8. [Inference Pipeline (CLI)](#inference-pipeline-cli)
9. [Model Training](#model-training)
10. [Tech Stack](#tech-stack)
11. [Environment Variables](#environment-variables)
12. [Local Setup](#local-setup)
13. [Deployment](#deployment)
14. [License](#license)

---

## What Is This?

AreoNET solves the problem of **terrain perception for autonomous robots navigating unstructured off-road environments** (desert, rocky, mixed terrain). It does this end-to-end:

1. **Trains** a fine-tuned SegFormer transformer on a synthetic procedurally-generated desert dataset (10 terrain classes).
2. **Serves** segmentation + traversability analysis through a high-performance FastAPI backend with 10+ advanced AI features.
3. **Visualises** everything through a Next.js web dashboard with live inference, 3D rover simulation, training analytics, and an AI-powered drag-and-drop workflow builder.

If you are working on autonomous navigation, path planning, or terrain analysis — this is the full platform to prototype and ship on.

---

## Repository Layout

```
AreoNET/
├── backend/                   # FastAPI segmentation API (Python)
│   ├── main.py                # App entry point + all route definitions
│   ├── segmentation.py        # SegFormer inference + mask colorization
│   ├── dataset.py             # Procedural synthetic terrain dataset generator
│   ├── train.py               # SegFormer fine-tuning script
│   ├── costmap.py             # Traversability cost-map generation
│   ├── model_registry.py      # Hot-swappable multi-model registry
│   ├── uncertainty.py         # MC-Dropout uncertainty quantification
│   ├── adversarial.py         # Adversarial perturbation detection
│   ├── quantum_optimizer.py   # Quantum-inspired hyperparameter optimizer
│   ├── neuromorphic.py        # Leaky Integrate-and-Fire SNN post-processor
│   ├── meta_learner.py        # Test-Time Adaptation (TTA) meta-learner
│   ├── multimodal.py          # CLIP zero-shot queries + SAM-lite prompts
│   ├── self_supervised.py     # Pseudo-label generation + contrastive learning
│   ├── requirements.txt
│   └── Dockerfile
│
├── AreoNet/                   # Main Next.js web application
│   ├── app/                   # Next.js App Router pages
│   │   ├── page.js            # Home / landing page
│   │   ├── dashboard/         # Analytics dashboard (training metrics, architecture)
│   │   ├── simulation/        # 3D digital twin rover simulation
│   │   ├── assistant/         # AI workflow canvas
│   │   ├── auth/              # Authentication pages
│   │   ├── profile/           # User profile
│   │   └── api/               # Next.js API routes (auth, simulation, workflows)
│   ├── components/            # Reusable UI components
│   ├── lib/                   # Utilities (auth, DB, Gemini AI, execution engine)
│   └── public/                # Static assets (3D models, textures, fonts)
│
├── frontend/                  # Separate public-facing landing/docs site (TypeScript)
│   └── app/, components/      # Hero, architecture explainer, segmentation demo
│
├── api_server/                # HuggingFace Spaces API deployment
│   ├── main.py
│   └── models/latest_model_ft.pth   # Fine-tuned checkpoint
│
├── inference_pipeline.py      # Standalone CLI for single-image / batch inference
├── segformer_desert_segmentation.ipynb  # Jupyter training notebook
├── training_metrics.json      # Saved training run results
└── docker-compose.yml         # Spin up backend + frontend together
```

---

## System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                      Browser / Client                     │
│   AreoNet Next.js App (AreoNet/)                          │
│   Landing Page (frontend/)                                │
└────────────────────────────┬──────────────────────────────┘
                             │ HTTP / REST
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI Backend  (backend/)                  │
│  v2.0 — port 8000                                           │
│                                                             │
│  SegFormer-B2 / B4 / B5  ──►  Segmentation Mask (10 cls)   │
│         │                          │                        │
│  MC-Dropout Uncertainty       Cost Map (traversability)     │
│  Adversarial Detection        Neuromorphic SNN Refine       │
│  CLIP Zero-shot Queries       SAM-lite Point Prompts        │
│  Quantum Hyperopt             Meta-learning TTA             │
│  Self-supervised Pseudo-labels                              │
└─────────────────────────────────────────────────────────────┘
                             │
               HuggingFace Transformers + PyTorch
                             │
             nvidia/segformer-b* pretrained weights
             + fine-tuned checkpoint (.pth)
```

The **AreoNet web app** calls the backend API, renders all outputs (segmentation overlay, cost map, uncertainty heatmap), shows the 3D rover simulation in WebGL via Three.js + Rapier physics, and provides the AI workflow canvas powered by Google Gemini 2.5 Pro.

---

## Terrain Classes

The model classifies every pixel into one of **10 terrain classes**:

| ID | Class       | Hex Color | Traversal Cost | Traversable |
|----|-------------|-----------|:--------------:|:-----------:|
| 0  | Rock        | `#8B7355` | 0.90           | ❌          |
| 1  | Bush        | `#4A7023` | 0.75           | ❌          |
| 2  | Log         | `#8B4513` | 0.85           | ❌          |
| 3  | Sand        | `#DEB887` | 0.20           | ✅          |
| 4  | Landscape   | `#C4A862` | 0.15           | ✅          |
| 5  | Sky / Clear | `#87CEEB` | 0.00           | —           |
| 6  | Gravel      | `#A9A9A9` | 0.35           | ✅          |
| 7  | Water       | `#4169E1` | 0.95           | ❌          |
| 8  | Vegetation  | `#228B22` | 0.50           | ✅          |
| 9  | Obstacle    | `#DC143C` | 1.00           | ❌          |

**Traversal cost** (0–1) feeds directly into the cost map consumed by path-planning algorithms. Lower = easier to traverse, 1.0 = impassable.

---

## Backend (Python / FastAPI)

Runs on port **8000**. Built with FastAPI v2.0. Visit `http://localhost:8000/docs` for the live Swagger UI once running.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | System info + active model + feature list |
| `GET`  | `/health` | Liveness check, returns device + model ID |
| `GET`  | `/models` | List all available model variants with size/FPS info |
| `POST` | `/models/switch` | Hot-swap the active model by ID (no restart needed) |
| `POST` | `/segment` | Standard segmentation — returns seg overlay, cost map, class distribution |
| `POST` | `/segment/uncertain` | MC-Dropout segmentation — adds pixel-level uncertainty heatmap + human review flag |
| `POST` | `/segment/adversarial-check` | Detect adversarial perturbations before inference |
| `POST` | `/segment/neuromorphic` | Spiking Neural Network post-processing refinement |
| `POST` | `/segment/meta-tta` | Test-Time Adaptation with class-prior correction |
| `POST` | `/segment/self-supervised` | Generate pseudo-labels + augmented views |
| `POST` | `/segment/clip-query` | CLIP zero-shot natural language terrain query |
| `POST` | `/segment/sam-point` | SAM-lite point-prompted region segmentation |
| `POST` | `/optimize/quantum` | Run Simulated Quantum Annealing hyperparameter search |
| `GET`  | `/dataset/generate` | Generate a synthetic terrain image on-the-fly |

All `/segment/*` endpoints accept a multipart file upload (`image/png` or `image/jpeg`) and return base64-encoded result images + JSON metadata.

### Advanced Features

| Feature | Module | What it does |
|---------|--------|--------------|
| **Multi-model hot-swap** | `model_registry.py` | Switch between SegFormer-B2, B4, B5 at runtime without restarting the server. The registry lazy-loads models and tracks the active one. |
| **MC-Dropout uncertainty** | `uncertainty.py` | Runs N forward passes (default 12) with dropout active, computes epistemic uncertainty per pixel, flags images that need human review when confidence is low or OOD score is high. |
| **Adversarial detection** | `adversarial.py` | Statistical + optional gradient-based check to detect perturbed inputs. Blocks `adversarial`-rated images, passes `suspicious` with a warning. |
| **Quantum-inspired optimizer** | `quantum_optimizer.py` | Simulated Quantum Annealing using path-integral Monte Carlo with parallel tempering replicas. Finds optimal learning rate, batch size, weight decay, dropout, warmup steps, and label smoothing. |
| **Neuromorphic SNN** | `neuromorphic.py` | Leaky Integrate-and-Fire spiking neural network post-processor. Converts logit features into spike trains, integrates over time steps for noise-robust, energy-efficient refinement. |
| **Meta-learning TTA** | `meta_learner.py` | Test-Time Adaptation with class-prior correction. Estimates the prior from the current batch and adjusts logits to handle domain shift without retraining. |
| **Self-supervised** | `self_supervised.py` | Pseudo-label generation from high-confidence predictions, temporal consistency scoring between frames, and contrastive similarity for unlabelled data. |
| **CLIP zero-shot** | `multimodal.py` | Send a natural language query ("show me all sandy patches") and get a heatmap highlighting matching terrain regions. |
| **SAM-lite** | `multimodal.py` | Click a point on the image and get the segmented region at that location, inspired by Segment Anything Model. |

---

## AreoNet Web App (Next.js)

Located in `AreoNet/`. This is the main application your teammates interact with day-to-day. Built with Next.js 16 App Router.

### Pages & Features

| Route | Description |
|-------|-------------|
| `/` | Home page: interactive 3D UGV model (Three.js GLB viewer), feature cards, animated `LaserFlow` and `LiquidEther` backgrounds, dark/light theme toggle |
| `/dashboard` | Analytics dashboard: training loss & mIoU curves (Recharts), per-class IoU bar charts, radar chart, SegFormer architecture flow diagram, live segmentation visualizer that calls the backend in real time |
| `/simulation` | 3D digital twin: WebGL simulation of the rover navigating procedurally-generated terrain using Three.js + `@react-three/rapier` physics engine |
| `/assistant` | AI workflow canvas — see [AI Workflow Builder](#ai-workflow-builder) below |
| `/auth/signin` | Sign in with email/password (NextAuth.js + MongoDB) |
| `/auth/signup` | Register with password strength validation |
| `/profile` | User profile management |

### AI Workflow Builder

The `/assistant` route is a **ReactFlow-based visual workflow canvas** for building multi-step AI pipelines without writing code.

**How it works:**

1. Drag **nodes** onto the canvas — each node represents an AI task (e.g. "Segment Image", "Analyse Traversability", "Generate Navigation Report").
2. Connect nodes with **edges** — edges carry data from one node's output to the next node's input. You can add **transfer logic** labels to edges describing what data flows.
3. Fill in a **campaign brief** and optional **KYC / business profile** for the context sidebar.
4. Hit **Run** — the execution engine (`lib/execution-engine.ts`) traverses the graph in topological order, compiles a structured prompt for each node (incorporating upstream outputs + campaign context + KYC), and sends it to **Google Gemini 2.5 Pro**.
5. Results stream back into each node and can be reviewed, edited, and re-run.

**Key files:**
- `lib/execution-engine.ts` — graph traversal, context compilation, prompt building
- `lib/gemini.ts` — Gemini 2.5 Pro / Flash / Image model wrappers with retry logic
- `lib/store.ts` — Zustand store for workflow state
- `app/api/workflows/` — API routes for saving/loading workflows
- `components/VisualWorkflow.jsx` — ReactFlow canvas component
- `components/NodeEditModal.jsx` — node configuration modal

---

## Frontend Landing Page

Located in `frontend/`. A separate TypeScript Next.js app serving as the public-facing documentation / marketing site. **This is what strangers see first.**

| Component | Purpose |
|-----------|---------|
| `Hero.tsx` | Animated hero section with tagline and CTA |
| `Architecture.tsx` | Visual architecture explainer diagram |
| `SegmentationDemo.tsx` | Interactive live segmentation demo widget |
| `RoverViewer.tsx` / `GameRoverViewer.tsx` | 3D rover model viewers |
| `TerrainClasses.tsx` | Terrain class colour reference |
| `StatsBar.tsx` | Model performance metrics bar |
| `APIReference.tsx` | Quick API endpoint reference |
| `Navigation.tsx` | Top navigation bar |
| `Footer.tsx` | Site footer |

---

## Inference Pipeline (CLI)

For running inference locally without standing up the web server:

```bash
# Activate the virtual environment (Windows)
.venv\Scripts\Activate.ps1

# Single image
python inference_pipeline.py --image path/to/image.png

# Batch on a folder
python inference_pipeline.py --folder path/to/images/

# Use the fine-tuned checkpoint
python inference_pipeline.py --image img.png --checkpoint api_server/models/latest_model_ft.pth
```

**What it outputs:** a colourised segmentation overlay saved next to the input, plus a per-class legend in the matplotlib figure.

The backbone is `nvidia/mit-b4` with 10 terrain classes:
`Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, Sky`

---

## Model Training

### Step 1 — Generate synthetic dataset

```bash
cd backend
python dataset.py --out ./terrain_dataset --n 2000 --size 512
```

This generates procedural 512×512 top-down terrain renders (desert, rocky, mixed) with pixel-level ground truth masks using sine-noise fields and pseudo-random spatial variation. Each sample is an RGB image + a single-channel label mask.

### Step 2 — Fine-tune SegFormer

```bash
python train.py \
  --data   ./terrain_dataset \
  --model  nvidia/segformer-b2-finetuned-ade-512-512 \
  --out    ./checkpoints/dunenet \
  --epochs 30 \
  --bs     8 \
  --lr     6e-5
```

The training script uses:
- **AdamW** optimiser
- **OneCycleLR / cosine warmup** scheduler
- Checkpoint saved when val mIoU improves
- Outputs `metrics.json` alongside the checkpoint

Alternatively, run the full training interactively in the Jupyter notebook at the repo root:

```
segformer_desert_segmentation.ipynb
```

Training results from the latest run are in `training_metrics.json` and are automatically visualised on the `/dashboard` page (no extra steps needed).

---

## Tech Stack

### Backend
| Layer | Technology |
|-------|-----------|
| API framework | FastAPI 0.109+ with Uvicorn |
| ML framework | PyTorch 2.2+ |
| Model | HuggingFace Transformers — SegFormer (nvidia/segformer-b*) |
| Image processing | Pillow, OpenCV, NumPy |
| Containerisation | Docker |
| Deployment | Railway / Render / any Docker host |

### AreoNet Web App
| Layer | Technology |
|-------|-----------|
| Framework | Next.js 16 (App Router, React 19) |
| Language | JavaScript (pages) + TypeScript (lib, types) |
| 3D rendering | Three.js, @react-three/fiber, @react-three/drei, Rapier physics |
| Workflow canvas | ReactFlow 11 |
| Charts | Recharts |
| UI components | Radix UI primitives + Tailwind CSS + shadcn/ui |
| Animations | Framer Motion, GSAP |
| AI | Google Gemini 2.5 Pro (`@google/generative-ai`) |
| Auth | NextAuth.js v4 |
| Database | MongoDB via Mongoose |
| Deployment | Vercel |

---

## Environment Variables

Create `AreoNet/.env.local`:

```env
# Google Gemini AI (required for the workflow builder / assistant)
GEMINI_API_KEY=your_gemini_api_key_here

# NextAuth (required for authentication)
NEXTAUTH_SECRET=some_random_string_at_least_32_chars
NEXTAUTH_URL=http://localhost:3000

# MongoDB (required for user accounts + saved workflows)
MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/areonet

# Backend API URL (where the FastAPI server is running)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For the **backend**, optionally restrict CORS:

```env
# Comma-separated list of allowed frontend origins
ALLOWED_ORIGINS=https://your-frontend.vercel.app,https://your-other-domain.com
```

---

## Local Setup

### Prerequisites
- Node.js v18+
- Python 3.9+
- Docker (optional, for the backend)

### 1. Clone the repo

```bash
git clone <repo-url>
cd AreoNET
```

### 2. Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is now at `http://localhost:8000`.  
Interactive Swagger docs: `http://localhost:8000/docs`

### 3. Start the AreoNet web app

```bash
cd AreoNet
npm install
cp .env.example .env.local   # then fill in your keys
npm run dev
```

App runs at `http://localhost:3000`.

### 4. (Optional) Start the frontend landing page

```bash
cd frontend
npm install
npm run dev   # http://localhost:3001
```

### 5. (Optional) Docker Compose — backend + frontend together

```bash
# from repo root
docker-compose up --build
```

---

## Deployment

### Web App → Vercel

1. Push to GitHub.
2. Import the repo at [vercel.com/new](https://vercel.com/new).
3. Set **Root Directory** to `AreoNet`.
4. Add environment variables (same as `.env.local`).
5. Click **Deploy** — Vercel auto-detects Next.js.

See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) for the full step-by-step guide.

### Backend → Railway / Render

The backend has a `Dockerfile` ready to go:

```bash
# Test locally
docker build -t areonet-backend ./backend
docker run -p 8000:8000 areonet-backend
```

**Railway:** Connect the GitHub repo, set root directory to `backend/`, and Railway will auto-build from the Dockerfile.  
**Render:** Create a new Web Service, point to `backend/`, select Docker environment.

After deploying, update `NEXT_PUBLIC_API_URL` in your Vercel project settings to point to the live backend URL.

---

## License

MIT License
