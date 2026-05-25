# SENTINEL — AI Video Threat Analysis System

> Upload a video. Get an AI-powered safety analysis with timestamped anomaly detection.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [System Flow](#system-flow)
- [Client-Side Processing](#client-side-processing)
- [Backend API](#backend-api)
- [AI Model & Prompt Engineering](#ai-model--prompt-engineering)
- [Response Schema](#response-schema)
- [Scoring Mechanism](#scoring-mechanism)
- [Anomaly Time-Range Detection](#anomaly-time-range-detection)
- [Limits & Constraints](#limits--constraints)
- [Tech Stack](#tech-stack)

---

## Overview

Sentinel is a web application that analyzes uploaded videos for safety threats using multimodal AI. It extracts frames client-side, sends them to a serverless backend function, and returns a structured analysis including what happened, whether it's harmful, a confidence score, and the exact time window of any detected anomaly.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     BROWSER (Client)                    │
│                                                         │
│  ┌──────────┐    ┌────────────────┐    ┌─────────────┐  │
│  │  Video    │───▶│ Frame Extractor │───▶│  Supabase   │  │
│  │  Upload   │    │ (Canvas API)   │    │  SDK Call   │  │
│  │ (100MB)   │    │ 16 frames @    │    │             │  │
│  └──────────┘    │ 512px JPEG     │    └──────┬──────┘  │
│                  └────────────────┘           │         │
└──────────────────────────────────────────────┼─────────┘
                                               │
                                               ▼
                              ┌─────────────────────────┐
                              │   Edge Function          │
                              │   /analyze-video         │
                              │                          │
                              │  • Receives base64       │
                              │    frames + timestamps   │
                              │  • Builds multimodal     │
                              │    prompt                │
                              │  • Calls Gemini 2.5      │
                              │    Flash via AI Gateway  │
                              │  • Returns structured    │
                              │    JSON                  │
                              └───────────┬─────────────┘
                                          │
                                          ▼
                              ┌─────────────────────────┐
                              │   AI Gateway             │
                              │   (Lovable Cloud)        │
                              │                          │
                              │   Model: google/         │
                              │   gemini-2.5-flash       │
                              │                          │
                              │   Multimodal input:      │
                              │   text + images          │
                              └─────────────────────────┘
```

---

## System Flow

1. **Upload** — User drops a video file (MP4, MOV, AVI, up to 100 MB).
2. **Extract** — The browser extracts **16 evenly-spaced frames** using the HTML5 `<canvas>` API. Each frame is:
   - Scaled down to max 512 px on the longest side
   - Encoded as JPEG at 70% quality
   - Tagged with its timestamp in seconds
3. **Send** — The client calls the `analyze-video` edge function with:
   - `frames[]` — array of base64 data URLs
   - `timestamps[]` — array of corresponding timestamps
   - `duration` — total video duration in seconds
4. **Analyze** — The edge function builds a multimodal prompt (text + images) and sends it to **Google Gemini 2.5 Flash** via the AI Gateway.
5. **Respond** — The AI returns structured JSON with summary, threat flag, confidence, event type, and anomaly time range.
6. **Display** — The frontend renders the results with:
   - Threat / All Clear status card
   - Confidence bar
   - Anomaly time window
   - Frame timeline with anomaly frames highlighted in red

---

## Client-Side Processing

**File:** `src/lib/extractFrames.ts`

| Parameter       | Value                          |
|-----------------|--------------------------------|
| Max frames      | 16                             |
| Max resolution  | 512 px (longest side)          |
| Format          | JPEG, 70% quality              |
| Interval        | `duration / 16` seconds apart  |
| Seek timeout    | 5 seconds per frame            |

The extractor creates a hidden `<video>` + `<canvas>`, seeks to each timestamp, draws the frame, and exports it as a data URL.

---

## Backend API

### `POST /functions/v1/analyze-video`

**Edge function** deployed on Lovable Cloud.

#### Request Body

```json
{
  "frames": ["data:image/jpeg;base64,...", "..."],
  "timestamps": [0.0, 1.87, 3.75, "..."],
  "duration": 30.0
}
```

#### Success Response (200)

```json
{
  "summary": "A person is walking across a parking lot when a car reverses into them.",
  "bad_event": true,
  "reason": "Vehicle strikes pedestrian causing potential injury.",
  "confidence": 0.92,
  "anomaly_start": 12.5,
  "anomaly_end": 18.7,
  "event_type": "accident"
}
```

#### Error Responses

| Status | Meaning                        |
|--------|--------------------------------|
| 400    | No frames provided             |
| 402    | AI usage credits exhausted     |
| 429    | Rate limit exceeded            |
| 500    | Internal / AI gateway error    |

#### Authentication

JWT verification is **disabled** (`verify_jwt = false` in `config.toml`). The function is publicly accessible.

---

## AI Model & Prompt Engineering

**Model:** `google/gemini-2.5-flash` (multimodal, low latency)

The system prompt instructs the model to:

1. Analyze all provided frames as a sequence
2. Determine if any harmful event occurred
3. Identify the approximate time range of the anomaly using frame timestamps
4. Categorize the event type

**Threat categories detected:**

| Category   | Examples                              |
|------------|---------------------------------------|
| `violence` | Physical fights, assault              |
| `accident` | Falls, vehicle collisions             |
| `fire`     | Fire, explosions, smoke               |
| `theft`    | Stealing, break-ins, suspicious behavior |
| `medical`  | Collapse, seizures, emergencies       |
| `unsafe`   | Dangerous stunts, hazardous conditions |
| `none`     | No threat detected                    |

---

## Response Schema

```typescript
interface AnalysisResult {
  summary: string;        // 2-4 sentence description of the video
  bad_event: boolean;     // true if a threat was detected
  reason: string;         // one-sentence justification
  confidence: number;     // 0.0 to 1.0
  anomaly_start: number | null;  // seconds into the video
  anomaly_end: number | null;    // seconds into the video
  event_type: string;     // category from the table above
}
```

---

## Scoring Mechanism

The **confidence score** (0.0–1.0) is generated by the AI model and represents how certain it is about its determination:

| Range       | Interpretation            |
|-------------|---------------------------|
| 0.90 – 1.00 | Very high confidence      |
| 0.70 – 0.89 | High confidence           |
| 0.50 – 0.69 | Moderate confidence       |
| 0.30 – 0.49 | Low confidence            |
| 0.00 – 0.29 | Very low / uncertain      |

The score reflects:
- **Visual clarity** — How clear and unambiguous the frames are
- **Temporal consistency** — Whether multiple frames confirm the event
- **Event severity** — How obviously harmful the detected activity is

> Note: The confidence is model-generated, not a calibrated probability. It should be treated as a relative signal, not an absolute measure.

---

## Anomaly Time-Range Detection

Each extracted frame is labeled with its exact timestamp before being sent to the AI. The model uses these labels to report:

- `anomaly_start` — The timestamp of the first frame where the threat begins
- `anomaly_end` — The timestamp of the last frame where the threat is visible

**Resolution:** The time precision is limited by the frame interval (`duration / 16`). For a 30-second video, precision is ~1.9 seconds.

In the UI, frames within the anomaly window are highlighted with a red border and a pulsing indicator.

---

## Limits & Constraints

| Constraint              | Value / Detail                                 |
|-------------------------|------------------------------------------------|
| Max file size           | 100 MB                                         |
| Accepted formats        | Any `video/*` MIME type (MP4, MOV, AVI, etc.)  |
| Frames extracted        | 16 per video                                   |
| Frame resolution        | Max 512 px (longest side)                      |
| Frame format            | JPEG, 70% quality                              |
| AI model                | `google/gemini-2.5-flash`                      |
| Rate limit              | Per-workspace, enforced by AI Gateway (429)    |
| Credit limit            | Per-workspace usage quota (402 when exhausted) |
| Seek timeout            | 5 seconds per frame                            |
| No database             | Results are not persisted; analysis is stateless|
| No authentication       | The endpoint is publicly accessible             |
| Time precision          | ~`duration / 16` seconds (frame interval)      |

---

## Tech Stack

| Layer       | Technology                                   |
|-------------|----------------------------------------------|
| Frontend    | React 18, TypeScript, Vite                   |
| Styling     | Tailwind CSS, shadcn/ui, Framer Motion       |
| Backend     | Lovable Cloud Edge Functions (Deno)          |
| AI          | Google Gemini 2.5 Flash via AI Gateway       |
| Deployment  | Lovable Cloud (automatic)                    |

---

## Project Structure

```
src/
├── components/
│   ├── VideoUpload.tsx       # Drag-and-drop video upload
│   ├── ProcessingStatus.tsx  # Extraction/analysis progress bar
│   └── ResultsCard.tsx       # Analysis results + frame timeline
├── lib/
│   └── extractFrames.ts      # Client-side frame extraction
├── pages/
│   └── Index.tsx             # Main page orchestrator
└── integrations/
    └── supabase/
        └── client.ts         # Auto-generated Supabase client

supabase/
└── functions/
    └── analyze-video/
        └── index.ts          # Edge function: AI analysis endpoint
```
