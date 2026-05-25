import { useState, useCallback, type Dispatch, type SetStateAction } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, AlertTriangle, CheckCircle2, Clock, RotateCcw, Scan, Shield } from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { VideoUpload } from "@/components/VideoUpload";
import { ProcessingStatus } from "@/components/ProcessingStatus";
import { ResultsCard, AnalysisResult } from "@/components/ResultsCard";
import { extractFramesFromVideoSource, ExtractedFrame, TimeRange } from "@/lib/extractFrames";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

const FLASK_API_BASE =
  import.meta.env.VITE_FLASK_API_BASE?.replace(/\/$/, "") || "http://localhost:5000";

interface AnomalyEvent {
  start_sec: number;
  end_sec: number;
  start_hms?: string;
  end_hms?: string;
}

interface ModelPipelineResult {
  durationSec: number;
  events: AnomalyEvent[];
  isAnomalous: boolean;
  maxScore: number;
  meanScore: number;
  nSegments: number;
  scores: number[];
  smoothScores: number[];
  timings?: Record<string, number>;
}

type PipelineStageStatus = "pending" | "running" | "done" | "error";

interface PipelineStage {
  id: string;
  label: string;
  description: string;
  status: PipelineStageStatus;
  duration_sec?: number | null;
  elapsed_sec?: number | null;
}

const INITIAL_PIPELINE_STAGES: PipelineStage[] = [
  {
    id: "upload",
    label: "UPLOAD",
    description: "Receiving video file",
    status: "pending",
    duration_sec: null,
  },
  {
    id: "preprocess",
    label: "PREPROCESS",
    description: "FFmpeg · 224x224 · 30 fps · strip audio",
    status: "pending",
    duration_sec: null,
  },
  {
    id: "features",
    label: "FEATURE EXTRACTION",
    description: "X3D-M · 16-frame windows · 50% overlap · (N, 2048)",
    status: "pending",
    duration_sec: null,
  },
  {
    id: "infer",
    label: "INFERENCE",
    description: "MultiScaleTCN + MIL scorer · median smoothing",
    status: "pending",
    duration_sec: null,
  },
];

const Index = () => {
  const [stage, setStage] = useState<"idle" | "detecting" | "extracting" | "analyzing">("idle");
  const [progress, setProgress] = useState(0);
  const [pipelineStages, setPipelineStages] = useState<PipelineStage[]>(INITIAL_PIPELINE_STAGES);
  const [modelResult, setModelResult] = useState<ModelPipelineResult | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [extractedFrames, setExtractedFrames] = useState<ExtractedFrame[]>([]);
  const { toast } = useToast();

  const processVideo = useCallback(
    async (file: File) => {
      setResult(null);
      setModelResult(null);
      setExtractedFrames([]);
      setPipelineStages(INITIAL_PIPELINE_STAGES);
      setStage("extracting");
      setProgress(0);

      const timings: Record<string, number> = {};

      try {
        // --- Upload to Flask + anomaly detection ---
        setStage("detecting");
        setProgress(5);
        setPipelineStages(markStage(INITIAL_PIPELINE_STAGES, "upload", "running"));

        const tUploadStart = performance.now();
        const formData = new FormData();
        formData.append("video", file);

        const jobResponse = await fetch(`${FLASK_API_BASE}/api/jobs`, {
          method: "POST",
          body: formData,
        });

        timings.video_upload_ms = Math.round(performance.now() - tUploadStart);

        const jobData = await jobResponse.json();
        if (!jobResponse.ok || !jobData.job_id) {
          throw new Error(jobData.message || "Could not start anomaly detection.");
        }

        const anomalyData = await waitForPipelineResult(
          jobData.job_id,
          setPipelineStages,
          setProgress
        );
        const events: AnomalyEvent[] = anomalyData.events ?? [];
        const videoDuration = Number(anomalyData.duration_sec ?? 0);
        const pipelineResult: ModelPipelineResult = {
          durationSec: videoDuration,
          events,
          isAnomalous: anomalyData.is_anomalous === true,
          maxScore: Number(anomalyData.max_score ?? 0),
          meanScore: Number(anomalyData.mean_score ?? 0),
          nSegments: Number(anomalyData.n_segments ?? 0),
          scores: anomalyData.scores ?? [],
          smoothScores: anomalyData.smooth_scores ?? [],
          timings: anomalyData.timings,
        };

        setModelResult(pipelineResult);
        setProgress(100);

        if (events.length === 0) {
          setResult({
            summary: "The anomaly detector did not flag any suspicious segment, so no frames were sent for VLM analysis.",
            badEvent: false,
            reason: "No anomaly windows crossed the configured model threshold.",
            confidence: Math.max(0, 1 - Number(anomalyData.max_score ?? 0)),
            anomalyStart: null,
            anomalyEnd: null,
            eventType: "none",
            duration: videoDuration,
          });
          return;
        }

        // --- Frame Extraction ---
        setStage("extracting");
        setProgress(0);

        const tExtractStart = performance.now();
        const anomalyRanges: TimeRange[] = events.map((event) => ({
          start: event.start_sec,
          end: event.end_sec,
        }));
        const frames = await extractFramesFromVideoSource(
          `${FLASK_API_BASE}${anomalyData.preprocessed_video_url}`,
          16,
          (p) => setProgress(p),
          anomalyRanges
        );
        timings.frame_extract_ms = Math.round(performance.now() - tExtractStart);

        if (frames.length === 0) {
          throw new Error("Could not extract any frames from the video.");
        }

        setExtractedFrames(frames);

        setStage("analyzing");
        setProgress(0);

        const progressInterval = setInterval(() => {
          setProgress((prev) => Math.min(prev + 2, 90));
        }, 500);

        // --- Network Upload + AI Inference (combined in one call) ---
        const tNetworkStart = performance.now();
        const { data, error } = await supabase.functions.invoke("analyze-video", {
          body: {
            frames: frames.map((f) => f.dataUrl),
            timestamps: frames.map((f) => f.timestamp),
            duration: videoDuration,
          },
        });
        const tNetworkEnd = performance.now();
        // We split the total round-trip: estimate ~20% network, ~80% inference
        const totalRoundTrip = tNetworkEnd - tNetworkStart;
        timings.network_upload_ms = Math.round(totalRoundTrip * 0.2);
        timings.ai_inference_ms = Math.round(totalRoundTrip * 0.8);

        clearInterval(progressInterval);
        setProgress(100);

        if (error) throw error;

        // --- Render Results ---
        const tRenderStart = performance.now();

        const analysisResult: AnalysisResult = {
          summary: data.summary || "Unable to analyze.",
          badEvent: data.bad_event === true || data.bad_event === "Yes",
          reason: data.reason || "",
          confidence: typeof data.confidence === "number" ? data.confidence : 0.5,
          anomalyStart: data.anomaly_start ?? events[0]?.start_sec ?? null,
          anomalyEnd: data.anomaly_end ?? events[events.length - 1]?.end_sec ?? null,
          eventType: data.event_type || "none",
          duration: videoDuration,
        };

        setResult(analysisResult);

        // Measure render after state update settles
        requestAnimationFrame(() => {
          timings.render_results_ms = Math.round(performance.now() - tRenderStart);
          timings.total_ms = Object.values(timings).reduce((a, b) => a + b, 0);

          // Silently log to database
          supabase
            .from("analysis_timing_logs")
            .insert({
              video_name: file.name,
              video_duration_s: videoDuration,
              video_upload_ms: timings.video_upload_ms,
              frame_extract_ms: timings.frame_extract_ms,
              network_upload_ms: timings.network_upload_ms,
              ai_inference_ms: timings.ai_inference_ms,
              render_results_ms: timings.render_results_ms,
              total_ms: timings.total_ms,
            })
            .then(({ error: logErr }) => {
              if (logErr) console.warn("Timing log failed:", logErr);
              else console.log("Timing logged:", timings);
            });
        });
      } catch (err: any) {
        console.error("Processing error:", err);
        toast({
          title: "Analysis Failed",
          description: err.message || "Something went wrong. Please try again.",
          variant: "destructive",
        });
      } finally {
        setStage("idle");
        setProgress(0);
      }
    },
    [toast]
  );

  const reset = () => {
    setResult(null);
    setModelResult(null);
    setExtractedFrames([]);
    setPipelineStages(INITIAL_PIPELINE_STAGES);
    setStage("idle");
    setProgress(0);
  };

  const isProcessing = stage !== "idle";

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Ambient glow */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/2 -translate-x-1/2 h-80 w-[600px] rounded-full bg-primary/5 blur-[100px]" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-border/50 glass px-6 py-4">
        <div className="mx-auto flex max-w-4xl items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 border border-primary/20">
            <Shield className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="font-mono text-sm font-bold text-foreground tracking-widest uppercase">
              Sentinel
            </h1>
            <p className="text-[10px] font-mono text-muted-foreground tracking-wider uppercase">
              AI Video Threat Analysis
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2 rounded-full border border-border/50 bg-secondary/50 px-3 py-1">
            <div className="h-2 w-2 rounded-full bg-success animate-pulse" />
            <span className="font-mono text-[10px] text-muted-foreground uppercase">System Online</span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="relative z-10 flex flex-1 flex-col items-center px-6 py-12">
        <div className="w-full max-w-3xl space-y-8">
          {/* Title */}
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-4"
          >
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-4 py-1.5 mb-5">
              <Scan className="h-3.5 w-3.5 text-primary" />
              <span className="font-mono text-xs text-primary">Multimodal AI Analysis</span>
            </div>
            <h2 className="text-4xl font-bold tracking-tight text-gradient mb-3">
              Video Threat Detection
            </h2>
            <p className="text-sm text-muted-foreground max-w-lg mx-auto leading-relaxed">
              Upload a video and the anomaly model narrows it to suspicious moments before multimodal AI explains what happened.
            </p>
          </motion.div>

          {/* Upload */}
          {!modelResult && !result && (
            <VideoUpload onFileSelect={processVideo} isProcessing={isProcessing} />
          )}

          {/* Processing */}
          <AnimatePresence>
            {stage === "detecting" ? (
              <PipelineStageBoard stages={pipelineStages} />
            ) : isProcessing ? (
              <ProcessingStatus stage={stage} progress={progress} />
            ) : null}
          </AnimatePresence>

          {/* Results */}
          <AnimatePresence>
            {(modelResult || result) && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-4"
              >
                {modelResult && <ModelPipelineCard result={modelResult} />}
                {result ? (
                  <ResultsCard result={result} frameThumbnails={extractedFrames} />
                ) : (
                  <div className="rounded-xl border border-border/50 glass p-5">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 border border-primary/20">
                        <Activity className="h-5 w-5 animate-pulse text-primary" />
                      </div>
                      <div>
                        <p className="font-mono text-sm font-medium text-foreground">
                          Semantic analysis running
                        </p>
                        <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                          VLM is analyzing frames from the flagged timestamp windows
                        </p>
                      </div>
                    </div>
                  </div>
                )}
                {!isProcessing && (
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={reset}
                    className="mx-auto flex items-center gap-2 rounded-xl border border-border/50 glass px-5 py-2.5 font-mono text-sm text-foreground transition-all hover:border-primary/30 hover:glow-border"
                  >
                    <RotateCcw className="h-4 w-4" />
                    Analyze Another Video
                  </motion.button>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-border/50 px-6 py-4">
        <p className="text-center font-mono text-[10px] text-muted-foreground tracking-wider uppercase">
          Powered by Gemini AI · Frame-by-frame analysis · Real-time threat detection
        </p>
      </footer>
    </div>
  );
};

function markStage(
  stages: PipelineStage[],
  stageId: string,
  status: PipelineStageStatus
): PipelineStage[] {
  return stages.map((stage) =>
    stage.id === stageId ? { ...stage, status } : stage
  );
}

async function waitForPipelineResult(
  jobId: string,
  setPipelineStages: Dispatch<SetStateAction<PipelineStage[]>>,
  setProgress: Dispatch<SetStateAction<number>>
) {
  while (true) {
    await sleep(700);
    const response = await fetch(`${FLASK_API_BASE}/api/jobs/${jobId}`);
    const job = await response.json();

    if (!response.ok) {
      throw new Error(job.message || "Could not read anomaly pipeline status.");
    }

    if (Array.isArray(job.stages)) {
      setPipelineStages(job.stages);
      setProgress(calculatePipelineProgress(job.stages));
    }

    if (job.status === "complete" && job.result) {
      return job.result;
    }

    if (job.status === "error") {
      throw new Error(job.message || "Anomaly detection failed.");
    }
  }
}

function calculatePipelineProgress(stages: PipelineStage[]): number {
  if (stages.length === 0) return 5;
  const doneWeight = stages.filter((stage) => stage.status === "done").length;
  const runningWeight = stages.some((stage) => stage.status === "running") ? 0.5 : 0;
  return Math.min(95, Math.round(((doneWeight + runningWeight) / stages.length) * 100));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function PipelineStageBoard({ stages }: { stages: PipelineStage[] }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="w-full rounded-xl border border-border/50 glass p-5"
    >
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <h3 className="font-mono text-sm font-semibold uppercase tracking-wider text-foreground">
            Pretrained Model Pipeline
          </h3>
          <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            Upload, FFmpeg preprocessing, X3D-M features, anomaly inference
          </p>
        </div>
        <Activity className="h-5 w-5 animate-pulse text-primary" />
      </div>

      <div className="space-y-2">
        {stages.map((stage, index) => (
          <div
            key={stage.id}
            className="flex items-center gap-3 rounded-lg border border-border/40 bg-secondary/30 px-3 py-3"
          >
            <StageBadge status={stage.status} index={index} />
            <div className="min-w-0 flex-1">
              <div className="font-mono text-xs font-semibold uppercase tracking-wider text-foreground">
                {stage.label}
              </div>
              <div className="truncate font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                {stage.description}
              </div>
            </div>
            <div className="min-w-16 text-right font-mono text-xs text-muted-foreground">
              {stage.status === "running"
                ? `${(stage.elapsed_sec ?? 0).toFixed(1)}s`
                : stage.duration_sec != null
                  ? `${stage.duration_sec.toFixed(1)}s`
                  : "--"}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

function StageBadge({ status, index }: { status: PipelineStageStatus; index: number }) {
  const label =
    status === "done" ? "OK" : status === "error" ? "!" : status === "running" ? "..." : `S${index}`;
  const className =
    status === "done"
      ? "border-success/50 bg-success/10 text-success"
      : status === "error"
        ? "border-destructive/50 bg-destructive/10 text-destructive"
        : status === "running"
          ? "border-warning/50 bg-warning/10 text-warning"
          : "border-border bg-muted/20 text-muted-foreground";

  return (
    <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border font-mono text-[10px] ${className}`}>
      {label}
    </div>
  );
}

function ModelPipelineCard({ result }: { result: ModelPipelineResult }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl border glass p-5 ${
        result.isAnomalous ? "border-destructive/40" : "border-success/40"
      }`}
    >
      <div className="mb-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div
            className={`flex h-11 w-11 items-center justify-center rounded-xl ${
              result.isAnomalous ? "bg-destructive/15" : "bg-success/15"
            }`}
          >
            {result.isAnomalous ? (
              <AlertTriangle className="h-5 w-5 text-destructive" />
            ) : (
              <CheckCircle2 className="h-5 w-5 text-success" />
            )}
          </div>
          <div>
            <h3 className="font-mono text-base font-semibold text-foreground">
              Model Pipeline Results
            </h3>
            <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              Flask preprocessing, features, and anomaly inference complete
            </p>
          </div>
        </div>
        <div className="rounded-full border border-border/50 bg-secondary/50 px-3 py-1 font-mono text-[10px] uppercase text-muted-foreground">
          {result.events.length} window{result.events.length === 1 ? "" : "s"}
        </div>
      </div>

      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric label="Max Score" value={result.maxScore.toFixed(3)} />
        <Metric label="Mean Score" value={result.meanScore.toFixed(3)} />
        <Metric label="Segments" value={String(result.nSegments)} />
        <Metric label="Duration" value={formatTime(result.durationSec)} />
      </div>

      <ScoreTimelineChart result={result} />

      <div className="rounded-lg border border-border/40 bg-muted/20 p-4">
        <div className="mb-3 flex items-center gap-2">
          <Clock className="h-4 w-4 text-primary" />
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Flagged Timestamps
          </span>
        </div>
        {result.events.length > 0 ? (
          <div className="space-y-2">
            {result.events.map((event, index) => (
              <div
                key={`${event.start_sec}-${event.end_sec}-${index}`}
                className="flex items-center justify-between rounded-md border border-destructive/20 bg-destructive/5 px-3 py-2"
              >
                <span className="font-mono text-xs text-foreground">
                  Event {index + 1}
                </span>
                <span className="font-mono text-xs font-semibold text-destructive">
                  {event.start_hms ?? formatTime(event.start_sec)} - {event.end_hms ?? formatTime(event.end_sec)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-foreground/80">
            No anomalous timestamp windows crossed the configured threshold.
          </p>
        )}
      </div>
    </motion.div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border/50 bg-secondary/40 p-3">
      <p className="mb-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p className="font-mono text-sm font-bold text-foreground">{value}</p>
    </div>
  );
}

function ScoreTimelineChart({ result }: { result: ModelPipelineResult }) {
  const chartData = result.smoothScores.map((smooth, index) => ({
    index,
    time: (index * 8) / 30,
    raw: result.scores[index] ?? smooth,
    smooth,
    event: result.events.some(
      (event) => (index * 8) / 30 >= event.start_sec && (index * 8) / 30 <= event.end_sec
    )
      ? 0.5
      : null,
  }));

  if (chartData.length === 0) {
    return null;
  }

  return (
    <div className="mb-4 rounded-lg border border-border/40 bg-muted/20 p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          Anomaly Score Timeline
        </span>
        <span className="font-mono text-[10px] uppercase tracking-widest text-warning">
          Threshold 0.50
        </span>
      </div>
      <div className="h-52 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -24 }}>
            <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="time"
              tickFormatter={(value) => formatTime(Number(value))}
              stroke="hsl(var(--muted-foreground))"
              tick={{ fontSize: 10 }}
            />
            <YAxis
              domain={[0, 1]}
              stroke="hsl(var(--muted-foreground))"
              tick={{ fontSize: 10 }}
            />
            <Tooltip
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                const raw = payload.find((item) => item.dataKey === "raw")?.value;
                const smooth = payload.find((item) => item.dataKey === "smooth")?.value;
                return (
                  <div className="rounded-md border border-border bg-background px-3 py-2 font-mono text-xs shadow-lg">
                    <div className="text-foreground">{formatTime(Number(label))}</div>
                    <div className="text-muted-foreground">raw: {Number(raw ?? 0).toFixed(3)}</div>
                    <div className="text-primary">smooth: {Number(smooth ?? 0).toFixed(3)}</div>
                  </div>
                );
              }}
            />
            <ReferenceLine y={0.5} stroke="hsl(var(--warning))" strokeDasharray="5 5" />
            <Area
              type="monotone"
              dataKey="event"
              fill="hsl(var(--destructive) / 0.18)"
              stroke="transparent"
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="raw"
              dot={false}
              stroke="hsl(var(--destructive) / 0.35)"
              strokeWidth={1}
            />
            <Line
              type="monotone"
              dataKey="smooth"
              dot={false}
              stroke="hsl(var(--primary))"
              strokeWidth={2}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const m = Math.floor(safeSeconds / 60);
  const s = Math.floor(safeSeconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default Index;
