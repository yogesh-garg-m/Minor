import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, AlertTriangle, Clock, Tag, Zap } from "lucide-react";

export interface AnalysisResult {
  summary: string;
  badEvent: boolean;
  reason: string;
  confidence: number;
  anomalyStart?: number | null;
  anomalyEnd?: number | null;
  eventType?: string;
  duration?: number;
}

interface ResultsCardProps {
  result: AnalysisResult;
  frameThumbnails?: { dataUrl: string; timestamp: number }[];
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function ResultsCard({ result, frameThumbnails }: ResultsCardProps) {
  const isBad = result.badEvent;
  const confidencePct = Math.round(result.confidence * 100);
  const hasTimeRange = result.anomalyStart != null && result.anomalyEnd != null;
  const duration = result.duration || 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full space-y-4"
    >
      {/* Main Result Card */}
      <div
        className={`rounded-xl border p-6 glass ${
          isBad
            ? "border-destructive/40"
            : "border-success/40"
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            {isBad ? (
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-destructive/15 animate-pulse-glow" style={{ "--tw-shadow-color": "hsl(var(--destructive))" } as any}>
                <ShieldAlert className="h-6 w-6 text-destructive" />
              </div>
            ) : (
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-success/15">
                <ShieldCheck className="h-6 w-6 text-success" />
              </div>
            )}
            <div>
              <h3 className="font-mono text-lg font-semibold text-foreground">
                {isBad ? "Threat Detected" : "All Clear"}
              </h3>
              <p className="text-xs text-muted-foreground font-mono">
                {confidencePct}% confidence
              </p>
            </div>
          </div>

          {/* Event type badge */}
          {result.eventType && result.eventType !== "none" && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className="flex items-center gap-1.5 rounded-full border border-destructive/30 bg-destructive/10 px-3 py-1"
            >
              <Tag className="h-3 w-3 text-destructive" />
              <span className="font-mono text-xs font-medium uppercase text-destructive">
                {result.eventType}
              </span>
            </motion.div>
          )}
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 gap-3 mb-5">
          <div className="rounded-lg bg-secondary/50 border border-border/50 p-3">
            <div className="flex items-center gap-2 mb-1">
              <Zap className="h-3.5 w-3.5 text-primary" />
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                Confidence
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${confidencePct}%` }}
                  transition={{ duration: 0.8, ease: "easeOut", delay: 0.3 }}
                  className={`h-full rounded-full ${isBad ? "bg-destructive" : "bg-success"}`}
                />
              </div>
              <span className="font-mono text-sm font-bold text-foreground">
                {confidencePct}%
              </span>
            </div>
          </div>

          <div className="rounded-lg bg-secondary/50 border border-border/50 p-3">
            <div className="flex items-center gap-2 mb-1">
              <Clock className="h-3.5 w-3.5 text-primary" />
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                {hasTimeRange ? "Anomaly Window" : "Time Range"}
              </span>
            </div>
            <span className="font-mono text-sm font-bold text-foreground">
              {hasTimeRange
                ? `${formatTime(result.anomalyStart!)} – ${formatTime(result.anomalyEnd!)}`
                : "N/A"}
            </span>
          </div>
        </div>

        {/* Summary & Reason */}
        <div className="space-y-3">
          <div className="rounded-lg bg-muted/30 border border-border/30 p-4">
            <label className="mb-1.5 block font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              Summary
            </label>
            <p className="text-sm leading-relaxed text-foreground/90">{result.summary}</p>
          </div>

          <div className="rounded-lg bg-muted/30 border border-border/30 p-4">
            <label className="mb-1.5 block font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              Reasoning
            </label>
            <p className="text-sm text-foreground/90">{result.reason}</p>
          </div>
        </div>

        {isBad && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-5 flex items-center gap-2 rounded-lg bg-warning/10 border border-warning/20 px-4 py-3 text-sm text-warning"
          >
            <AlertTriangle className="h-4 w-4 flex-shrink-0" />
            This video may contain harmful or dangerous content. Review recommended.
          </motion.div>
        )}
      </div>

      {/* Timeline with frame thumbnails */}
      {frameThumbnails && frameThumbnails.length > 0 && duration > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="rounded-xl border border-border glass p-5"
        >
          <label className="mb-3 block font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Frame Timeline
          </label>

          {/* Timeline bar */}
          <div className="relative mb-3">
            <div className="h-2 rounded-full bg-secondary overflow-hidden">
              {hasTimeRange && (
                <div
                  className="absolute h-2 rounded-full bg-destructive/40"
                  style={{
                    left: `${(result.anomalyStart! / duration) * 100}%`,
                    width: `${((result.anomalyEnd! - result.anomalyStart!) / duration) * 100}%`,
                  }}
                />
              )}
            </div>
            {/* Time markers */}
            <div className="flex justify-between mt-1">
              <span className="font-mono text-[10px] text-muted-foreground">0:00</span>
              <span className="font-mono text-[10px] text-muted-foreground">{formatTime(duration)}</span>
            </div>
          </div>

          {/* Frame thumbnails */}
          <div className="flex gap-1.5 overflow-x-auto pb-2 scrollbar-thin">
            {frameThumbnails.map((frame, i) => {
              const isInAnomaly =
                hasTimeRange &&
                frame.timestamp >= result.anomalyStart! &&
                frame.timestamp <= result.anomalyEnd!;
              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.5 + i * 0.05 }}
                  className={`relative flex-shrink-0 rounded-md overflow-hidden border-2 transition-all ${
                    isInAnomaly
                      ? "border-destructive/70 shadow-[0_0_10px_hsl(0_72%_55%/0.3)]"
                      : "border-border/30"
                  }`}
                >
                  <img
                    src={frame.dataUrl}
                    alt={`Frame at ${formatTime(frame.timestamp)}`}
                    className="h-16 w-24 object-cover"
                  />
                  <div className="absolute bottom-0 inset-x-0 bg-background/80 backdrop-blur-sm px-1 py-0.5">
                    <span className="font-mono text-[9px] text-foreground/80">
                      {formatTime(frame.timestamp)}
                    </span>
                  </div>
                  {isInAnomaly && (
                    <div className="absolute top-0.5 right-0.5">
                      <div className="h-2 w-2 rounded-full bg-destructive animate-pulse" />
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
