import { motion } from "framer-motion";
import { Loader2, Cpu, Eye } from "lucide-react";

interface ProcessingStatusProps {
  stage: "detecting" | "extracting" | "analyzing" | "idle";
  progress: number;
}

const stageConfig = {
  idle: { label: "Ready", icon: Cpu },
  detecting: { label: "Detecting anomalies...", icon: Cpu },
  extracting: { label: "Extracting flagged frames...", icon: Eye },
  analyzing: { label: "AI analyzing flagged moments...", icon: Cpu },
};

export function ProcessingStatus({ stage, progress }: ProcessingStatusProps) {
  if (stage === "idle") return null;
  const config = stageConfig[stage];
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="w-full rounded-xl border border-border/50 glass p-6"
    >
      <div className="flex items-center gap-3 mb-4">
        <div className="relative">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 border border-primary/20">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          <Loader2 className="absolute -top-1 -right-1 h-4 w-4 animate-spin text-primary" />
        </div>
        <div>
          <span className="font-mono text-sm font-medium text-foreground">
            {config.label}
          </span>
          <p className="font-mono text-[10px] text-muted-foreground">
            {stage === "detecting"
              ? "Running the pretrained anomaly pipeline"
              : stage === "extracting"
                ? "Sampling from standardized anomaly windows"
                : "Running multimodal threat detection"}
          </p>
        </div>
      </div>

      <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
        <motion.div
          className="h-full rounded-full bg-primary"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          style={{ boxShadow: "0 0 12px hsl(185 100% 50% / 0.5)" }}
        />
      </div>
      <p className="mt-2 text-right font-mono text-xs text-muted-foreground">
        {Math.round(progress)}%
      </p>
    </motion.div>
  );
}
