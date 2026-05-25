import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Film, X } from "lucide-react";

interface VideoUploadProps {
  onFileSelect: (file: File) => void;
  isProcessing: boolean;
}

const MAX_SIZE_MB = 100;

export function VideoUpload({ onFileSelect, isProcessing }: VideoUploadProps) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      setError(null);
      if (!file.type.startsWith("video/")) {
        setError("Please upload a video file.");
        return;
      }
      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        setError(`File too large. Maximum size is ${MAX_SIZE_MB}MB.`);
        return;
      }
      setSelectedFile(file);
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const clear = () => {
    setSelectedFile(null);
    setError(null);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full"
    >
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={`relative overflow-hidden rounded-lg border-2 border-dashed transition-all duration-300 ${
          dragOver
            ? "border-primary glow-border-strong bg-primary/5"
            : "border-border hover:border-primary/50 hover:glow-border"
        } ${isProcessing ? "pointer-events-none opacity-60" : "cursor-pointer"}`}
      >
        <div className="relative z-10 flex flex-col items-center justify-center gap-4 py-16 px-6">
          <AnimatePresence mode="wait">
            {selectedFile ? (
              <motion.div
                key="selected"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex flex-col items-center gap-3"
              >
                <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <Film className="h-8 w-8" />
                </div>
                <div className="text-center">
                  <p className="font-mono text-sm text-primary">{selectedFile.name}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                  </p>
                </div>
                {!isProcessing && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      clear();
                    }}
                    className="mt-2 flex items-center gap-1 rounded-md px-3 py-1 text-xs text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                  >
                    <X className="h-3 w-3" /> Remove
                  </button>
                )}
              </motion.div>
            ) : (
              <motion.div
                key="upload"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex flex-col items-center gap-3"
              >
                <div className="flex h-16 w-16 items-center justify-center rounded-xl border border-border bg-secondary/50">
                  <Upload className="h-8 w-8 text-muted-foreground" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-foreground">
                    Drop your video here
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    or click to browse · MP4, MOV, AVI · up to {MAX_SIZE_MB}MB
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <input
          type="file"
          accept="video/*"
          className="absolute inset-0 z-20 cursor-pointer opacity-0"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
          disabled={isProcessing}
        />
      </div>

      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-3 text-center text-sm text-destructive"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
