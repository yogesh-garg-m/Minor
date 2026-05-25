export interface ExtractedFrame {
  dataUrl: string;
  timestamp: number;
}

export interface TimeRange {
  start: number;
  end: number;
}

export async function extractFramesFromVideo(
  file: File,
  maxFrames = 16,
  onProgress?: (progress: number) => void
): Promise<ExtractedFrame[]> {
  return extractFramesFromVideoSource(file, maxFrames, onProgress);
}

export async function extractFramesFromVideoSource(
  source: File | string,
  maxFrames = 16,
  onProgress?: (progress: number) => void,
  ranges?: TimeRange[]
): Promise<ExtractedFrame[]> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    video.crossOrigin = "anonymous";

    const isObjectUrl = source instanceof File;
    const url = isObjectUrl ? URL.createObjectURL(source) : source;
    video.src = url;

    video.onloadedmetadata = async () => {
      const duration = video.duration;
      if (duration === 0 || !isFinite(duration)) {
        if (isObjectUrl) URL.revokeObjectURL(url);
        reject(new Error("Invalid video duration"));
        return;
      }

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d")!;

      // Scale down for efficiency
      const scale = Math.min(1, 512 / Math.max(video.videoWidth, video.videoHeight));
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);

      const sampleTimes = buildSampleTimes(duration, maxFrames, ranges);
      const frames: ExtractedFrame[] = [];

      for (let i = 0; i < sampleTimes.length; i++) {
        const time = sampleTimes[i];
        try {
          await seekTo(video, time);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          frames.push({
            dataUrl: canvas.toDataURL("image/jpeg", 0.7),
            timestamp: time,
          });
          onProgress?.(((i + 1) / sampleTimes.length) * 100);
        } catch {
          // Skip frame on error
        }
      }

      if (isObjectUrl) URL.revokeObjectURL(url);
      resolve(frames);
    };

    video.onerror = () => {
      if (isObjectUrl) URL.revokeObjectURL(url);
      reject(new Error("Failed to load video. Unsupported format or corrupted file."));
    };
  });
}

function buildSampleTimes(
  duration: number,
  maxFrames: number,
  ranges?: TimeRange[]
): number[] {
  const cleanRanges = (ranges ?? [])
    .map((range) => ({
      start: clamp(range.start, 0, duration),
      end: clamp(range.end, 0, duration),
    }))
    .filter((range) => range.end > range.start);

  if (cleanRanges.length === 0) {
    return Array.from({ length: maxFrames }, (_, i) =>
      clamp((duration / maxFrames) * i, 0, duration)
    );
  }

  const totalRangeDuration = cleanRanges.reduce(
    (sum, range) => sum + range.end - range.start,
    0
  );

  return cleanRanges.flatMap((range) => {
    const share = (range.end - range.start) / totalRangeDuration;
    const count = Math.max(1, Math.round(maxFrames * share));
    const interval = (range.end - range.start) / count;

    return Array.from({ length: count }, (_, i) =>
      clamp(range.start + interval * i, range.start, range.end)
    );
  }).slice(0, maxFrames);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function seekTo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Seek timeout")), 5000);
    video.onseeked = () => {
      clearTimeout(timeout);
      resolve();
    };
    video.currentTime = time;
  });
}
