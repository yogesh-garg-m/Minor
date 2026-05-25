
CREATE TABLE public.analysis_timing_logs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  video_name TEXT,
  video_duration_s NUMERIC,
  video_upload_ms NUMERIC,
  frame_extract_ms NUMERIC,
  network_upload_ms NUMERIC,
  ai_inference_ms NUMERIC,
  render_results_ms NUMERIC,
  total_ms NUMERIC
);

-- Public insert, public select (no auth needed for this logging table)
ALTER TABLE public.analysis_timing_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can insert timing logs"
  ON public.analysis_timing_logs FOR INSERT
  WITH CHECK (true);

CREATE POLICY "Anyone can read timing logs"
  ON public.analysis_timing_logs FOR SELECT
  USING (true);
