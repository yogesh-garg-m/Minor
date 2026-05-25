import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
    );

    const { data, error } = await supabase
      .from("analysis_timing_logs")
      .select("*")
      .order("created_at", { ascending: true });

    if (error) throw error;

    const headers = [
      "id", "created_at", "video_name", "video_duration_s",
      "video_upload_ms", "frame_extract_ms", "network_upload_ms",
      "ai_inference_ms", "render_results_ms", "total_ms"
    ];

    let csv = headers.join(",") + "\n";
    for (const row of data || []) {
      csv += headers.map((h) => row[h] ?? "").join(",") + "\n";
    }

    return new Response(csv, {
      headers: {
        ...corsHeaders,
        "Content-Type": "text/csv",
        "Content-Disposition": "attachment; filename=timing_logs.csv",
      },
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: e.message }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
