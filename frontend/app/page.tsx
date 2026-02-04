"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import {
  VideoIcon,
  Download,
  Loader2,
  Sparkles,
  Wand2,
  Layers,
  Orbit,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function Home() {
  const [topic, setTopic] = useState("");
  const [background, setBackground] = useState("");
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/generate-video`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ topic, background }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate video");
      }

      const data = await response.json();
      setVideoUrl(data.videoUrl);
      toast({
        title: "Success!",
        description: "Your video has been generated successfully.",
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to generate video. Please try again.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/2 h-96 w-96 -translate-x-1/2 rounded-full bg-cyan-500/20 blur-[140px]" />
        <div className="absolute bottom-0 right-0 h-80 w-80 rounded-full bg-fuchsia-500/20 blur-[140px]" />
        <div className="absolute left-10 top-24 h-24 w-24 rounded-full border border-cyan-400/30" />
        <div className="absolute bottom-20 left-1/4 h-36 w-36 rounded-full border border-fuchsia-400/20" />
      </div>

      <div className="relative container max-w-6xl mx-auto px-4 py-16 lg:py-20">
        <div className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs uppercase tracking-[0.3em] text-cyan-200">
              <Sparkles className="h-4 w-4" />
              Neural video studio
            </div>
            <div className="space-y-5">
              <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
                VidCraft <span className="text-cyan-300">Synthesizes</span> cinematic stories from a single prompt.
              </h1>
              <p className="text-lg text-slate-300">
                Generate immersive AI videos with dynamic backgrounds, crisp narration, and styled subtitles.
                Go from idea to publish-ready clip in minutes.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              {[
                { icon: Wand2, title: "Prompt-to-video", body: "Turn a short prompt into a moving visual scene." },
                { icon: Layers, title: "Layered audio", body: "AI narration aligned with subtitle timing." },
                { icon: Orbit, title: "Motion engine", body: "Looped cinematic backgrounds and smooth playback." },
              ].map((item) => (
                <Card
                  key={item.title}
                  className="border-white/10 bg-white/5 p-4 text-slate-200 backdrop-blur"
                >
                  <item.icon className="mb-3 h-5 w-5 text-cyan-300" />
                  <p className="text-sm font-semibold">{item.title}</p>
                  <p className="mt-2 text-xs text-slate-400">{item.body}</p>
                </Card>
              ))}
            </div>
          </div>

          <Card className="border-white/10 bg-white/5 p-6 shadow-2xl shadow-cyan-500/10 backdrop-blur">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <label htmlFor="topic" className="text-sm font-medium text-slate-200">
                  Video Topic
                </label>
                <Input
                  id="topic"
                  placeholder="e.g. The rise of quantum computing"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  required
                  disabled={loading}
                  className="border-white/10 bg-slate-900/60 text-white placeholder:text-slate-500"
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="background" className="text-sm font-medium text-slate-200">
                  Background Description
                </label>
                <Textarea
                  id="background"
                  placeholder="Describe the cinematic scene you want the AI to render..."
                  value={background}
                  onChange={(e) => setBackground(e.target.value)}
                  required
                  disabled={loading}
                  className="min-h-[120px] border-white/10 bg-slate-900/60 text-white placeholder:text-slate-500"
                />
              </div>

              <Button
                type="submit"
                className="w-full bg-cyan-500 text-slate-950 hover:bg-cyan-400"
                disabled={loading}
                size="lg"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating Video...
                  </>
                ) : (
                  <>
                    <VideoIcon className="mr-2 h-4 w-4" />
                    Generate Video
                  </>
                )}
              </Button>
            </form>
          </Card>
        </div>

        <div className="mt-12 grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          {videoUrl ? (
            <Card className="border-white/10 bg-white/5 p-6 backdrop-blur">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold">Generated Video</h2>
                  <p className="text-sm text-slate-400">Preview and export your AI-crafted clip.</p>
                </div>
                <span className="rounded-full border border-cyan-400/40 px-3 py-1 text-xs text-cyan-200">
                  Ready
                </span>
              </div>
              <div className="mt-4 aspect-video rounded-2xl border border-white/10 bg-slate-900/60 overflow-hidden">
                <video src={videoUrl} controls className="h-full w-full object-contain" />
              </div>
              <Button
                variant="outline"
                className="mt-4 w-full border-white/20 text-white hover:bg-white/10"
                onClick={() => window.open(videoUrl, "_blank")}
              >
                <Download className="mr-2 h-4 w-4" />
                Download Video
              </Button>
            </Card>
          ) : (
            <Card className="border-dashed border-white/20 bg-white/5 p-6 text-slate-400 backdrop-blur">
              <h2 className="text-xl font-semibold text-white">Awaiting your first render</h2>
              <p className="mt-2 text-sm">
                Fill in the prompts and launch the generation pipeline to preview your video here.
              </p>
            </Card>
          )}

          <Card className="border-white/10 bg-gradient-to-br from-white/5 to-cyan-500/10 p-6 text-slate-200 backdrop-blur">
            <h3 className="text-lg font-semibold">Generation pipeline</h3>
            <ul className="mt-4 space-y-3 text-sm text-slate-300">
              <li className="flex items-center gap-3">
                <span className="h-2 w-2 rounded-full bg-cyan-400" />
                Narrative script + voiceover aligned to topic.
              </li>
              <li className="flex items-center gap-3">
                <span className="h-2 w-2 rounded-full bg-cyan-400" />
                Cinematic background video rendered from your prompt.
              </li>
              <li className="flex items-center gap-3">
                <span className="h-2 w-2 rounded-full bg-cyan-400" />
                Subtitle burn-in with optimal pacing.
              </li>
              <li className="flex items-center gap-3">
                <span className="h-2 w-2 rounded-full bg-cyan-400" />
                Final MP4 export ready for social sharing.
              </li>
            </ul>
          </Card>
        </div>
      </div>
    </main>
  );
}
