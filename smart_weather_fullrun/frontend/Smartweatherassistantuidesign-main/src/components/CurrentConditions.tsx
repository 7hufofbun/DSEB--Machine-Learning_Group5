import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Droplets, Wind, CloudSun, Thermometer, Sun, Sunrise, Flame } from "lucide-react";
import { apiGet } from "../lib/api";
import { formatTemperature } from "../lib/format";

interface NowData {
  temperature?: number;
  humidity?: number;
  conditions?: string;
  wind?: string;
  feels_like?: number;
  location?: string;
  updated_at?: string;
}

function MetricRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number | undefined; }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-border last:border-b-0">
      <div className="flex items-center gap-3">
        <div className="text-primary">{icon}</div>
        <span className="text-muted-foreground">{label}</span>
      </div>
      <span className="font-mono">{value ?? "-"}</span>
    </div>
  );
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const classifyHeatStress = (temperature?: number, humidity?: number): string => {
  if (temperature === undefined || temperature === null || humidity === undefined || humidity === null) {
    return "—";
  }

  const heatExcess = Math.max(0, temperature - 30);
  const humidityLoad = Math.max(0, humidity - 65) * 0.5;
  const score = heatExcess * 2 + humidityLoad;

  if (score >= 18) return "Very high · prioritize shade";
  if (score >= 10) return "High · hydrate often";
  if (score >= 5) return "Moderate · pace activity";
  return "Low · easy to handle";
};

const describeHumidityMood = (humidity?: number): string => {
  if (humidity === undefined || humidity === null) {
    return "—";
  }
  if (humidity >= 85) return `${Math.round(humidity)}% (Tropical)`;
  if (humidity >= 70) return `${Math.round(humidity)}% (Muggy)`;
  if (humidity >= 50) return `${Math.round(humidity)}% (Comfortable)`;
  if (humidity >= 35) return `${Math.round(humidity)}% (Dry)`;
  return `${Math.round(humidity)}% (Parched)`;
};

const parseWindSpeed = (wind?: string): number | null => {
  if (!wind) return null;
  const match = wind.match(/([\d.]+)/);
  if (!match) return null;
  const parsed = parseFloat(match[1]);
  return Number.isNaN(parsed) ? null : parsed;
};

const describeWind = (wind?: string): string => {
  if (!wind) return "—";
  const speed = parseWindSpeed(wind);
  if (speed === null) return wind;
  if (speed < 6) return `${wind} (Calm)`;
  if (speed < 15) return `${wind} (Gentle breeze)`;
  if (speed < 25) return `${wind} (Breezy)`;
  return `${wind} (Windy)`;
};

const formatComfortIndex = (temperature?: number, humidity?: number): string => {
  if (temperature === undefined && humidity === undefined) {
    return "—";
  }

  let score = 90;
  if (temperature !== undefined && temperature !== null) {
    score -= Math.abs(temperature - 27) * 3.5;
  }
  if (humidity !== undefined && humidity !== null) {
    if (humidity > 55) {
      score -= (humidity - 55) * 0.8;
    } else {
      score -= (55 - humidity) * 0.35;
    }
  }

  const clamped = Math.round(clamp(score, 0, 100));
  let label = "Oppressive";
  if (clamped >= 75) {
    label = "Comfortable";
  } else if (clamped >= 55) {
    label = "Manageable";
  } else if (clamped >= 35) {
    label = "Sticky";
  }
  return `${clamped} / 100 (${label})`;
};

export function CurrentConditions() {
  const [data, setData] = useState<NowData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGet<NowData>("/now").then(setData).finally(() => setLoading(false));
  }, []);

  return (
    <Card className="shadow-md h-full">
      <CardHeader>
        <CardTitle>{data?.location ? `Right Now in ${data.location}` : "Right Now"}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-center gap-6 pb-6 border-b border-border">
          <CloudSun className="w-28 h-28 text-primary" strokeWidth={1.5} />
          <div>
            <div className="text-7xl font-mono leading-none mb-2">{loading ? "…" : formatTemperature(data?.temperature)}</div>
            <p className="text-muted-foreground">{data?.conditions ?? "—"}</p>
          </div>
        </div>
        <div className="space-y-1">
          <MetricRow icon={<Thermometer className="w-5 h-5" />} label="Feels Like" value={formatTemperature(data?.feels_like)} />
          <MetricRow icon={<Flame className="w-5 h-5" />} label="Heat Stress" value={classifyHeatStress(data?.temperature, data?.humidity)} />
          <MetricRow icon={<Droplets className="w-5 h-5" />} label="Humidity" value={describeHumidityMood(data?.humidity)} />
          <MetricRow icon={<Wind className="w-5 h-5" />} label="Wind" value={describeWind(data?.wind)} />
          <MetricRow icon={<Sun className="w-5 h-5" />} label="Comfort Index" value={formatComfortIndex(data?.temperature, data?.humidity)} />
          <MetricRow icon={<Sunrise className="w-5 h-5" />} label="Updated" value={data?.updated_at ? new Date(data.updated_at).toLocaleTimeString() : "—"} />
        </div>
        <div className="pt-4 border-t border-border">
          <p className="text-xs text-muted-foreground text-center">
            {data?.updated_at ? `Updated at ${new Date(data.updated_at).toLocaleString()}` : "—"}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
