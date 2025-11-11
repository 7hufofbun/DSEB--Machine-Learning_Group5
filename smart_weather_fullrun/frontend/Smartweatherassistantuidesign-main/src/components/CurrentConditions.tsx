import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Droplets, Wind, CloudSun, Thermometer, Sun, Sunrise, Sunset } from "lucide-react";
import { apiGet } from "../lib/api";

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
            <div className="text-7xl font-mono leading-none mb-2">{loading ? "…" : `${Math.round(data?.temperature ?? 0)}°C`}</div>
            <p className="text-muted-foreground">{data?.conditions ?? "—"}</p>
          </div>
        </div>
        <div className="space-y-1">
          <MetricRow icon={<Thermometer className="w-5 h-5" />} label="Feels Like" value={data?.feels_like ? `${Math.round(data.feels_like)}°C` : "-"} />
          <MetricRow icon={<Droplets className="w-5 h-5" />} label="Humidity" value={data?.humidity ? `${data.humidity}%` : "-"} />
          <MetricRow icon={<Wind className="w-5 h-5" />} label="Wind" value={data?.wind ?? "-"} />
          <MetricRow icon={<Sun className="w-5 h-5" />} label="UV Index" value={"—"} />
          <MetricRow icon={<Sunrise className="w-5 h-5" />} label="Sunrise" value={"—"} />
          <MetricRow icon={<Sunset className="w-5 h-5" />} label="Sunset" value={"—"} />
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
