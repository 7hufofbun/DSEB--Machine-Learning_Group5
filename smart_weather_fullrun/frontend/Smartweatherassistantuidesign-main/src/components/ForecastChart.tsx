import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart } from "recharts";
import { Lightbulb, CloudRain, CloudSun, Cloud, Sun, CloudDrizzle, ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import { apiPost } from "../lib/api";
import { formatTemperature, roundToTenth } from "../lib/format";

type Item = {
  day: string;
  temp_avg: number;
  temp_min: number;
  temp_max: number;
  condition: string;
  precipChance: number;
  calendarDate?: string;
  displayDate?: string;
};

const getWeatherIcon = (condition: string) => {
  if (condition.includes("Rain")) return <CloudRain className="w-6 h-6" />;
  if (condition.includes("Sunny")) return <Sun className="w-6 h-6" />;
  if (condition.includes("Partly Cloudy")) return <CloudSun className="w-6 h-6" />;
  if (condition.includes("Drizzle")) return <CloudDrizzle className="w-6 h-6" />;
  return <Cloud className="w-6 h-6" />;
};

const formatDayLabel = (label: string): string => {
  const normalised = label.trim();
  if (normalised.length === 0) return normalised;
  const lookup: Record<string, string> = {
    mon: "Monday",
    tue: "Tuesday",
    wed: "Wednesday",
    thu: "Thursday",
    fri: "Friday",
    sat: "Saturday",
    sun: "Sunday",
  };
  const key = normalised.slice(0, 3).toLowerCase();
  if (lookup[key]) return lookup[key];
  return normalised
    .toLowerCase()
    .split(" ")
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
};

const attachForecastDate = (list: Item[]): Item[] => {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  return list.map((entry, index) => {
    const date = new Date(start);
    date.setDate(start.getDate() + index);
    return {
      ...entry,
      calendarDate: date.toISOString(),
      displayDate: date.toLocaleDateString("en-US", { month: "long", day: "numeric" }),
    };
  });
};

const temperatureTone = (temp: number) => {
  if (temp >= 35) {
    return {
      gradient: "linear-gradient(135deg,#f97316 0%,#ef4444 45%,#b91c1c 100%)",
      textClass: "text-white",
      shadow: "0 18px 36px rgba(239,68,68,0.4)",
    };
  }
  if (temp >= 32) {
    return {
      gradient: "linear-gradient(135deg,#f59e0b 0%,#f97316 50%,#ea580c 100%)",
      textClass: "text-white",
      shadow: "0 18px 32px rgba(245,158,11,0.35)",
    };
  }
  if (temp >= 29) {
    return {
      gradient: "linear-gradient(135deg,#fbbf24 0%,#facc15 45%,#22c55e 100%)",
      textClass: "text-slate-900",
      shadow: "0 16px 28px rgba(250,204,21,0.35)",
    };
  }
  if (temp >= 26) {
    return {
      gradient: "linear-gradient(135deg,#34d399 0%,#14b8a6 45%,#38bdf8 100%)",
      textClass: "text-white",
      shadow: "0 16px 28px rgba(45,212,191,0.35)",
    };
  }
  return {
    gradient: "linear-gradient(135deg,#22d3ee 0%,#2563eb 45%,#312e81 100%)",
    textClass: "text-white",
    shadow: "0 18px 32px rgba(59,130,246,0.35)",
  };
};

const temperatureTagline = (temp: number): string => {
  if (temp >= 35) return "Heatwave alert";
  if (temp >= 32) return "Sweltering stretch";
  if (temp >= 29) return "Tropical toastiness";
  if (temp >= 26) return "Balmy comfort";
  return "Cooler break";
};

const describeTemperature = (temp: number): string => {
  if (temp >= 35) return "Coffee turns to iced the moment it leaves the cup—hydrate nonstop.";
  if (temp >= 32) return "Consider declaring siesta hours; the asphalt is basically a griddle.";
  if (temp >= 29) return "Perfect excuse for cold coconut water and a shady hammock mission.";
  if (temp >= 26) return "Stroll weather: bring sunglasses, skip the heavy layers.";
  return "Break out the light jacket and brag about the rare chill in Saigon.";
};

const actionHint = (temp: number): string => {
  if (temp >= 35) return "Plan: indoor everything.";
  if (temp >= 32) return "Plan: finish errands before noon.";
  if (temp >= 29) return "Plan: sunset cafe hop.";
  if (temp >= 26) return "Plan: evening walk works.";
  return "Plan: grab that lightweight jacket.";
};

const trendIcon = (delta: number | null) => {
  if (delta === null) return <Minus className="w-4 h-4 text-muted-foreground" />;
  if (delta > 0.05) return <ArrowUpRight className="w-4 h-4 text-red-500" />;
  if (delta < -0.05) return <ArrowDownRight className="w-4 h-4 text-blue-500" />;
  return <Minus className="w-4 h-4 text-muted-foreground" />;
};

export function ForecastChart() {
  const [items, setItems] = useState<Item[]>([]);
  useEffect(() => {
    apiPost<Item[]>("/forecast_detailed").then(data => {
      const normalised = data.map(item => {
        const avg = roundToTenth(item.temp_avg);
        const min = roundToTenth(item.temp_min);
        const max = roundToTenth(item.temp_max);
        return {
          ...item,
          temp_avg: avg ?? item.temp_avg,
          temp_min: min ?? item.temp_min,
          temp_max: max ?? item.temp_max,
        };
      });
      setItems(attachForecastDate(normalised));
    });
  }, []);

  return (
    <Card className="shadow-md h-full">
      <CardHeader>
        <CardTitle>The Week Ahead</CardTitle>
        <p className="text-sm text-muted-foreground">5-Day Forecast</p>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={items} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="tempRange" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#007BFF" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#007BFF" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" vertical={false} />
              <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#6c757d', fontSize: 14 }} />
              <YAxis axisLine={false} tickLine={false} tick={{ fill: '#6c757d', fontSize: 14 }} label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', style: { fill: '#6c757d' } }} domain={[20, 40]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e0e0e0', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
                formatter={(value: any) => {
                  const numeric = typeof value === "number" ? value : Number(value);
                  return [formatTemperature(numeric), ''];
                }}
                labelStyle={{ fontWeight: 600, marginBottom: 4 }}
              />
              <Line type="monotone" dataKey="temp_avg" stroke="#007BFF" strokeWidth={3} dot={{ fill: '#007BFF', r: 6, strokeWidth: 2, stroke: '#ffffff' }} activeDot={{ r: 8 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-3">
          <h4 className="text-sm text-muted-foreground">Daily Temperature Outlook</h4>
          <div className="space-y-3">
            {items.map((day, index) => {
              const prev = index > 0 ? items[index - 1] : null;
              const deltaRaw = prev ? roundToTenth(day.temp_avg - prev.temp_avg) : null;
              const deltaLabel = deltaRaw === null ? "Start" : `${deltaRaw > 0 ? "+" : ""}${deltaRaw.toFixed(1)}°C`;
              const deltaCaption = prev ? "vs previous day" : "starting point";
              const tone = temperatureTone(day.temp_avg);
              return (
                <div key={day.day} className="p-4 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors space-y-3">
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="flex-shrink-0 text-primary">{getWeatherIcon(day.condition)}</div>
                      <div className="min-w-0">
                        <div className="text-3xl font-semibold tracking-tight text-slate-900">{formatDayLabel(day.day)}</div>
                        {day.displayDate ? (
                          <div className="text-sm font-medium text-slate-600">{day.displayDate}</div>
                        ) : null}
                        <div className="text-xs font-semibold text-muted-foreground truncate">{temperatureTagline(day.temp_avg)}</div>
                      </div>
                    </div>
                    <div
                      className="relative overflow-hidden rounded-xl border border-white/20 px-4 py-2 text-right shadow-xl ring-1 ring-white/20 transition-all duration-200 hover:-translate-y-0.5"
                      style={{ backgroundImage: tone.gradient, boxShadow: tone.shadow }}
                    >
                      <span className={`text-3xl font-mono leading-none drop-shadow-sm ${tone.textClass}`}>{formatTemperature(day.temp_avg)}</span>
                      <div className={`text-[11px] font-semibold tracking-wide uppercase ${tone.textClass} opacity-80`}>avg temp</div>
                    </div>
                  </div>
                  <div className="text-sm text-muted-foreground leading-relaxed">{describeTemperature(day.temp_avg)}</div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                      {trendIcon(deltaRaw)}
                      <span className="font-mono text-sm">{deltaLabel}</span>
                      <span>{deltaCaption}</span>
                    </div>
                    <span className="text-xs font-semibold text-primary/80">{actionHint(day.temp_avg)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-blue-900 mb-1">Smart Insight</h4>
              <p className="text-sm text-blue-800">
                Forecast powered by your local ML backend. Confidence improves when ONNX models are available in the server.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
