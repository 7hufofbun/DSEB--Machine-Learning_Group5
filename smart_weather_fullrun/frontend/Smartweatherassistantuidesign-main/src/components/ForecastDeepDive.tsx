import { CSSProperties, useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell, 
  ReferenceLine
} from "recharts";
import { 
  Sun, 
  TrendingUp, 
  Droplets, 
  Cloud, 
  Wind, 
  ArrowUp, 
  ArrowDown, 
  Calendar,
  ThermometerSun,
  Info
} from "lucide-react";
import { apiGet } from "../lib/api";

// Icon mapping for weather features
const featureIcons = {
  solarradiation: Sun,
  temp_lag1: TrendingUp,
  humidity: Droplets,
  cloudcover: Cloud,
  windspeed: Wind,
  forecast_delta: ThermometerSun,
};

type TakeawayTone = "warm" | "cool" | "neutral";

interface TakeawayTheme {
  card: string;
  content: string;
  iconBg: string;
  icon: string;
  title: string;
  right: string;
  rightValue: string;
  rightCaption: string;
  background: string;
  shadow: string;
}

interface TakeawaySummary {
  tone: TakeawayTone;
  highlight: string;
  highlightSuffix: string;
}

const TAKEAWAY_THEMES: Record<TakeawayTone, TakeawayTheme> = {
  warm: {
    card: "border border-[rgba(251,146,60,0.35)] text-amber-900",
    content: "text-amber-900",
    iconBg: "bg-white/80",
    icon: "text-amber-500",
    title: "text-amber-900",
    right: "border border-white/40 bg-white/60 text-amber-900",
    rightValue: "text-amber-900",
    rightCaption: "text-amber-700",
    background: "linear-gradient(165deg, rgba(255,247,237,0.96) 0%, rgba(254,226,177,0.86) 55%, rgba(253,186,116,0.78) 100%)",
    shadow: "0 30px 58px -28px rgba(251,146,60,0.32)",
  },
  cool: {
    card: "border border-[rgba(56,189,248,0.35)] text-sky-900",
    content: "text-sky-900",
    iconBg: "bg-white/80",
    icon: "text-sky-500",
    title: "text-sky-900",
    right: "border border-white/40 bg-white/60 text-sky-900",
    rightValue: "text-sky-900",
    rightCaption: "text-sky-700",
    background: "linear-gradient(165deg, rgba(236,254,255,0.96) 0%, rgba(191,219,254,0.84) 55%, rgba(165,243,252,0.78) 100%)",
    shadow: "0 30px 58px -28px rgba(56,189,248,0.32)",
  },
  neutral: {
    card: "border border-[rgba(148,163,184,0.32)] text-slate-900",
    content: "text-slate-900",
    iconBg: "bg-white/80",
    icon: "text-slate-600",
    title: "text-slate-900",
    right: "border border-white/50 bg-white/70 text-slate-900",
    rightValue: "text-slate-900",
    rightCaption: "text-slate-600",
    background: "linear-gradient(165deg, rgba(248,250,252,0.96) 0%, rgba(226,232,240,0.84) 100%)",
    shadow: "0 28px 54px -28px rgba(148,163,184,0.32)",
  },
};

const HISTORICAL_RANGE_GRADIENTS: Record<TakeawayTone, { range: string; middle: string }> = {
  warm: {
    range: "linear-gradient(90deg, rgba(14,165,233,0.12) 0%, rgba(251,146,60,0.22) 100%)",
    middle: "linear-gradient(90deg, rgba(59,130,246,0.18) 0%, rgba(249,115,22,0.32) 100%)",
  },
  cool: {
    range: "linear-gradient(90deg, rgba(14,165,233,0.16) 0%, rgba(251,191,36,0.18) 100%)",
    middle: "linear-gradient(90deg, rgba(37,99,235,0.24) 0%, rgba(245,158,11,0.28) 100%)",
  },
  neutral: {
    range: "linear-gradient(90deg, rgba(96,165,250,0.12) 0%, rgba(248,181,94,0.18) 100%)",
    middle: "linear-gradient(90deg, rgba(59,130,246,0.2) 0%, rgba(251,146,60,0.26) 100%)",
  },
};

interface DeepDiveFeature {
  name: string;
  value?: number;
  description: string;
  actualValue?: number;
  unit?: string;
  vsAverage?: string;
  monthlyAvg?: number;
}

interface DeepDiveHistoricalRange {
  min?: number;
  percentile25?: number;
  avg?: number;
  percentile75?: number;
  max?: number;
}

interface DeepDiveAnalogueDay {
  date: string;
  label: string;
  actualTemp?: number;
  similarity?: number;
}

interface DeepDiveDay {
  id: string;
  label: string;
  day?: string;
  date?: string;
  baseValue?: number;
  finalPrediction?: number;
  features: DeepDiveFeature[];
  historicalRange?: DeepDiveHistoricalRange;
  analogueDays: DeepDiveAnalogueDay[];
}

interface DeepDiveResponse {
  today?: string;
  source?: string;
  items: DeepDiveDay[];
}

export function ForecastDeepDive() {
  const [days, setDays] = useState<DeepDiveDay[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        setIsLoading(true);
        const response = await apiGet<DeepDiveResponse>("/forecast_deep_dive");
        if (cancelled) return;
        const items = response?.items ?? [];
        setDays(items);
        setSelectedId(items[0]?.id);
        setSource(response?.source ?? null);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Failed to load forecast insights";
        setError(message);
        setDays([]);
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const selected = useMemo(
    () => days.find(day => day.id === selectedId) ?? (days.length ? days[0] : undefined),
    [days, selectedId],
  );

  const topFeatures = useMemo(() => {
    const features = selected?.features ?? [];
    const ordered = features.filter(
      feature => typeof feature.value === "number" && feature.name !== "forecast_delta",
    );
    const ranked = [...ordered].sort((a, b) => Math.abs((b.value ?? 0)) - Math.abs((a.value ?? 0)));
    const include = new Set(ranked.slice(0, 3).map(feature => feature.name));
    const humidityFeature = ranked.find(feature => feature.name === "humidity");
    if (humidityFeature) {
      include.add(humidityFeature.name);
    }
    return ordered.filter(feature => include.has(feature.name));
  }, [selected]);

  const baseValue = selected?.baseValue ?? selected?.finalPrediction ?? 0;
  const finalPrediction = selected?.finalPrediction ?? baseValue;

  const formatSignedChange = (value: number) => {
    if (Number.isNaN(value)) return "—";
    const abs = Math.abs(value);
    const decimals = abs >= 10 ? 0 : 1;
    const rounded = abs.toFixed(decimals);
    if (value > 0) return `+${rounded}`;
    if (value < 0) return `-${rounded}`;
    return "0";
  };

  const { waterfallData, totalDelta } = useMemo(() => {
    if (!selected) return { waterfallData: [], totalDelta: 0 };
    const contributions = selected.features
      .filter(feature => feature.name !== "forecast_delta")
      .map(feature => {
        const value = typeof feature.value === "number" ? feature.value : 0;
        const friendlyName =
          feature.name === "temp_lag1"
            ? "Yesterday"
            : feature.name === "solarradiation"
              ? "Sunshine"
              : feature.name === "cloudcover"
                ? "Cloud cover"
                : feature.name === "windspeed"
                  ? "Wind"
                  : feature.description;
        return {
          name: friendlyName,
          value,
          type: value >= 0 ? "warming" : "cooling",
          label: `${formatSignedChange(value)}°C`,
        };
      });
    const contributionSum = contributions.reduce((sum, item) => sum + item.value, 0);
    const finalDelta = typeof finalPrediction === "number" && typeof baseValue === "number"
      ? finalPrediction - baseValue
      : contributionSum;
    const rows = [
      ...contributions,
      {
        name: "Today’s forecast",
        value: finalDelta,
        type: "final",
        label: `${formatSignedChange(finalDelta)}°C`,
      },
    ];
    return { waterfallData: rows, totalDelta: finalDelta };
  }, [selected, baseValue, finalPrediction]);

  const historical = selected?.historicalRange;
  const rangeMin = historical?.min ?? 0;
  const rangeMax = historical?.max ?? rangeMin + 1;
  const rangeSpan = rangeMax - rangeMin || 1;
  const percentile25 = historical?.percentile25 ?? rangeMin;
  const percentile75 = historical?.percentile75 ?? rangeMax;
  const averageValue = historical?.avg ?? (rangeMin + rangeMax) / 2;

  const positionPercent = (value: number) => Math.min(100, Math.max(0, ((value - rangeMin) / rangeSpan) * 100));

  const formatNumber = (value?: number | null, digits = 1) => {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return value.toFixed(digits);
  };

  const axisExtent = useMemo(() => {
    const maxAbsChange = Math.max(1, ...waterfallData.map(item => Math.abs(item.value)));
    return Number((maxAbsChange + 0.5).toFixed(1));
  }, [waterfallData]);

  const axisTicks = useMemo(() => {
    if (axisExtent <= 0) return [0];
    return [-axisExtent, 0, axisExtent];
  }, [axisExtent]);

  const formatDayDisplay = (day?: DeepDiveDay) => {
    if (!day) return "";
    let label = day.label ?? day.day ?? day.id;
    if (day.date) {
      const parsed = new Date(day.date);
      if (!Number.isNaN(parsed.getTime())) {
        label = new Intl.DateTimeFormat("en-US", {
          weekday: "short",
          month: "short",
          day: "numeric",
          year: "numeric",
        }).format(parsed);
      }
    }
    label = label.replace(/\s*\([^)]*\)$/, "");
    const temperature = typeof day.finalPrediction === "number" ? `${formatNumber(day.finalPrediction)}°C` : null;
    return temperature ? `${label} · ${temperature}` : label;
  };

  const renderChangeTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
    if (!active || !payload?.length) return null;
    const item = payload[0];
    const numeric = typeof item.value === "number" ? item.value : Number(item.value);
    let message = "";
    if (item.payload?.type === "final") {
      if (Math.abs(numeric) < 0.1) {
        message = "Overall change is right on the usual temperature.";
      } else {
        const direction = numeric > 0 ? "warmer" : "cooler";
        message = `Overall change: ${formatSignedChange(numeric)}°C (${direction}).`;
      }
    } else if (Math.abs(numeric) < 0.1) {
      message = "Barely moves the temperature.";
    } else {
      const direction = numeric > 0 ? "warmer" : "cooler";
      message = `Pushes it ${direction} by ${formatSignedChange(numeric)}°C.`;
    }

    return (
      <div
        style={{
          backgroundColor: "#ffffff",
          border: "1px solid #e0e0e0",
          borderRadius: "8px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
          padding: "8px 12px",
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 4 }}>{item.payload?.name}</div>
        <div style={{ fontSize: 13, color: "#1f2937" }}>{message}</div>
      </div>
    );
  };

  const takeaway = useMemo<TakeawaySummary>(() => {
    const absDelta = Math.abs(totalDelta);
    if (!Number.isFinite(totalDelta) || absDelta < 0.1) {
      return {
        tone: "neutral" as TakeawayTone,
        highlight: "≈0°C",
        highlightSuffix: "No change",
      };
    }

    const isWarmer = totalDelta > 0;
    const tone: TakeawayTone = isWarmer ? "warm" : "cool";
    let descriptor: string;
    if (absDelta >= 3) {
      descriptor = isWarmer ? "Much warmer" : "Much cooler";
    } else if (absDelta >= 1.5) {
      descriptor = isWarmer ? "Noticeably warmer" : "Noticeably cooler";
    } else if (absDelta >= 0.5) {
      descriptor = isWarmer ? "Warmer" : "Cooler";
    } else {
      descriptor = isWarmer ? "Slightly warmer" : "Slightly cooler";
    }

    const sign = isWarmer ? "+" : "-";
    const digits = absDelta >= 10 ? 0 : 1;
    const formatted = formatNumber(absDelta, digits);

    return {
      tone,
      highlight: `${sign}${formatted}°C`,
      highlightSuffix: descriptor,
    };
  }, [totalDelta]);

  const theme = TAKEAWAY_THEMES[takeaway.tone];
  const historicalGradient = HISTORICAL_RANGE_GRADIENTS[takeaway.tone];
  const takeawayCardStyle: CSSProperties = { background: theme.background, boxShadow: theme.shadow };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Card className="shadow-md">
          <CardContent className="py-12 text-center text-muted-foreground">
            Loading forecast insights…
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <Card className="shadow-md border-red-200 bg-red-50">
          <CardContent className="py-6 text-center text-red-700">
            {error}
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!selected) {
    return (
      <div className="space-y-6">
        <Card className="shadow-md">
          <CardContent className="py-12 text-center text-muted-foreground">
            Forecast insights are not available right now.
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <Card className="shadow-md">
            <CardHeader className="pb-0">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
                <div className="flex flex-row flex-wrap items-center gap-2 sm:flex-nowrap">
                  <CardTitle className="text-base font-semibold text-slate-900 whitespace-nowrap">
                    Why Today Is Warmer or Cooler
                  </CardTitle>
                  <button
                    type="button"
                    className="text-slate-400 transition-colors hover:text-slate-600"
                    aria-label="What does this chart show?"
                    title="Each bar shows how a weather factor nudges today’s temperature up or down compared with the usual for this week."
                  >
                    <Info className="h-4 w-4" aria-hidden="true" />
                  </button>
                </div>
                <div className="flex w-full flex-col gap-1 sm:w-auto">
                  <Select value={selectedId} onValueChange={value => setSelectedId(value)}>
                    <SelectTrigger
                      id="deep-dive-day"
                      aria-label="Select forecast day"
                      className="flex w-full items-center justify-between gap-1.5 rounded-full border border-slate-300/70 bg-white/80 px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:border-amber-400 hover:bg-amber-50 focus:outline-none focus:ring-2 focus:ring-amber-500 data-[state=open]:border-amber-500 sm:w-[200px]"
                    >
                      <SelectValue placeholder="Select day" />
                    </SelectTrigger>
                    <SelectContent className="text-sm">
                      {days.map(day => (
                        <SelectItem key={day.id} value={day.id} className="text-sm">
                          {formatDayDisplay(day)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6 pt-4">
              <div className="relative h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={waterfallData}
                    layout="vertical"
                    margin={{ top: 20, right: 60, left: 100, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" horizontal={false} />
                    <XAxis
                      type="number"
                      axisLine={true}
                      tickLine={false}
                      tick={{ fill: "#6c757d", fontSize: 12 }}
                      label={{ value: "Change from usual (°C)", position: "bottom", style: { fill: "#475569", fontSize: 12 } }}
                      domain={[-axisExtent, axisExtent]}
                      ticks={axisTicks}
                      tickFormatter={value => `${formatSignedChange(value)}°`}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#1a1a1a", fontSize: 13 }}
                      width={90}
                    />
                    <Tooltip content={renderChangeTooltip} />
                    <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="4 4" strokeWidth={1.5} />
                    <Bar dataKey="value" radius={[4, 4, 4, 4]}>
                      {waterfallData.map((entry, index) => {
                        let fill = "#6c757d";
                        if (entry.type === "warming") fill = "#dc3545";
                        if (entry.type === "cooling") fill = "#007BFF";
                        if (entry.type === "final") fill = entry.value >= 0 ? "#22c55e" : "#0ea5e9";
                        return <Cell key={`cell-${index}`} fill={fill} />;
                      })}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="-mt-2 flex justify-center px-6 text-xs text-slate-600">
                <span className="font-medium">Usual for this week</span>
              </div>
              <Card className={`overflow-hidden ${theme.card}`} style={takeawayCardStyle}>
                <CardContent className={`flex flex-col gap-6 px-6 py-6 text-sm sm:flex-row sm:items-start sm:justify-between ${theme.content}`}>
                  <div className="flex flex-1 flex-col gap-3 sm:basis-2/3">
                    <div className="flex items-start gap-3">
                      <div className={`rounded-full p-3 shadow-sm ${theme.iconBg}`}>
                        <ThermometerSun className={`h-6 w-6 ${theme.icon}`} />
                      </div>
                      <CardTitle className={`mt-1 text-base font-semibold ${theme.title}`}>
                        How today compares with usual
                      </CardTitle>
                    </div>
                  </div>
                  <div className={`flex flex-col gap-1 rounded-2xl px-5 py-4 text-right sm:basis-1/3 sm:ml-6 sm:pl-6 ${theme.right}`}>
                    <span className={`text-xl font-semibold leading-tight sm:text-2xl ${theme.rightValue}`}>{takeaway.highlight}</span>
                    <span className={`text-xs font-medium sm:text-sm ${theme.rightCaption}`}>{takeaway.highlightSuffix}</span>
                  </div>
                </CardContent>
              </Card>
            </CardContent>
          </Card>

          <Card className="shadow-md">
            <CardHeader>
              <CardTitle>Top Things To Know</CardTitle>
              <p className="mt-2 text-sm text-muted-foreground">
                A quick, plain-language rundown of what&rsquo;s pushing the temperature up or down today.
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {topFeatures.map((feature, idx) => {
                  const IconComponent = featureIcons[feature.name as keyof typeof featureIcons] ?? ThermometerSun;
                  const value = feature.value ?? 0;
                  const isWarming = value >= 0;
                  const bgColor = isWarming ? "bg-red-50" : "bg-blue-50";
                  const borderColor = isWarming ? "border-red-200" : "border-blue-200";
                  const textColor = isWarming ? "text-red-800" : "text-blue-800";
                  const iconColor = isWarming ? "text-red-600" : "text-blue-600";
                  const badgeColor = isWarming ? "bg-red-100 text-red-700" : "bg-blue-100 text-blue-700";
                  const unitSuffix = feature.unit ? ` ${feature.unit}` : "";
                  const todayValue = feature.actualValue !== undefined ? `${formatNumber(feature.actualValue)}${unitSuffix}` : null;
                  const typicalValue = feature.monthlyAvg !== undefined ? `${formatNumber(feature.monthlyAvg)}${unitSuffix}` : null;
                  const differenceTextRaw = feature.vsAverage && feature.vsAverage !== "—" ? feature.vsAverage : null;
                  const differencePercent = differenceTextRaw
                    ?.replace(/\s*compared with the usual\.*/i, "")
                    ?.replace(/\s*vs\s*typical\.*/i, "")
                    ?.replace(/\s*thường ngày\.*/i, "")
                    ?.trim();
                  const currentDetail = todayValue
                    ? `Current: ${todayValue}${differencePercent ? ` (${differencePercent})` : ""}`
                    : "—";
                  const summaryParts = [
                    typicalValue ? `Typical: ${typicalValue}` : null,
                  ].filter(Boolean) as string[];
                  const summaryText = summaryParts.join(" · ");
                  const friendlyName =
                    feature.name === "temp_lag1"
                      ? "Yesterday"
                      : feature.name === "solarradiation"
                        ? "Sunshine"
                        : feature.name === "cloudcover"
                          ? "Cloud cover"
                          : feature.name === "windspeed"
                            ? "Wind"
                            : feature.description;
                  return (
                    <div key={idx} className={`p-4 border rounded-lg ${bgColor} ${borderColor}`}>
                      <div className="mb-3 flex items-start justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg bg-white border ${borderColor}`}>
                            <IconComponent className={`h-5 w-5 ${iconColor}`} />
                          </div>
                          <div>
                            <div className={textColor}>{friendlyName}</div>
                            <div className="mt-0.5 font-mono text-sm text-muted-foreground">
                                {currentDetail}
                            </div>
                          </div>
                        </div>
                        <div className={`px-2.5 py-1 font-mono text-sm rounded ${badgeColor}`}>
                          {value >= 0 ? "+" : ""}{formatNumber(value)}°C
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {summaryText || "No context available yet."}
                      </p>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle>How Today Stacks Up</CardTitle>
              <p className="mt-2 text-sm text-muted-foreground">
                Compare the forecast with typical temperatures for the same time of year.
              </p>
            </CardHeader>
            <CardContent>
              {historical ? (
                <div className="space-y-6">
                  <div className="py-8">
                    <div className="relative h-20 px-4">
                      <div className="absolute -top-6 left-0 right-0 flex justify-between font-mono text-xs text-muted-foreground">
                        <span>{formatNumber(rangeMin, 0)}°C</span>
                        <span>{formatNumber(rangeMax, 0)}°C</span>
                      </div>
                      <div
                        className="absolute font-mono text-xs text-muted-foreground"
                        style={{
                          left: `${positionPercent(averageValue)}%`,
                          transform: "translateX(-50%)",
                          top: -26,
                        }}
                      >
                        {formatNumber(averageValue)}°C
                      </div>
                      <div className="relative flex h-12 items-center">
                        <div
                          className="absolute h-8 w-full rounded border border-gray-300/70 bg-transparent backdrop-blur-sm"
                          style={{ background: historicalGradient.range }}
                        />
                        <div
                          className="absolute h-8 rounded border border-gray-400/70 bg-transparent"
                          style={{
                            left: `${positionPercent(percentile25)}%`,
                            width: `${Math.max(positionPercent(percentile75) - positionPercent(percentile25), 2)}%`,
                            background: historicalGradient.middle,
                          }}
                        />
                        <div
                          className="absolute h-10 w-0.5 border-l-2 border-dashed border-[#6c757d] bg-[#6c757d]"
                          style={{ left: `${positionPercent(averageValue)}%` }}
                        />
                        <div
                          className="absolute -mt-1"
                          style={{ left: `${positionPercent(finalPrediction)}%`, transform: "translateX(-50%)" }}
                        >
                          <div className="flex flex-col items-center">
                            <div className="h-4 w-4 rounded-full border-2 border-white bg-[#28a745] shadow-md">
                              <div className="h-full w-full rounded-full border-2 border-[#28a745]" />
                            </div>
                            <div className="mt-2 whitespace-nowrap rounded bg-green-100 px-2 py-0.5 font-mono text-xs text-green-800">
                              {formatNumber(finalPrediction)}°C
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="h-3 w-6 rounded border border-gray-300 bg-[#e9ecef]" />
                      <span className="text-muted-foreground">Historical Range</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-3 w-6 rounded border border-gray-300 bg-[#ced4da]" />
                      <span className="text-muted-foreground">25th-75th Percentile</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-0.5 w-6 border-t-2 border-dashed border-[#6c757d] bg-[#6c757d]" />
                      <span className="text-muted-foreground">Historical Average</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full border-2 border-white bg-[#28a745] shadow-sm" />
                      <span className="text-muted-foreground">Forecast</span>
                    </div>
                  </div>

                  <div className="rounded-lg border border-green-200 bg-green-50 p-4">
                    <div className="flex items-start gap-2">
                      <ThermometerSun className="mt-0.5 h-5 w-5 flex-shrink-0 text-green-600" />
                      <div className="text-sm text-green-800">
                        {finalPrediction > percentile75 && (
                          <>The forecast of <span className="font-mono">{formatNumber(finalPrediction)}°C</span> is <span className="font-semibold">warmer than typical</span>, above the 75th percentile.</>
                        )}
                        {finalPrediction < percentile25 && (
                          <>The forecast of <span className="font-mono">{formatNumber(finalPrediction)}°C</span> is <span className="font-semibold">cooler than typical</span>, below the 25th percentile.</>
                        )}
                        {finalPrediction >= percentile25 && finalPrediction <= percentile75 && (
                          <>The forecast of <span className="font-mono">{formatNumber(finalPrediction)}°C</span> sits <span className="font-semibold">within the normal range</span> for this time of year.</>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3">
                    <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-center">
                      <div className="mb-1 text-xs text-muted-foreground">Historical Avg</div>
                      <div className="font-mono text-lg">{formatNumber(averageValue)}°C</div>
                    </div>
                    <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-center">
                      <div className="mb-1 text-xs text-muted-foreground">Typical Range</div>
                      <div className="font-mono text-lg">{formatNumber(percentile25)}-{formatNumber(percentile75)}°C</div>
                    </div>
                    <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-center">
                      <div className="mb-1 text-xs text-muted-foreground">Record Range</div>
                      <div className="font-mono text-lg">{formatNumber(rangeMin)}-{formatNumber(rangeMax)}°C</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="py-12 text-center text-muted-foreground">Historical context unavailable.</div>
              )}
            </CardContent>
          </Card>

          <Card className="shadow-md">
            <CardHeader>
              <CardTitle>Days That Felt Similar</CardTitle>
              <p className="mt-2 text-sm text-muted-foreground">
                Real dates from Ho Chi Minh City that shared nearly the same mix of conditions.
              </p>
            </CardHeader>
            <CardContent>
              {selected.analogueDays.length ? (
                <div className="space-y-3">
                  {selected.analogueDays.map((day, idx) => (
                    <div key={idx} className="rounded-lg border border-blue-200 bg-blue-50 p-4">
                      <div className="mb-2 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="rounded-lg border border-blue-200 bg-white p-2">
                            <Calendar className="h-5 w-5 text-blue-600" />
                          </div>
                          <div>
                            <div className="text-blue-900">{day.label}</div>
                            <div className="text-sm text-blue-700">{day.similarity ?? 0}% similarity match</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="mb-1 text-xs text-blue-600">Actual Temp</div>
                          <div className="font-mono text-xl text-blue-900">{formatNumber(day.actualTemp)}°C</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-blue-800">
                        {day.actualTemp !== undefined && Math.abs((day.actualTemp ?? 0) - finalPrediction) < 0.5 ? (
                          <span className="inline-flex items-center gap-1">
                            <span className="h-2 w-2 rounded-full bg-green-500" />
                            Nearly identical to current forecast
                          </span>
                        ) : day.actualTemp !== undefined && (day.actualTemp ?? 0) > finalPrediction ? (
                          <>
                            <ArrowUp className="h-4 w-4" />
                            <span>{formatNumber((day.actualTemp ?? 0) - finalPrediction)}°C warmer than forecast</span>
                          </>
                        ) : day.actualTemp !== undefined ? (
                          <>
                            <ArrowDown className="h-4 w-4" />
                            <span>{formatNumber(finalPrediction - (day.actualTemp ?? 0))}°C cooler than forecast</span>
                          </>
                        ) : (
                          <span>Temperature data unavailable.</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-12 text-center text-muted-foreground">
                  No similar historical days found.
                </div>
              )}

              <div className="mt-4 rounded-lg border border-gray-200 bg-gray-50 p-4">
                <p className="text-sm text-muted-foreground leading-relaxed">
                  We look for past days that felt the same &mdash; similar humidity, cloud cover, and breeze. If those
                  days ended close to today&rsquo;s forecast, it boosts our confidence in the outlook.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {source ? (
        <p className="text-xs text-muted-foreground text-right">
          Powered by {source === "onnx" ? "our temperature model" : source === "seasonal" ? "seasonal trends" : "recent historical data"}.
        </p>
      ) : null}
    </div>
  );
}
