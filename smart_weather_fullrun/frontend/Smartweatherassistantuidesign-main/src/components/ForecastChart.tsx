import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  type TooltipProps,
} from "recharts";
import {
  CloudRain,
  CloudSun,
  Cloud,
  Sun,
  CloudDrizzle,
  ArrowDownRight,
  ArrowUpRight,
  Minus,
  ChevronDown,
  CalendarDays,
} from "lucide-react";
import { apiGet, apiPost } from "../lib/api";
import { formatTemperature, roundToTenth } from "../lib/format";
import "./ForecastChart.css";

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

type NowResponse = {
  temperature?: number;
  feels_like?: number;
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

const resolveForecastCardTheme = (condition?: string): string => {
  if (!condition) return "forecast-card--theme-day-cloud";
  const lowered = condition.toLowerCase();
  if (lowered.includes("storm") || lowered.includes("thunder")) return "forecast-card--theme-day-storm";
  if (lowered.includes("rain") || lowered.includes("drizzle") || lowered.includes("shower")) return "forecast-card--theme-day-rain";
  if (lowered.includes("sun") || lowered.includes("clear")) return "forecast-card--theme-day-clear";
  return "forecast-card--theme-day-cloud";
};

const trendIcon = (delta: number | null) => {
  if (delta === null) return <Minus className="w-4 h-4" />;
  if (delta > 0.05) return <ArrowUpRight className="w-4 h-4" />;
  if (delta < -0.05) return <ArrowDownRight className="w-4 h-4" />;
  return <Minus className="w-4 h-4" />;
};

export function ForecastChart() {
  const [items, setItems] = useState<Item[]>([]);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const [currentTemp, setCurrentTemp] = useState<number | null>(null);
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const [forecastData, nowData] = await Promise.all([
          apiPost<Item[]>("/forecast_detailed").catch(() => []),
          apiGet<NowResponse>("/now").catch(() => null),
        ]);

        if (cancelled) return;

        const normalised = (forecastData ?? []).map(item => {
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

        const liveTemp = typeof nowData?.temperature === "number" ? nowData.temperature : typeof nowData?.feels_like === "number" ? nowData.feels_like : null;
        setCurrentTemp(liveTemp ?? null);
      } catch (error) {
        console.error("Failed to load forecast data", error);
      }
    };

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  const yDomain = useMemo<[number, number]>(() => {
    if (!items.length) return [20, 40];
    const temps = items.map(item => item.temp_avg);
    const min = Math.min(...temps);
    const max = Math.max(...temps);
    const paddedMin = Math.floor(min - 2);
    const paddedMax = Math.ceil(max + 2);
    return [paddedMin, paddedMax > paddedMin ? paddedMax : paddedMin + 4];
  }, [items]);

  const yTicks = useMemo<number[]>(() => {
    const [min, max] = yDomain;
    const steps = 3;
    if (max <= min) return [min];
    const interval = (max - min) / steps;
    return Array.from({ length: steps + 1 }, (_, index) => Number((min + interval * index).toFixed(1)));
  }, [yDomain]);

  const renderXTicks = (props: any): JSX.Element => {
    const { x, y, payload } = props;
    if (typeof x !== "number" || typeof y !== "number" || !payload) return <></>;
    const index: number | undefined = typeof payload.index === "number" ? payload.index : undefined;
    const isActive = index !== undefined && activeIndex === index;
    return (
      <text
        x={x}
        y={y + 16}
        textAnchor="middle"
        fill={isActive ? "#1d4ed8" : "#6B7280"}
        fontSize={13}
        fontWeight={isActive ? 600 : 500}
      >
        {payload.value}
      </text>
    );
  };

  const renderDot = (props: any): JSX.Element => {
    const { cx, cy } = props;
    if (typeof cx !== "number" || typeof cy !== "number") return <></>;
    return (
      <g>
        <circle
          cx={cx}
          cy={cy}
          r={7}
          fill="#2563eb"
          stroke="#ffffff"
          strokeWidth={2}
          style={{ filter: "drop-shadow(0 2px 6px rgba(37,99,235,0.35))" }}
        />
      </g>
    );
  };

  const renderActiveDot = (props: any): JSX.Element => {
    const { cx, cy } = props;
    if (typeof cx !== "number" || typeof cy !== "number") return <></>;
    return (
      <g>
        <circle cx={cx} cy={cy} r={12} fill="rgba(37,99,235,0.15)" />
        <circle
          cx={cx}
          cy={cy}
          r={9}
          fill="#2563eb"
          stroke="#ffffff"
          strokeWidth={2}
          style={{ filter: "drop-shadow(0 3px 8px rgba(37,99,235,0.35))" }}
        />
      </g>
    );
  };

  const renderTooltip = ({ active, label, payload }: TooltipProps<number, string>): JSX.Element | null => {
    if (!active || !payload || payload.length === 0) return null;
    const first = payload[0];
    const numeric = typeof first.value === "number" ? first.value : Number(first.value ?? 0);
    const title = formatDayLabel(String(label ?? ""));
    return (
      <div
        style={{
          backgroundColor: "#ffffff",
          border: "1px solid #e0e0e0",
          borderRadius: "12px",
          boxShadow: "0 6px 18px rgba(15,23,42,0.08)",
          padding: "12px 16px",
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: "6px", color: "#1f2937" }}>{title}</div>
        <div style={{ color: "#475569", fontWeight: 500 }}>{`Average temperature ${formatTemperature(numeric)}`}</div>
      </div>
    );
  };

  return (
    <Card className="forecast-chart shadow-md h-full">
      <CardHeader className="forecast-chart__header">
        <div className="forecast-chart__title-row">
          <CalendarDays className="forecast-chart__title-icon" />
          <CardTitle className="forecast-chart__title">The Week Ahead</CardTitle>
        </div>
        <p className="forecast-chart__subtitle">Average daytime temperature across the next five days</p>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={items}
              margin={{ top: 16, right: 16, left: 16, bottom: 8 }}
              onMouseMove={state => {
                if (state && typeof state.activeTooltipIndex === "number") {
                  setActiveIndex(state.activeTooltipIndex);
                }
              }}
              onMouseLeave={() => setActiveIndex(null)}
            >
              <defs>
                <linearGradient id="tempLineGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#93c5fd" />
                  <stop offset="50%" stopColor="#3b82f6" />
                  <stop offset="100%" stopColor="#1d4ed8" />
                </linearGradient>
                <linearGradient id="tempAreaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(59,130,246,0.15)" />
                  <stop offset="100%" stopColor="rgba(59,130,246,0.02)" />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 6" stroke="#E5E7EB" vertical={false} />
              <XAxis dataKey="day" axisLine={false} tickLine={false} tick={renderXTicks} interval={0} />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#64748b", fontSize: 13 }}
                domain={yDomain}
                ticks={yTicks}
                width={44}
                tickMargin={10}
              />
              <Tooltip content={renderTooltip} cursor={{ stroke: "#93c5fd", strokeDasharray: "4 4" }} />
              <Line
                type="monotone"
                dataKey="temp_avg"
                stroke="url(#tempLineGradient)"
                strokeWidth={2.5}
                dot={renderDot}
                activeDot={renderActiveDot}
                strokeLinecap="round"
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="temp_avg"
                stroke="none"
                fill="url(#tempAreaGradient)"
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-3">
          <h4 className="forecast-chart__section-heading">Daily Temperature Outlook</h4>
          <div className="flex flex-col gap-3">
            {items.map((day, index) => {
              const prev = index > 0 ? items[index - 1] : null;
              const baselineDelta = !prev && currentTemp !== null ? roundToTenth(day.temp_avg - currentTemp) : null;
              const deltaRaw = prev ? roundToTenth(day.temp_avg - prev.temp_avg) : baselineDelta;
              const deltaLabel = deltaRaw === null ? (prev ? "—" : "Start") : `${deltaRaw > 0 ? "+" : ""}${deltaRaw.toFixed(1)}°C`;
              const deltaCaption = prev ? "vs previous day" : currentTemp !== null ? "vs right now" : "starting point";
              const isOpen = openIndex === index;
              const deltaVariant =
                deltaRaw === null || deltaRaw === 0 ? "neutral" : deltaRaw > 0 ? "warm" : "cool";
              const deltaBlockClass = `forecast-card__delta-block forecast-card__delta-block--${deltaVariant}`;
              const deltaPillClass = `forecast-card__delta-pill forecast-card__delta-pill--${deltaVariant}`;
              const deltaToneClass = `forecast-card--delta-${deltaVariant}`;
              const deltaBadgeLabel =
                deltaRaw === null ? deltaCaption : `${deltaRaw > 0 ? "Warmer" : deltaRaw < 0 ? "Cooler" : "No change"}`;
              const deltaCaptionShort =
                deltaCaption === "vs previous day"
                  ? "vs prev"
                  : deltaCaption === "vs right now"
                  ? "vs now"
                  : "baseline";
              const forecastThemeClass = resolveForecastCardTheme(day.condition);
              return (
                <article
                  key={day.day}
                  className={["forecast-card", forecastThemeClass, deltaToneClass].filter(Boolean).join(" ")}
                >
                  <button
                    type="button"
                    className="forecast-card__trigger"
                    aria-expanded={isOpen}
                    onClick={() => setOpenIndex(prevIndex => (prevIndex === index ? null : index))}
                  >
                    <div className="forecast-card__left">
                      <div className="forecast-card__icon text-sky-500">{getWeatherIcon(day.condition)}</div>
                      <div className="forecast-card__text">
                        <span className="forecast-card__day">{formatDayLabel(day.day)}</span>
                        {day.displayDate ? <span className="forecast-card__date">{day.displayDate}</span> : null}
                      </div>
                    </div>
                    <div className="forecast-card__meta">
                      <div className={deltaBlockClass}>
                        <span className="forecast-card__delta-main">
                          {trendIcon(deltaRaw)}
                          <span>{deltaLabel}</span>
                          <span className="forecast-card__delta-caption">{deltaCaptionShort}</span>
                        </span>
                        <span className={deltaPillClass}>
                          {deltaCaption === "starting point" ? "Baseline" : deltaBadgeLabel}
                        </span>
                      </div>
                      <div className="forecast-card__temp-chip">
                        <span className="forecast-card__temp-value">{formatTemperature(day.temp_avg)}</span>
                        <span className="forecast-card__temp-label">avg temp</span>
                      </div>
                      <ChevronDown
                        className={`forecast-card__chevron ${isOpen ? "rotate-180" : ""}`}
                      />
                    </div>
                  </button>
                  {isOpen ? (
                    <div className="forecast-card__details">
                      <div className="forecast-card__details-header">{temperatureTagline(day.temp_avg)}</div>
                      <p>{describeTemperature(day.temp_avg)}</p>
                      <div className="forecast-card__details-footer">
                        <span>{actionHint(day.temp_avg)}</span>
                      </div>
                    </div>
                  ) : null}
                </article>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
