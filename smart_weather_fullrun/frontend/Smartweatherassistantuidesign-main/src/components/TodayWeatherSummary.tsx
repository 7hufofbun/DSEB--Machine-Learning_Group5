import { useEffect, useMemo, useState, useId } from "react";
import { format, parseISO } from "date-fns";
import {
  Cloud,
  CloudRain,
  Droplets,
  Eye,
  Gauge,
  MapPin,
  Moon,
  MoonStar,
  Sun,
  SunDim,
  Sunrise,
  Sunset,
  Thermometer,
  Umbrella,
  Wind,
} from "lucide-react";
import { apiGet, apiPost } from "../lib/api";
import { formatTemperature, formatTemperatureValue } from "../lib/format";
import "./TodayWeatherSummary.css";

interface NowResponse {
  temperature?: number;
  humidity?: number;
  conditions?: string;
  wind?: string;
  feels_like?: number;
  location?: string;
  updated_at?: string;
}

interface ForecastResponseItem {
  day: string;
  temp_avg: number;
  temp_min: number;
  temp_max: number;
  condition: string;
  precipChance: number;
}

export interface TodayWeather {
  name: string;
  address: string;
  datetime: string;
  temp: number;
  tempmax: number;
  tempmin: number;
  feelslike: number;
  humidity: number;
  precipprob: number;
  precip: number;
  preciptype?: string[];
  windspeed: number;
  winddir: number;
  uvindex: number;
  cloudcover: number;
  visibility: number;
  sunrise: string;
  sunset: string;
  moonphase: number;
  conditions: string;
  description: string;
  icon: string;
  severerisk?: number;
}

export interface TodayWeatherSummaryProps {
  weather: TodayWeather | null;
  onDetailClick?: () => void;
  isLoading?: boolean;
}

export const mockTodayWeather: TodayWeather = {
  name: "Hanoi",
  address: "Hoan Kiem District, Hanoi",
  datetime: "2025-11-15",
  temp: 28,
  tempmax: 32,
  tempmin: 24,
  feelslike: 31,
  humidity: 78,
  precipprob: 60,
  precip: 8,
  preciptype: ["rain", "thunderstorm"],
  windspeed: 18,
  winddir: 135,
  uvindex: 8,
  cloudcover: 70,
  visibility: 10,
  sunrise: "05:32",
  sunset: "17:15",
  moonphase: 0.62,
  conditions: "Rain showers and thunderstorm",
  description:
    "Mostly cloudy with showers and thunderstorms likely in the afternoon. Strong wind gusts possible.",
  icon: "rain",
  severerisk: 72,
};

const windDirections = [
  "N",
  "NNE",
  "NE",
  "ENE",
  "E",
  "ESE",
  "SE",
  "SSE",
  "S",
  "SSW",
  "SW",
  "WSW",
  "W",
  "WNW",
  "NW",
  "NNW",
];

const parseWindSpeed = (value?: string): number | undefined => {
  if (!value) return undefined;
  const normalized = value.replace(",", ".");
  const match = normalized.match(/([\d.]+)/);
  if (!match) return undefined;
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const inferWindDirection = (text?: string): number | undefined => {
  if (!text) return undefined;
  const upper = text.toUpperCase();
  const compass: Record<string, number> = {
    N: 0,
    NE: 45,
    E: 90,
    SE: 135,
    S: 180,
    SW: 225,
    W: 270,
    NW: 315,
  };
  for (const [dir, deg] of Object.entries(compass)) {
    if (upper.includes(dir)) return deg;
  }
  return undefined;
};

const inferWeatherIcon = (condition?: string): string => {
  if (!condition) return "cloud";
  const lowered = condition.toLowerCase();
  if (lowered.includes("thunder")) return "storm";
  if (lowered.includes("rain")) return "rain";
  if (lowered.includes("cloud")) return "cloud";
  if (lowered.includes("sun") || lowered.includes("clear")) return "clear";
  return "cloud";
};

const inferPrecipType = (condition?: string): string[] | undefined => {
  if (!condition) return undefined;
  const lowered = condition.toLowerCase();
  const arr: string[] = [];
  if (lowered.includes("rain")) arr.push("rain");
  if (lowered.includes("drizzle")) arr.push("drizzle");
  if (lowered.includes("storm")) arr.push("thunderstorm");
  if (lowered.includes("hail")) arr.push("hail");
  return arr.length ? arr : undefined;
};

const cloudCoverFromCondition = (condition?: string): number | undefined => {
  if (!condition) return undefined;
  const lowered = condition.toLowerCase();
  if (lowered.includes("clear")) return 20;
  if (lowered.includes("mostly sunny")) return 30;
  if (lowered.includes("partly")) return 50;
  if (lowered.includes("cloud")) return 70;
  return 80;
};

const compassDirection = (degrees: number): string => {
  const index = Math.round(degrees / 22.5) % 16;
  return windDirections[index < 0 ? index + 16 : index];
};

const fullWindDirection = (abbr?: string): string | undefined => {
  if (!abbr) return undefined;
  const mapping: Record<string, string> = {
    N: "North",
    NNE: "North North East",
    NE: "North East",
    ENE: "East North East",
    E: "East",
    ESE: "East South East",
    SE: "South East",
    SSE: "South South East",
    S: "South",
    SSW: "South South West",
    SW: "South West",
    WSW: "West South West",
    W: "West",
    WNW: "West North West",
    NW: "North West",
    NNW: "North North West",
  };
  return mapping[abbr] ?? abbr;
};

const uvBadge = (uv: number) => {
  if (uv >= 11) return { text: "Extreme", className: "today-card__highlight-badge--purple", description: "Seek shade immediately" };
  if (uv >= 8) return { text: "Very High", className: "today-card__highlight-badge--red", description: "Avoid midday sun" };
  if (uv >= 6) return { text: "High", className: "today-card__highlight-badge--orange", description: "Limit midday exposure" };
  if (uv >= 3) return { text: "Moderate", className: "today-card__highlight-badge--yellow", description: "Use sun protection" };
  return { text: "Low", className: "today-card__highlight-badge--emerald", description: "Minimal risk" };
};

const moonPhaseInfo = (phase: number) => {
  const baseClass = "today-card__chip-icon today-card__astro-icon today-card__astro-icon--moon";
  if (phase <= 0.1 || phase >= 0.9) {
    return { label: "New Moon", icon: <Moon className={baseClass} /> };
  }
  if (Math.abs(phase - 0.25) <= 0.05) {
    return { label: "First Quarter", icon: <MoonStar className={`${baseClass} today-card__astro-icon--moon-quarter`} /> };
  }
  if (Math.abs(phase - 0.5) <= 0.05) {
    return {
      label: "Full Moon",
      icon: <Moon className={`${baseClass} today-card__astro-icon--moon-full`} />,
    };
  }
  if (Math.abs(phase - 0.75) <= 0.05) {
    return {
      label: "Last Quarter",
      icon: <MoonStar className={`${baseClass} today-card__astro-icon--moon-quarter`} style={{ transform: "rotate(180deg)" }} />,
    };
  }
  if (phase < 0.5) {
    return {
      label: "Waxing Gibbous",
      icon: <Moon className={`${baseClass} today-card__astro-icon--moon-waxing`} style={{ transform: "rotate(45deg)" }} />,
    };
  }
  return {
    label: "Waning Gibbous",
    icon: <Moon className={`${baseClass} today-card__astro-icon--moon-waning`} style={{ transform: "rotate(-45deg)" }} />,
  };
};

const weatherIcon = (code: string, className = "today-card__icon") => {
  const normalized = code.toLowerCase();
  const iconClass = `${className}`;
  if (normalized.includes("storm")) {
    return <CloudRain className={iconClass} strokeWidth={1.5} />;
  }
  if (normalized.includes("rain")) {
    return <CloudRain className={iconClass} strokeWidth={1.5} />;
  }
  if (normalized.includes("cloud")) {
    return <Cloud className={iconClass} strokeWidth={1.5} />;
  }
  return <Sun className={iconClass} strokeWidth={1.5} />;
};

const skeletonBlocks = Array.from({ length: 6 });

const humidityDescription = (humidity: number) => {
  if (humidity >= 80) return "Tropical moisture";
  if (humidity >= 60) return "Humid air";
  if (humidity >= 40) return "Comfortable levels";
  return "Dry air";
};

const precipDescription = (chance: number, types?: string[]) => {
  const formattedTypes = types?.join(" & ");
  if (chance >= 70) return formattedTypes ? `Likely ${formattedTypes}` : "Likely showers";
  if (chance >= 40) return formattedTypes ? `Possible ${formattedTypes}` : "Scattered showers";
  return "Low chance of rain";
};

const windDescription = (speed: number) => {
  if (speed >= 40) return "Strong winds";
  if (speed >= 25) return "Breezy conditions";
  if (speed >= 10) return "Light breeze";
  return "Calm winds";
};

const cloudDescription = (cover: number) => {
  if (cover >= 80) return "Overcast skies";
  if (cover >= 60) return "Mostly cloudy";
  if (cover >= 35) return "Partly cloudy";
  return "Mostly clear";
};

const visibilityDescription = (distance: number) => {
  if (distance >= 10) return "Clear views";
  if (distance >= 6) return "Light haze";
  if (distance >= 3) return "Limited visibility";
  return "Very low visibility";
};

type HighlightVariant = "radial";

type RadialTheme = "sky" | "indigo" | "slate" | "rose";

interface HighlightItem {
  icon: JSX.Element;
  label: string;
  value: string;
  title: string;
  badge?: string;
  badgeClass?: string;
  description?: string;
  detail?: string;
  variant?: HighlightVariant;
  percent?: number;
  radialTheme?: RadialTheme;
}

const RADIAL_GRADIENTS: Record<RadialTheme, { start: string; end: string }> = {
  sky: { start: "#bae6fd", end: "#0284c7" },
  indigo: { start: "#ddd6fe", end: "#4f46e5" },
  slate: { start: "#e2e8f0", end: "#1f2937" },
  rose: { start: "#fecdd3", end: "#be123c" },
};

const clampPercent = (value?: number) => {
  if (typeof value !== "number" || Number.isNaN(value)) return undefined;
  return Math.max(0, Math.min(100, Math.round(value)));
};

const severeRiskDescription = (risk: number) => {
  if (risk >= 80) return "High risk: monitor official warnings";
  if (risk >= 60) return "Elevated risk: stay weather-aware";
  if (risk >= 40) return "Moderate risk: scattered impacts possible";
  return "Low risk for severe weather";
};

function RadialProgress({ percent, label, theme }: { percent: number; label: string; theme: RadialTheme }) {
  const id = useId().replace(/:/g, "");
  const safe = Math.max(0, Math.min(100, percent));
  const radius = 45;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - safe / 100);
  const gradientKey = `today-card-radial-${theme}-${id}`;
  const gradient = RADIAL_GRADIENTS[theme];

  return (
    <div className={`today-card__highlight-radial-chart today-card__highlight-radial-chart--${theme}`}>
      <svg viewBox="0 0 100 100" aria-hidden focusable="false">
        <defs>
          <linearGradient id={gradientKey} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={gradient.start} />
            <stop offset="100%" stopColor={gradient.end} />
          </linearGradient>
        </defs>
        <circle className="today-card__highlight-radial-track" cx="50" cy="50" r="45" strokeWidth="8" />
        <circle
          className="today-card__highlight-radial-progress"
          cx="50"
          cy="50"
          r="45"
          strokeWidth="8"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={dashOffset}
          stroke={`url(#${gradientKey})`}
        />
      </svg>
      <div className="today-card__highlight-radial-value">{label}</div>
    </div>
  );
}

const highlightItems = (data: TodayWeather): HighlightItem[] => {
  const uv = uvBadge(data.uvindex);
  const windDirection = compassDirection(data.winddir);
  const windDirectionFull = fullWindDirection(windDirection);
  const humidityPercent = clampPercent(data.humidity) ?? 0;
  const precipPercent = clampPercent(data.precipprob) ?? 0;
  const cloudPercent = clampPercent(data.cloudcover) ?? 0;
  const severePercent = clampPercent(data.severerisk);

  const items: HighlightItem[] = [
    {
      icon: <Droplets className="today-card__highlight-icon today-card__highlight-icon--sky" />,
      label: "Humidity",
      value: `${humidityPercent}%`,
      title: "Relative humidity",
      description: humidityDescription(data.humidity),
      variant: "radial",
      percent: humidityPercent,
      radialTheme: "sky",
    },
    {
      icon: <Umbrella className="today-card__highlight-icon today-card__highlight-icon--indigo" />,
      label: "Chance of Rain",
      value: `${precipPercent}%`,
      title: `Expected precipitation chance (${data.precip.toFixed(1)} mm)`,
      description: precipDescription(data.precipprob, data.preciptype),
      detail: data.precip > 0 ? `${data.precip.toFixed(1)} mm forecast` : undefined,
      variant: "radial",
      percent: precipPercent,
      radialTheme: "indigo",
    },
    {
      icon: <Wind className="today-card__highlight-icon today-card__highlight-icon--emerald" />,
      label: "Wind",
      value: `${Math.round(data.windspeed)} km/h`,
      title: windDirectionFull ? `Wind coming from the ${windDirectionFull.toLowerCase()}` : "Wind speed",
      badge: windDirectionFull,
      badgeClass: "today-card__highlight-badge--emerald",
      description: windDescription(data.windspeed),
    },
    {
      icon: <SunDim className="today-card__highlight-icon today-card__highlight-icon--orange" />,
      label: "UV Index",
      value: `${data.uvindex}`,
      title: "Current UV index",
      badge: uv.text.toUpperCase(),
      badgeClass: uv.className,
      description: uv.description,
    },
    {
      icon: <Cloud className="today-card__highlight-icon today-card__highlight-icon--slate" />,
      label: "Cloud Cover",
      value: `${cloudPercent}%`,
      title: "Percentage of sky covered by clouds",
      description: cloudDescription(data.cloudcover),
      variant: "radial",
      percent: cloudPercent,
      radialTheme: "slate",
    },
    {
      icon: <Eye className="today-card__highlight-icon today-card__highlight-icon--blue" />,
      label: "Visibility",
      value: `${data.visibility.toFixed(1)} km`,
      title: "Horizontal visibility",
      description: visibilityDescription(data.visibility),
    },
  ];

  if (typeof severePercent === "number") {
    items.push({
      icon: <Gauge className="today-card__highlight-icon today-card__highlight-icon--orange" />,
      label: "Severe Risk",
      value: `${severePercent}%`,
      title: "Potential severe weather impact",
      description: severeRiskDescription(severePercent),
      variant: "radial",
      percent: severePercent,
      radialTheme: "rose",
    });
  }

  return items;
};

const getHourFromDatetime = (value?: string): number | undefined => {
  if (!value) return undefined;
  const parsed = Date.parse(value);
  if (!Number.isNaN(parsed)) {
    return new Date(parsed).getHours();
  }
  try {
    return parseISO(value).getHours();
  } catch (error) {
    return undefined;
  }
};

const resolveCardTheme = (icon?: string, conditions?: string, datetime?: string) => {
  const descriptor = `${icon ?? ""} ${conditions ?? ""}`.toLowerCase();
  const hour = getHourFromDatetime(datetime);
  const isNight = typeof hour === "number" ? hour >= 19 || hour < 6 : false;

  const isStorm = descriptor.includes("storm") || descriptor.includes("thunder");
  const isRain = descriptor.includes("rain") || descriptor.includes("drizzle") || descriptor.includes("shower");
  const isCloud = descriptor.includes("cloud") || descriptor.includes("overcast") || descriptor.includes("fog");
  const isClear = descriptor.includes("sun") || descriptor.includes("clear") || descriptor.includes("fair");

  if (isStorm) return isNight ? "today-card--theme-night-storm" : "today-card--theme-day-storm";
  if (isRain) return isNight ? "today-card--theme-night-rain" : "today-card--theme-day-rain";
  if (isCloud) return isNight ? "today-card--theme-night-cloud" : "today-card--theme-day-cloud";
  if (isClear) return isNight ? "today-card--theme-night-clear" : "today-card--theme-day-clear";
  return isNight ? "today-card--theme-night-cloud" : "today-card--theme-day-cloud";
};

const formatLongDate = (isoDate: string) => {
  try {
    return format(parseISO(isoDate), "EEEE, MMM d, h:mm a");
  } catch (error) {
    return "";
  }
};

const formatUpdatedLabel = (value?: string) => {
  if (!value) return "Updated just now";
  const time = Date.parse(value);
  if (Number.isNaN(time)) return `Updated ${value}`;

  const diffMs = Date.now() - time;
  const minutes = Math.round(diffMs / 60000);

  if (minutes <= 1) return "Updated just now";
  if (minutes < 60) return `Updated ${minutes} minutes ago`;

  const hours = Math.round(minutes / 60);
  if (hours < 24) return `Updated ${hours} hour${hours === 1 ? "" : "s"} ago`;

  const days = Math.round(hours / 24);
  return `Updated ${days} day${days === 1 ? "" : "s"} ago`;
};

const formatAddress = (value: string) => value.replace(/HCMC/gi, "Ho Chi Minh City");

const renderSkeleton = () => (
  <div className="today-skeleton">
    <div className="today-skeleton__bar" style={{ width: "120px", height: "14px" }} />
    <div className="today-skeleton__bar" style={{ width: "220px", height: "18px" }} />
    <div className="today-skeleton__bar" style={{ width: "160px", height: "12px" }} />

    <div className="today-skeleton__hero">
      <div className="today-skeleton__hero-left">
        <div className="today-skeleton__bar" style={{ width: "140px", height: "60px" }} />
        <div className="today-skeleton__bar" style={{ width: "180px", height: "14px" }} />
        <div className="today-skeleton__hero-chips">
          <div className="today-skeleton__chip" />
          <div className="today-skeleton__chip" />
          <div className="today-skeleton__chip" />
        </div>
      </div>
      <div className="today-skeleton__icon" />
    </div>

    <div className="today-skeleton__grid">
      {skeletonBlocks.map((_, index) => (
        <div key={index} className="today-skeleton__tile" />
      ))}
    </div>

    <div className="today-card__astro">
      <div className="today-skeleton__bar" style={{ width: "90px", height: "12px" }} />
      <div className="today-skeleton__bar" style={{ width: "90px", height: "12px" }} />
      <div className="today-skeleton__bar" style={{ width: "110px", height: "12px" }} />
    </div>

    <div className="today-skeleton__bar" style={{ width: "100%", height: "14px" }} />

    <div className="today-card__footer">
      <div className="today-skeleton__bar" style={{ width: "160px", height: "12px" }} />
      <div className="today-skeleton__bar" style={{ width: "130px", height: "36px", borderRadius: "999px" }} />
    </div>
  </div>
);

export function TodayWeatherSummary({ weather, onDetailClick, isLoading }: TodayWeatherSummaryProps) {
  const data = weather ?? mockTodayWeather;
  const shouldRenderSkeleton = !weather && isLoading !== false;

  const highlights = useMemo(() => highlightItems(data), [data]);
  const moon = useMemo(() => moonPhaseInfo(data.moonphase), [data.moonphase]);
  const showDetailButton = Boolean(onDetailClick);
  const formattedDate = formatLongDate(data.datetime);
  const updatedLabel = formatUpdatedLabel(data.datetime);
  const cardThemeClass = resolveCardTheme(data.icon, data.conditions, data.datetime);

  if (shouldRenderSkeleton) return renderSkeleton();

  return (
    <div className={["today-card", cardThemeClass].filter(Boolean).join(" ")}>
      <header className="today-card__header">
        <div className="today-card__location" title="Current location">
          <MapPin className="today-card__location-icon" />
          <span>{formatAddress(data.address)}</span>
        </div>
        <div className="today-card__meta">
          <span>{updatedLabel}</span>
          {formattedDate ? <span>•</span> : null}
          {formattedDate ? <span>{formattedDate}</span> : null}
        </div>
      </header>

      <div className="today-card__hero">
        <div className="today-card__hero-left">
          <div className="today-card__temp" title="Current temperature">
            <span className="today-card__temp-value">{formatTemperatureValue(data.temp)}</span>
            <span className="today-card__temp-unit">°C</span>
          </div>
          <div className="today-card__feels" title="Apparent temperature">
            <Thermometer className="today-card__feels-icon" />
            <span>
              Feels like {formatTemperature(data.feelslike)}
            </span>
          </div>
          <div className="today-card__high-low">
            <span>High {formatTemperature(data.tempmax, { unit: "°" })}</span>
            <span className="today-card__high-low-separator" aria-hidden>
              •
            </span>
            <span>Low {formatTemperature(data.tempmin, { unit: "°" })}</span>
          </div>
        </div>
        <div className="today-card__hero-icon" aria-hidden>
          {weatherIcon(data.icon)}
        </div>
      </div>

      <div className="today-card__divider" aria-hidden />

      <div className="today-card__highlights">
        {highlights.map(item => {
          const highlightClass = [
            "today-card__highlight",
            item.variant === "radial" ? "today-card__highlight--radial" : "",
          ]
            .filter(Boolean)
            .join(" ");

          return (
            <div key={item.label} className={highlightClass} title={item.title}>
              <div className="today-card__highlight-header">
                <div className="today-card__highlight-header-left">
                  {item.icon}
                  <span>{item.label}</span>
                </div>
                {item.badge ? (
                  <span className={`today-card__highlight-badge ${item.badgeClass ?? ""}`.trim()}>{item.badge}</span>
                ) : null}
              </div>

              {item.variant === "radial" && typeof item.percent === "number" ? (
                <div className="today-card__highlight-radial">
                  <RadialProgress
                    percent={item.percent}
                    label={item.value}
                    theme={item.radialTheme ?? "sky"}
                  />
                  <div className="today-card__highlight-radial-copy">
                    {item.description ? (
                      <p className="today-card__highlight-description">{item.description}</p>
                    ) : null}
                    {item.detail ? <p className="today-card__highlight-detail">{item.detail}</p> : null}
                  </div>
                </div>
              ) : (
                <>
                  <p className="today-card__highlight-value">{item.value}</p>
                  {item.description ? (
                    <p className="today-card__highlight-description">{item.description}</p>
                  ) : null}
                  {item.detail ? <p className="today-card__highlight-detail">{item.detail}</p> : null}
                </>
              )}
            </div>
          );
        })}
      </div>

      <div className="today-card__astro">
        <div className="today-card__astro-item" title="Sunrise">
          <Sunrise className="today-card__astro-icon today-card__astro-icon--sunrise" />
          <span className="today-card__astro-value">{data.sunrise}</span>
        </div>
        <div className="today-card__astro-item" title="Sunset">
          <Sunset className="today-card__astro-icon today-card__astro-icon--sunset" />
          <span className="today-card__astro-value">{data.sunset}</span>
        </div>
        <div className="today-card__astro-item" title="Moon phase">
          {moon.icon}
          <span className="today-card__astro-value today-card__astro-value--moon">{moon.label}</span>
        </div>
      </div>
      {showDetailButton ? (
        <div className="today-card__actions">
          <button type="button" onClick={onDetailClick} className="today-card__button">
            View Details
          </button>
        </div>
      ) : null}
    </div>
  );
}

export function TodayWeatherSummaryPreview() {
  return (
    <TodayWeatherSummary
      weather={mockTodayWeather}
      onDetailClick={() => {}}
      isLoading={false}
    />
  );
}

export function TodayWeatherSummaryContainer({ onDetailClick }: { onDetailClick?: () => void }) {
  const [weather, setWeather] = useState<TodayWeather | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const [nowData, forecastData] = await Promise.all([
          apiGet<NowResponse>("/now").catch(() => undefined),
          apiPost<ForecastResponseItem[]>("/forecast_detailed").catch(() => undefined),
        ]);

        if (cancelled) return;

        const primary = forecastData?.[0];
        let merged: TodayWeather = { ...mockTodayWeather };

        if (nowData) {
          merged = {
            ...merged,
            name: nowData.location ?? merged.name,
            address: nowData.location ?? merged.address,
            datetime: nowData.updated_at ?? merged.datetime,
            temp: nowData.temperature ?? merged.temp,
            feelslike: nowData.feels_like ?? nowData.temperature ?? merged.feelslike,
            humidity: nowData.humidity ?? merged.humidity,
            conditions: nowData.conditions ?? merged.conditions,
            windspeed: parseWindSpeed(nowData.wind) ?? merged.windspeed,
            winddir: inferWindDirection(nowData.wind) ?? merged.winddir,
          };
        }

        if (primary) {
          const precipChance = primary.precipChance ?? merged.precipprob;
          merged = {
            ...merged,
            tempmax: primary.temp_max ?? merged.tempmax,
            tempmin: primary.temp_min ?? merged.tempmin,
            precipprob: precipChance,
            precip: Number(((precipChance / 100) * 5).toFixed(1)),
            icon: inferWeatherIcon(primary.condition),
            description: primary.condition ?? merged.description,
            preciptype: inferPrecipType(primary.condition) ?? merged.preciptype,
            cloudcover: cloudCoverFromCondition(primary.condition) ?? merged.cloudcover,
            severerisk: Math.min(100, Math.round((precipChance ?? 0) * 1.2)),
          };
        }

        setWeather(merged);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <TodayWeatherSummary
      weather={weather}
      onDetailClick={onDetailClick}
      isLoading={loading}
    />
  );
}
